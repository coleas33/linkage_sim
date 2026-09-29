# Payload Weights and Gravity-Assist Readout (Track 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Named, draggable weights whose gravity load is broken down per weight into shares of the actuator's required force and power, with a helping/hurting and motoring/braking readout on the canvas and in the plots.

**Architecture:** Weights stay point masses on links, so the solver and inertia model are unchanged. Tasks 1–2 give them identity and an id-addressed, one-undo-step editing API. Tasks 3–4 add a pure gravity-power breakdown module and store its per-sample series in the sweep. Tasks 5–9 add selection, drag and drop, plots, placement and the canvas readout. Task 10 finishes the docs and the hands-on checklist.

**Tech Stack:** Rust 1.89 (`linkage-sim-rs/`), egui/eframe 0.32, egui_plot, nalgebra; `scripts/gate.sh` (cargo test --all, clippy, WASM check); Git Bash on Windows.

**Spec:** `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md` (Track 2; accepted deviations recorded in the spec on 2026-09-29).

## Global Constraints

- Execute in the worktree `/c/Users/Cole/source/repos/linkage_simulation-payload` on branch `agent/payload-track2`, created from `main` in the Setup step below. Never work in the main checkout.
- After each task's commit, run `cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh`; it must print `GATE PASS`. If the run rewrote `docs/chebyshev_lambda/*.png`, restore them with `git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/` and never commit them.
- Repo invariants (`docs/ai/02-system.yaml`): blueprint edits go through `push_undo` then `rebuild` (prefer `AppState::mutate_and_rebuild`); every sweep channel has `angles_deg.len()` entries (NaN rows on failure); SI units internally; docs change in the same commit as the code they describe.
- **Reviews:** every task gets a fresh-context review before the next task starts. Tasks 1, 3 and 4 (which masses reach the physics, the breakdown math, the sweep integration) are `risk: physics`: review them with the `fbd-math-reviewer` agent type, and the user must view those diffs before merge. Other tasks use the general code reviewer.
- **Reference implementation:** branch `draft/payload-linear` holds one verified commit per task (Task 1 `b29154d`, 2 `535d027`, 3 `c0e4709`, 4 `352f43a`, 5 `4a26188`, 6 `d617e1a`, 7 `c3224d2`, 8 `59b8ac7`, 9 `b6b0499`, 10 `5305e04`). Every code block in this plan matches it (614 blocks checked, 0 mismatches), and each passed the full gate. Reviewers may compare a task's resulting tree with its reference commit.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
  ```

## Review Focus

Inputs a person will hit that default-path tests could miss; each is pinned by the named task's tests:

1. **Stored-force actuators** (a stored drive force instead of sizing mode): the breakdown equals the sizing-mode breakdown at every sample (Task 4).
2. **Solver failures and non-Grashof gaps:** every breakdown series keeps `angles_deg.len()` entries, with NaN rows (Task 4).
3. **Weights the loader rejects** (on ground, mass ≤ 0, non-finite): skipped identically by the loader, the live mass sync and the breakdown sources (Tasks 1, 3).
4. **Near a stroke reversal:** force share is a gap, power share stays finite, and helping/hurting stays defined (Tasks 3, 4, 9).
5. **Editing a weighted link's mass without a rebuild:** the breakdown sum invariant still holds (Task 4).

## Setup

- [ ] **Create the execution worktree and confirm a green baseline**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation worktree add /c/Users/Cole/source/repos/linkage_simulation-payload -b agent/payload-track2 main
git -C /c/Users/Cole/source/repos/linkage_simulation-payload rev-parse --abbrev-ref HEAD
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected: `agent/payload-track2`, then `GATE PASS` with the lib test total of `main` before Task 1 (729 passed).

---

### Task 1: Weight identity in the data model

**Files:**
- Create: `linkage-sim-rs/src/gui/test_support.rs` (`#[cfg(test)]` helpers shared by the GUI module tests; starts with `sorted_link_ids`)
- Modify: `linkage-sim-rs/src/gui/mod.rs` (register `#[cfg(test)] mod test_support;`)
- Modify: `linkage-sim-rs/src/io/schema.rs` (import line 5; `PointMassJson` lines 211-216; new id helpers between `BodyJson` and `JointJson`; new test module at the end of the file)
- Modify: `linkage-sim-rs/src/io/from_json.rs` (new "Point-mass validation" section above the "JSON -> Mechanism" banner; the point-mass loop in `load_mechanism_unbuilt_from_json`; new test module at the end of the file)
- Modify: `linkage-sim-rs/src/io/mod.rs` (the `pub use from_json::...` line)
- Modify: `linkage-sim-rs/src/gui/state/file_io.rs` (`load_from_json_str`: the parse near the top and the tail just before `Ok(())`)
- Modify: `linkage-sim-rs/src/gui/state/blueprint_ops.rs` (`sync_live_mass_props`, which is main's BL-024 code, and `add_point_mass`)
- Modify: `linkage-sim-rs/src/gui/state/tests.rs` (import, the BL-023 and BL-025 fixtures, `assert_round_trip_preserves_mass`, four new identity tests above the `// -- BL-025` banner, one new BL-024 test)
- Modify: `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`, `docs/architecture/ARCHITECTURE.md`

**Interfaces:**
- Consumes: nothing from earlier tasks. Existing code this task builds on:
  - `Body::add_point_mass(&mut self, mass: f64, local_pos: Vector2<f64>)` (`core/body.rs`).
  - `AppState::{load_from_json_str, serialize_to_json_string, generate_share_url, add_point_mass, set_body_mass}` and `AppState::sync_live_mass_props(&mut self, body_id: &str)` (private, BL-024, already on main).
  - `gui::state::file_io::decode_mechanism_from_url`.
  - Test helpers already in `gui/state/tests.rs`: `built_mass_props`, `fresh_built_mass_props`, `blueprint_mass_props`, `assert_mass_props_match`, `assert_round_trip_preserves_mass`, `four_bar_with_point_masses`, `four_bar_with_one_point_mass`.
- Produces (all later tasks address weights through these):
  - `pub struct PointMassJson { #[serde(default)] pub id: String, #[serde(default, skip_serializing_if = "Option::is_none")] pub label: Option<String>, pub mass: f64, pub local_pos: [f64; 2] }` deriving `Debug, Clone, PartialEq, Serialize, Deserialize` (`crate::io::PointMassJson`).
  - `pub fn assign_point_mass_ids(bodies: &mut HashMap<String, BodyJson>)` (`crate::io::assign_point_mass_ids`).
  - `pub fn next_point_mass_id(bodies: &HashMap<String, BodyJson>) -> String` (`crate::io::next_point_mass_id`).
  - `pub(crate) fn is_blank_point_mass_id(id: &str) -> bool` (`crate::io::schema::is_blank_point_mass_id`).
  - `pub fn point_mass_skip_reason(body_id: &str, pm: &PointMassJson) -> Option<&'static str>` (`crate::io::point_mass_skip_reason`).
  - `pub fn point_mass_warnings(json: &MechanismJson) -> Vec<String>` (`crate::io::point_mass_warnings`).
  - `pub(crate) fn apply_point_masses(body: &mut Body, point_masses: &[PointMassJson])` (`crate::io::from_json::apply_point_masses`): the only place blueprint weights reach the physics.
  - `#[cfg(test)] pub(crate) fn sorted_link_ids(state: &AppState) -> Vec<String>` (`crate::gui::test_support::sorted_link_ids`): non-ground blueprint body ids, sorted. It lives here, in the first task that needs it, so no task adds a private copy and deletes it later.
  - `AppState::add_point_mass` keeps its signature (`-> ()`) and now gives each new weight `next_point_mass_id`.
  - `AppState::load_from_json_str` assigns ids and appends `point_mass_warnings` to `error_log`, setting `show_error_panel = true`.

Why `apply_point_masses` exists: main's BL-024 fix made `sync_live_mass_props` fold every blueprint point mass into the live body after a base-mass edit. Once the loader starts skipping weights (on ground, non-positive or non-finite), that loop would keep applying weights a rebuild drops, and "live equals a fresh build" would break. The loader and `sync_live_mass_props` therefore both call `apply_point_masses`.

Convention in this file: `Add after it:` means insert the block after the `Find` text, separated from it by one blank line.

- [ ] **Step 1: Write the failing tests**

Create `linkage-sim-rs/src/gui/test_support.rs`:

```rust
//! Helpers shared by the tests of the GUI modules.

use crate::core::state::GROUND_ID;
use crate::gui::state::AppState;

/// Non-ground blueprint body (link) ids, sorted: a deterministic fixture
/// order, since `blueprint.bodies` is a `HashMap`.
pub(crate) fn sorted_link_ids(state: &AppState) -> Vec<String> {
    let mut ids: Vec<String> = state
        .blueprint
        .as_ref()
        .expect("blueprint")
        .bodies
        .keys()
        .filter(|k| k.as_str() != GROUND_ID)
        .cloned()
        .collect();
    ids.sort();
    ids
}
```

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
pub mod sweep;
mod theme;
```

Replace with:

```rust
pub mod sweep;
#[cfg(test)]
mod test_support;
mod theme;
```

Find in `linkage-sim-rs/src/io/schema.rs` (the last lines of the file):

```rust
    RevoluteDriver {
        body_i: String,
        body_j: String,
        #[serde(default)]
        note: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
}
```

Add after it:

```rust
#[cfg(test)]
mod point_mass_id_tests {
    use super::*;

    /// A body whose point masses carry exactly `ids` (1 kg each, at the origin).
    fn body_with(ids: &[&str]) -> BodyJson {
        BodyJson {
            attachment_points: HashMap::new(),
            mass: 1.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: ids
                .iter()
                .map(|id| PointMassJson {
                    id: id.to_string(),
                    label: None,
                    mass: 1.0,
                    local_pos: [0.0, 0.0],
                })
                .collect(),
            label: None,
            geometry: None,
        }
    }

    fn bodies(spec: &[(&str, &[&str])]) -> HashMap<String, BodyJson> {
        spec.iter().map(|(b, ids)| (b.to_string(), body_with(ids))).collect()
    }

    fn ids_of(bodies: &HashMap<String, BodyJson>, body: &str) -> Vec<String> {
        bodies[body].point_masses.iter().map(|pm| pm.id.clone()).collect()
    }

    #[test]
    fn point_mass_without_id_or_label_deserializes_with_defaults() {
        let pm: PointMassJson =
            serde_json::from_str(r#"{"mass": 2.5, "local_pos": [0.1, -0.2]}"#).unwrap();
        assert_eq!(
            pm,
            PointMassJson { id: String::new(), label: None, mass: 2.5, local_pos: [0.1, -0.2] }
        );
    }

    #[test]
    fn point_mass_serializes_id_always_and_label_only_when_set() {
        let mut pm = PointMassJson { id: "W1".into(), label: None, mass: 2.5, local_pos: [0.1, -0.2] };
        let v = serde_json::to_value(&pm).unwrap();
        assert_eq!(v["id"], "W1");
        assert!(v.get("label").is_none(), "absent label must not be written: {v}");

        pm.label = Some("Robot torso".into());
        let v = serde_json::to_value(&pm).unwrap();
        assert_eq!(v["label"], "Robot torso");
        let back: PointMassJson = serde_json::from_value(v).unwrap();
        assert_eq!(back, pm);
    }

    #[test]
    fn assign_fills_blank_ids_in_sorted_body_then_list_order() {
        let mut b = bodies(&[("rocker", &["", ""]), ("coupler", &["", "   "])]);
        assign_point_mass_ids(&mut b);
        assert_eq!(ids_of(&b, "coupler"), ["W1", "W2"]);
        assert_eq!(ids_of(&b, "rocker"), ["W3", "W4"]);
    }

    #[test]
    fn assign_keeps_unique_ids_and_hands_out_smallest_unused_number() {
        // "W01" is a distinct string from "W1", so it does not reserve W1.
        let mut b = bodies(&[("a", &["W2", ""]), ("b", &["", "payload", "W01"])]);
        assign_point_mass_ids(&mut b);
        assert_eq!(ids_of(&b, "a"), ["W2", "W1"]);
        assert_eq!(ids_of(&b, "b"), ["W3", "payload", "W01"]);
    }

    #[test]
    fn assign_renumbers_only_repeats_after_the_first_occurrence() {
        let mut b = bodies(&[("a", &["W1", "W1", "x"]), ("b", &["W1", "x"])]);
        assign_point_mass_ids(&mut b);
        assert_eq!(ids_of(&b, "a"), ["W1", "W2", "x"]);
        assert_eq!(ids_of(&b, "b"), ["W3", "W4"]);
    }

    #[test]
    fn assign_is_idempotent_and_leaves_bodies_without_masses_alone() {
        let mut b = bodies(&[("a", &["", "W1"]), ("empty", &[])]);
        assign_point_mass_ids(&mut b);
        assert_eq!(ids_of(&b, "a"), ["W2", "W1"]);
        assert!(ids_of(&b, "empty").is_empty());

        assign_point_mass_ids(&mut b);
        assert_eq!(ids_of(&b, "a"), ["W2", "W1"], "second pass must change nothing");
    }

    #[test]
    fn next_point_mass_id_is_smallest_unused_across_all_bodies() {
        assert_eq!(next_point_mass_id(&HashMap::new()), "W1");
        assert_eq!(next_point_mass_id(&bodies(&[("a", &[])])), "W1");
        assert_eq!(next_point_mass_id(&bodies(&[("a", &["W1", "W3"]), ("b", &["W2"])])), "W4");
        assert_eq!(next_point_mass_id(&bodies(&[("a", &["W1", "W3"]), ("b", &["payload"])])), "W2");
    }
}
```

Find in `linkage-sim-rs/src/io/from_json.rs` (the last lines of the file):

```rust
pub fn load_mechanism(json_str: &str) -> Result<Mechanism, SerializationError> {
    let mut mech = load_mechanism_unbuilt(json_str)?;
    mech.build()
        .map_err(|e| SerializationError::Build(e.to_string()))?;
    Ok(mech)
}
```

Add after it:

```rust
#[cfg(test)]
mod point_mass_validation_tests {
    use super::*;

    /// Ground plus one 2 kg bar (CG at x = 0.5 m, Izz 0.1). No joints are
    /// needed to check composite mass properties.
    fn ground_and_bar() -> MechanismJson {
        serde_json::from_str(
            r#"{
                "schema_version": "1.1.0",
                "bodies": {
                    "ground": {"attachment_points": {"O": [0.0, 0.0]},
                               "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                    "bar": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                            "mass": 2.0, "cg_local": [0.5, 0.0], "izz_cg": 0.1}
                },
                "joints": {}
            }"#,
        )
        .unwrap()
    }

    fn pm(id: &str, mass: f64, x: f64, y: f64) -> PointMassJson {
        PointMassJson { id: id.to_string(), label: None, mass, local_pos: [x, y] }
    }

    /// (mass, cg, izz) of body `id` after loading `json`.
    fn mass_props(json: &MechanismJson, id: &str) -> (f64, [f64; 2], f64) {
        let mech = load_mechanism_unbuilt_from_json(json).unwrap();
        let b = &mech.bodies()[id];
        (b.mass, [b.cg_local.x, b.cg_local.y], b.izz_cg)
    }

    #[test]
    fn valid_point_mass_is_applied_and_not_reported() {
        let mut json = ground_and_bar();
        json.bodies.get_mut("bar").unwrap().point_masses.push(pm("W1", 2.0, 1.0, 0.0));
        let (m, cg, izz) = mass_props(&json, "bar");
        assert!((m - 4.0).abs() < 1e-12, "mass {m}");
        assert!((cg[0] - 0.75).abs() < 1e-12 && cg[1].abs() < 1e-12, "cg {cg:?}");
        // 0.1 + 2 kg * (0.25 m)^2 + 2 kg * (0.25 m)^2
        assert!((izz - 0.35).abs() < 1e-12, "izz {izz}");
        assert!(point_mass_warnings(&json).is_empty());
    }

    #[test]
    fn point_mass_on_ground_is_skipped_and_reported() {
        let mut json = ground_and_bar();
        json.bodies.get_mut("ground").unwrap().point_masses.push(pm("W1", 5.0, 0.3, 0.0));
        let (m, _, izz) = mass_props(&json, "ground");
        assert_eq!(m, 0.0, "ground must stay massless");
        assert_eq!(izz, 0.0);
        let w = point_mass_warnings(&json);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("'W1'") && w[0].contains("'ground'"), "{w:?}");
    }

    #[test]
    fn non_positive_or_non_finite_point_masses_are_skipped_and_reported() {
        let base = mass_props(&ground_and_bar(), "bar");
        let bad = [
            pm("neg", -1.0, 1.0, 0.0), // used to be subtracted from the bar
            pm("zero", 0.0, 1.0, 0.0),
            pm("nan", f64::NAN, 1.0, 0.0),
            pm("inf", f64::INFINITY, 1.0, 0.0),
            pm("far", 1.0, f64::INFINITY, 0.0),
            pm("nanpos", 1.0, 0.0, f64::NAN),
        ];
        for p in &bad {
            let mut json = ground_and_bar();
            json.bodies.get_mut("bar").unwrap().point_masses.push(p.clone());
            assert_eq!(mass_props(&json, "bar"), base, "{}: must not change the bar", p.id);
            let w = point_mass_warnings(&json);
            assert_eq!(w.len(), 1, "{}: {w:?}", p.id);
            assert!(w[0].contains(&format!("'{}'", p.id)) && w[0].contains("'bar'"), "{w:?}");
        }
    }

    #[test]
    fn invalid_point_mass_does_not_block_valid_ones_on_the_same_body() {
        let mut json = ground_and_bar();
        json.bodies
            .get_mut("bar")
            .unwrap()
            .point_masses
            .extend([pm("bad", -3.0, 1.0, 0.0), pm("good", 2.0, 1.0, 0.0)]);
        let (m, cg, _) = mass_props(&json, "bar");
        assert!((m - 4.0).abs() < 1e-12 && (cg[0] - 0.75).abs() < 1e-12, "{m} {cg:?}");
    }

    #[test]
    fn warnings_are_sorted_by_body_and_name_blank_ids_by_list_position() {
        let mut json = ground_and_bar();
        json.bodies
            .get_mut("bar")
            .unwrap()
            .point_masses
            .extend([pm("ok", 1.0, 0.0, 0.0), pm("", -2.0, 0.0, 0.0)]);
        json.bodies.get_mut("ground").unwrap().point_masses.push(pm("G", 1.0, 0.0, 0.0));
        let w = point_mass_warnings(&json);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("#2") && w[0].contains("'bar'"), "{w:?}");
        assert!(w[1].contains("'G'") && w[1].contains("'ground'"), "{w:?}");
    }
}
```

In `linkage-sim-rs/src/gui/state/tests.rs`, import the shared helper and use it in both point-mass fixtures (this removes the duplicated "sorted non-ground ids" block).

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
    use super::blueprint_ops::{joint_body_ids, seed_q_by_body_id};
    use crate::io::JointJson;
```

Replace with:

```rust
    use super::blueprint_ops::{joint_body_ids, seed_q_by_body_id};
    use crate::gui::test_support::sorted_link_ids;
    use crate::io::JointJson;
```

Find in `linkage-sim-rs/src/gui/state/tests.rs` (head of `four_bar_with_point_masses`):

```rust
        state.load_sample(SampleMechanism::FourBar);

        let mut ids: Vec<String> = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .keys()
            .filter(|k| k.as_str() != GROUND_ID)
            .cloned()
            .collect();
        ids.sort();

        let base = blueprint_mass_props(&state);
```

Replace with:

```rust
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);

        let base = blueprint_mass_props(&state);
```

In `assert_round_trip_preserves_mass`, make the per-point-mass loop also compare ids and labels.

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
            for (s, d) in sb.point_masses.iter().zip(&db.point_masses) {
                assert!((s.mass - d.mass).abs() < 1e-12, "{what}: body '{id}' point-mass mass");
```

Replace with:

```rust
            for (s, d) in sb.point_masses.iter().zip(&db.point_masses) {
                assert_eq!(s.id, d.id, "{what}: body '{id}' point-mass id");
                assert_eq!(s.label, d.label, "{what}: body '{id}' point-mass label");
                assert!((s.mass - d.mass).abs() < 1e-12, "{what}: body '{id}' point-mass mass");
```

Find in `linkage-sim-rs/src/gui/state/tests.rs` (head of `four_bar_with_one_point_mass`, BL-025 section):

```rust
        state.load_sample(SampleMechanism::FourBar);
        let mut ids: Vec<String> = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .keys()
            .filter(|k| k.as_str() != GROUND_ID)
            .cloned()
            .collect();
        ids.sort();
        state.add_point_mass(&ids[0], 2.0, [0.03, 0.02]);
```

Replace with:

```rust
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);
        state.add_point_mass(&ids[0], 2.0, [0.03, 0.02]);
```

Now the four identity tests. They go directly after the BL-023 file save/load test, above the `// -- BL-025` banner.

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        assert_round_trip_preserves_mass("file save/load", &src, &dst);
    }
```

Add after it:

```rust
    // ── Payload weights: point-mass identity (ids, labels, loader validation) ──

    /// Ids of the point masses on `body_id`, in list order.
    fn point_mass_ids(state: &AppState, body_id: &str) -> Vec<String> {
        state.blueprint.as_ref().unwrap().bodies[body_id]
            .point_masses
            .iter()
            .map(|pm| pm.id.clone())
            .collect()
    }

    #[test]
    fn saved_file_carries_point_mass_ids() {
        let (src, heavy) = four_bar_with_point_masses();
        let links = sorted_link_ids(&src);
        assert_eq!(point_mass_ids(&src, &heavy), ["W1", "W2"], "add_point_mass assigns W<n>");
        assert_eq!(point_mass_ids(&src, &links[1]), ["W3"]);

        let v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        assert_eq!(v["bodies"][heavy.as_str()]["point_masses"][1]["id"], "W2");
        assert_eq!(v["bodies"][links[1].as_str()]["point_masses"][0]["id"], "W3");
    }

    #[test]
    fn old_file_without_point_mass_ids_gets_ids_on_load() {
        let (src, heavy) = four_bar_with_point_masses();
        let links = sorted_link_ids(&src);
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        for body in v["bodies"].as_object_mut().unwrap().values_mut() {
            if let Some(pms) = body.get_mut("point_masses").and_then(|p| p.as_array_mut()) {
                for pm in pms {
                    let pm = pm.as_object_mut().unwrap();
                    pm.remove("id");
                    pm.remove("label");
                }
            }
        }

        let mut dst = AppState::default();
        dst.load_from_json_str(&v.to_string()).expect("old file should load");
        // Bodies sorted by id, masses in list order.
        assert_eq!(point_mass_ids(&dst, &heavy), ["W1", "W2"]);
        assert_eq!(point_mass_ids(&dst, &links[1]), ["W3"]);
        // Masses and positions intact and applied exactly once.
        assert_round_trip_preserves_mass("old file", &src, &dst);
        assert!(dst.error_log.is_empty(), "a valid file reports nothing: {:?}", dst.error_log);
    }

    #[test]
    fn point_mass_ids_and_labels_round_trip_through_save_and_share() {
        let (mut src, heavy) = four_bar_with_point_masses();
        src.blueprint.as_mut().unwrap().bodies.get_mut(&heavy).unwrap().point_masses[1].label =
            Some("Robot torso".to_string());

        let mut dst = AppState::default();
        dst.load_from_json_str(&src.serialize_to_json_string().unwrap()).unwrap();
        assert_round_trip_preserves_mass("save/load with label", &src, &dst);

        let url = src.generate_share_url().expect("generate_share_url failed");
        let encoded = url.split("?m=").nth(1).expect("share URL missing ?m=");
        let json = super::file_io::decode_mechanism_from_url(encoded).expect("decode failed");
        let mut dst2 = AppState::default();
        dst2.load_from_json_str(&json).unwrap();
        assert_round_trip_preserves_mass("share URL with label", &src, &dst2);
    }

    #[test]
    fn load_skips_and_reports_invalid_point_masses() {
        let (src, body, other) = four_bar_with_one_point_mass();
        let base = blueprint_mass_props(&src);
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][GROUND_ID]["point_masses"] =
            serde_json::json!([{"id": "on_ground", "mass": 5.0, "local_pos": [0.0, 0.0]}]);
        v["bodies"][other.as_str()]["point_masses"] = serde_json::json!([
            {"id": "neg", "mass": -3.0, "local_pos": [0.01, 0.0]},
            {"mass": 0.0, "local_pos": [0.02, 0.0]}
        ]);

        let mut dst = AppState::default();
        dst.load_from_json_str(&v.to_string()).expect("a file with bad weights still loads");

        // Skipped: ground stays massless and `other` keeps its base mass; the
        // valid 2 kg weight on `body` is still applied.
        let built = built_mass_props(&dst);
        assert_eq!(built[GROUND_ID].mass, 0.0);
        assert!((built[&other].mass - base[&other].mass).abs() < 1e-12, "{}", built[&other].mass);
        assert!((built[&body].mass - (base[&body].mass + 2.0)).abs() < 1e-12);

        // Reported once each, naming the weight and body.
        assert_eq!(dst.error_log.len(), 3, "{:?}", dst.error_log);
        assert!(dst.error_log.iter().any(|w| w.contains("'on_ground'")), "{:?}", dst.error_log);
        assert!(dst.error_log.iter().any(|w| w.contains("'neg'")), "{:?}", dst.error_log);
        assert!(dst.show_error_panel, "load warnings must be visible");

        // Kept in the blueprint (a save writes them back unchanged) and
        // addressable: the blank id got the smallest unused W<n>.
        assert_eq!(point_mass_ids(&dst, GROUND_ID), ["on_ground"]);
        assert_eq!(point_mass_ids(&dst, &other), ["neg", "W2"]);
    }
```

The last new test pins the BL-024 conflict: the no-rebuild mass sync must apply the loader's skip rules. It goes after `set_body_izz_keeps_point_masses_in_live_mechanism_bl024`.

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        // Izz does not move mass or CG.
        assert!((live[&body].mass - mass_before).abs() < 1e-12);
        assert!((live[&body].cg[0] - cg_before[0]).abs() < 1e-12);
        assert!((live[&body].cg[1] - cg_before[1]).abs() < 1e-12);
    }
```

Add after it:

```rust
    /// The no-rebuild mass sync must apply the loader's skip rules too: a
    /// weight the loader rejected (here a negative mass, kept in the
    /// blueprint) stays out of the live body after a base-mass edit.
    #[test]
    fn set_body_mass_skips_point_masses_the_loader_rejects() {
        let (src, body, _) = four_bar_with_one_point_mass(); // valid 2 kg "W1"
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][body.as_str()]["point_masses"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"id": "neg", "mass": -3.0, "local_pos": [0.05, 0.0]}));
        let mut state = AppState::default();
        state.load_from_json_str(&v.to_string()).expect("a file with a bad weight still loads");
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body].point_masses.len(),
            2,
            "setup: the rejected weight stays in the blueprint"
        );

        let new_base = state.blueprint.as_ref().unwrap().bodies[&body].mass + 1.0;
        state.set_body_mass(&body, new_base);

        let live = built_mass_props(&state);
        assert!(
            (live[&body].mass - (new_base + 2.0)).abs() < 1e-12,
            "live mass should be new base + the valid 2 kg weight only, got {}",
            live[&body].mass
        );
        assert_mass_props_match("live vs fresh build with a rejected weight", &fresh_built_mass_props(&state), &live);
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib point_mass
```

Expected: the lib test target does not compile (34 errors in this tree), so no test runs. The errors come in four families, one per missing piece:
- ``error[E0560]: struct `schema::PointMassJson` has no field named `id` `` (and the same for `label`): the new fields do not exist yet.
- ``error[E0609]: no field `id` on type `&schema::PointMassJson` `` (and `label`): `assert_round_trip_preserves_mass` and the identity tests read them.
- ``error[E0425]: cannot find function `assign_point_mass_ids` in this scope`` (also `point_mass_warnings`, `next_point_mass_id`): the new `io` functions do not exist yet.
- ``error[E0369]: binary operation `==` cannot be applied to type `schema::PointMassJson` ``: `PartialEq` is not derived yet.

The `gui/test_support.rs` helper itself compiles; it is not among the errors.

- [ ] **Step 3: Implement**

All edits below are in the execution worktree `/c/Users/Cole/source/repos/linkage_simulation-payload`; paths are relative to its root.

**3a. `io/schema.rs`: the data model.**

Find in `linkage-sim-rs/src/io/schema.rs`:

```rust
use std::collections::HashMap;
```

Replace with:

```rust
use std::collections::{HashMap, HashSet};
```

Find in `linkage-sim-rs/src/io/schema.rs`:

```rust
/// A point mass attached to a body at a local position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointMassJson {
    pub mass: f64,
    pub local_pos: [f64; 2],
}
```

Replace with:

```rust
/// A point mass (a "weight") attached to a body at a local position.
///
/// `id` addresses the weight in the GUI and in analyses; it is unique across
/// the mechanism. Files written before ids existed omit it (it deserializes
/// as empty) and [`assign_point_mass_ids`] fills it in on load. `label` is an
/// optional display name; the id is shown when it is absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointMassJson {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub mass: f64,
    pub local_pos: [f64; 2],
}
```

The id helpers. `assign_point_mass_ids` walks bodies sorted by id and masses in list order, so which duplicate is "first" and which `W<n>` numbers get handed out never depend on `HashMap` iteration order. Non-blank ids keep their first occurrence; blank ids and later repeats are renumbered.

Find in `linkage-sim-rs/src/io/schema.rs`:

```rust
    /// Optional visual geometry for rendering and force zone overlap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<crate::core::body::BodyGeometry>,
}
```

Add after it:

```rust
/// True when `id` cannot address a weight (empty or whitespace only).
pub(crate) fn is_blank_point_mass_id(id: &str) -> bool {
    id.trim().is_empty()
}

/// The smallest `W<n>` (n >= 1) that is not in `used`.
fn smallest_unused_point_mass_id(used: &HashSet<String>) -> String {
    let mut n: usize = 1;
    loop {
        let id = format!("W{n}");
        if !used.contains(&id) {
            return id;
        }
        n += 1;
    }
}

/// Give every point mass in `bodies` a unique, non-blank id.
///
/// Non-blank ids are kept at their first occurrence. Blank ids, and every
/// repeat of an id after its first occurrence, are replaced by `W<n>` with
/// `n` the smallest positive integer whose `W<n>` is not used anywhere in the
/// mechanism. The walk order is deterministic (bodies sorted by id, point
/// masses in list order), so "first occurrence" and the numbers handed out do
/// not depend on `HashMap` iteration order. Idempotent.
pub fn assign_point_mass_ids(bodies: &mut HashMap<String, BodyJson>) {
    let mut body_ids: Vec<String> = bodies.keys().cloned().collect();
    body_ids.sort();

    // Pass 1: keep the first occurrence of every non-blank id.
    let mut used: HashSet<String> = HashSet::new();
    let mut needs_id: Vec<(String, usize)> = Vec::new();
    for body_id in &body_ids {
        for (i, pm) in bodies[body_id].point_masses.iter().enumerate() {
            if is_blank_point_mass_id(&pm.id) || !used.insert(pm.id.clone()) {
                needs_id.push((body_id.clone(), i));
            }
        }
    }

    // Pass 2: hand out the smallest unused W<n>, in the same order.
    for (body_id, i) in needs_id {
        let id = smallest_unused_point_mass_id(&used);
        used.insert(id.clone());
        if let Some(body) = bodies.get_mut(&body_id) {
            body.point_masses[i].id = id;
        }
    }
}

/// The id for a new point mass: the smallest `W<n>` not used by any point
/// mass in `bodies`.
pub fn next_point_mass_id(bodies: &HashMap<String, BodyJson>) -> String {
    let used: HashSet<String> = bodies
        .values()
        .flat_map(|b| b.point_masses.iter().map(|pm| pm.id.clone()))
        .collect();
    smallest_unused_point_mass_id(&used)
}
```

**3b. `io/from_json.rs`: validation, and the single entry point into the physics.** `point_mass_skip_reason` is the one definition of "the loader rejects this weight"; `point_mass_warnings` (for the error panel) and `apply_point_masses` (for the physics) both call it, so the list the user is warned about is exactly the list that is skipped.

Find in `linkage-sim-rs/src/io/from_json.rs`:

```rust
        ForceElement::LinearActuator(a) => Some((a.body_a.clone(), a.body_b.clone())),
        _ => None,
    }
}
```

Add after it:

```rust
// ---------------------------------------------------------------------------
// Point-mass validation
// ---------------------------------------------------------------------------

/// Why the loader does not apply point mass `pm` on body `body_id`, or `None`
/// when it is applied.
///
/// A weight on ground is rejected: ground is fixed and massless, so the weight
/// could never load the mechanism and accepting it would hide a modelling
/// mistake. A mass that is not a positive finite number, or a non-finite
/// position, would corrupt the composite mass/CG/Izz (`Body::add_point_mass`
/// subtracts a negative mass). Shared by the loader, [`point_mass_warnings`],
/// the GUI editing API and anything that lists the weights the physics sees.
pub fn point_mass_skip_reason(body_id: &str, pm: &PointMassJson) -> Option<&'static str> {
    if body_id == GROUND_ID {
        Some("point masses on ground are not supported")
    } else if !(pm.mass.is_finite() && pm.mass > 0.0) {
        Some("mass must be a positive, finite number of kg")
    } else if !(pm.local_pos[0].is_finite() && pm.local_pos[1].is_finite()) {
        Some("position must be finite")
    } else {
        None
    }
}

/// One human-readable warning per point mass the loader skips (see
/// [`point_mass_skip_reason`]): bodies sorted by id, masses in list order.
/// A weight without an id is named by its 1-based list position (`#2`).
/// Empty when every point mass is applied.
pub fn point_mass_warnings(json: &MechanismJson) -> Vec<String> {
    let mut body_ids: Vec<&String> = json.bodies.keys().collect();
    body_ids.sort();
    let mut warnings = Vec::new();
    for body_id in body_ids {
        for (i, pm) in json.bodies[body_id].point_masses.iter().enumerate() {
            let Some(reason) = point_mass_skip_reason(body_id, pm) else { continue };
            let name = if is_blank_point_mass_id(&pm.id) {
                format!("#{}", i + 1)
            } else {
                format!("'{}'", pm.id)
            };
            warnings.push(format!(
                "Point mass {name} on body '{body_id}' skipped: {reason} (mass {} kg)",
                pm.mass
            ));
        }
    }
    warnings
}

/// Fold `point_masses` into `body`'s composite mass, CG and Izz, skipping the
/// ones [`point_mass_skip_reason`] rejects (named `body.id`).
///
/// The single place blueprint weights reach the physics: the loader and the
/// GUI's no-rebuild mass sync (`AppState::sync_live_mass_props`) both call it,
/// so a live body always equals a fresh build of the blueprint.
pub(crate) fn apply_point_masses(body: &mut Body, point_masses: &[PointMassJson]) {
    for pm in point_masses {
        if let Some(reason) = point_mass_skip_reason(&body.id, pm) {
            log::warn!("Skipping point mass '{}' on body '{}': {}", pm.id, body.id, reason);
            continue;
        }
        body.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));
    }
}
```

The loader applies the weights to the `Body` before `mech.add_body` (which only stores the body), so the body is made mutable and the old inline loop goes away. `is_blank_point_mass_id` and `PointMassJson` reach this file through the existing `use super::schema::*;`; `GROUND_ID`, `Body` and `Vector2` are already imported.

Find in `linkage-sim-rs/src/io/from_json.rs`:

```rust
        let body = Body {
            id: body_id.clone(),
```

Replace with:

```rust
        let mut body = Body {
            id: body_id.clone(),
```

Find in `linkage-sim-rs/src/io/from_json.rs`:

```rust
        mech.add_body(body)
            .map_err(|e| SerializationError::Build(e.to_string()))?;

        // Apply point masses to update composite mass/CG/Izz
        for pm in &body_json.point_masses {
            if let Some(body_mut) = mech.body_mut(body_id) {
                body_mut.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));
            }
        }
```

Replace with:

```rust
        // Apply point masses to update composite mass/CG/Izz, skipping the
        // ones the loader rejects (listed for the user by `point_mass_warnings`).
        apply_point_masses(&mut body, &body_json.point_masses);
        mech.add_body(body)
            .map_err(|e| SerializationError::Build(e.to_string()))?;
```

**3c. `io/mod.rs`: re-export the two public functions.**

Find in `linkage-sim-rs/src/io/mod.rs`:

```rust
pub use from_json::{load_mechanism, load_mechanism_unbuilt, load_mechanism_unbuilt_from_json};
```

Replace with:

```rust
pub use from_json::{
    load_mechanism, load_mechanism_unbuilt, load_mechanism_unbuilt_from_json,
    point_mass_skip_reason, point_mass_warnings,
};
```

**3d. `gui/state/file_io.rs`: `load_from_json_str` is the single funnel for file, autosave, share-URL, template and recent loads.** It assigns ids before the blueprint is stored (so old files get `W<n>` ids and stay addressable), and it reports skipped weights. The skipped weights stay in the blueprint so a save writes the file back unchanged; the error panel is left open because callers set a status-bar message after a successful load, which would otherwise hide the warning.

Find in `linkage-sim-rs/src/gui/state/file_io.rs`:

```rust
        let json_struct: crate::io::MechanismJson =
            serde_json::from_str(json_str).map_err(|e| e.to_string())?;

        // Restore GUI sweep state (mode + trajectory severity) if present.
```

Replace with:

```rust
        let mut json_struct: crate::io::MechanismJson =
            serde_json::from_str(json_str).map_err(|e| e.to_string())?;
        // Weights are addressed by id. Files written before ids existed (or
        // hand-edited with blanks/duplicates) get W<n> ids here — the single
        // funnel for file, autosave, share-URL, template and recent loads.
        crate::io::assign_point_mass_ids(&mut json_struct.bodies);
        let point_mass_warnings = crate::io::point_mass_warnings(&json_struct);

        // Restore GUI sweep state (mode + trajectory severity) if present.
```

Find in `linkage-sim-rs/src/gui/state/file_io.rs`:

```rust
        self.recompute_driver_display_offset();
        self.dirty = false;
        self.autosave_timer = 0.0;

        Ok(())
    }

    /// Load a mechanism from a JSON file, solve at t=0, and update all state.
```

Replace with:

```rust
        self.recompute_driver_display_offset();
        self.dirty = false;
        self.autosave_timer = 0.0;

        // Weights the loader skipped (on ground, non-positive mass, ...) stay
        // in the blueprint so a save writes the file back unchanged; tell the
        // user they are not in the physics. The error panel survives the
        // status-bar message callers set after a successful load.
        if !point_mass_warnings.is_empty() {
            self.error_log.extend(point_mass_warnings);
            self.show_error_panel = true;
        }

        Ok(())
    }

    /// Load a mechanism from a JSON file, solve at t=0, and update all state.
```

**3e. `gui/state/blueprint_ops.rs`: one entry point for weights (the BL-024 conflict), and ids on add.** `sync_live_mass_props` keeps building a composite `Body` from the blueprint base values and now folds the weights in through `apply_point_masses`, exactly as the loader does.

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
    /// Uses `Body::add_point_mass`, the same composite math the loader applies
    /// (`io/from_json.rs`), so the live body equals a fresh build. The composite
    /// CG and Izz depend on the base mass, so base edits must re-derive all
    /// three rather than patching a single field.
```

Replace with:

```rust
    /// Uses `io::from_json::apply_point_masses`, the loader's own composite
    /// math and skip rules (weights the loader rejects stay out), so the live
    /// body equals a fresh build. The composite CG and Izz depend on the base
    /// mass, so base edits must re-derive all three rather than patching a
    /// single field.
```

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
        for pm in &bp_body.point_masses {
            composite.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));
        }
```

Replace with:

```rust
        crate::io::from_json::apply_point_masses(&mut composite, &bp_body.point_masses);
```

`add_point_mass` builds a `PointMassJson` literal that no longer compiles without `id` and `label`; it takes the next free id before it borrows the body.

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
    /// Pushes undo, appends the point mass, and rebuilds (which recomputes
    /// composite mass, CG, and Izz via parallel axis theorem).
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.push(crate::io::PointMassJson {
            mass,
            local_pos,
        });
```

Replace with:

```rust
    /// Pushes undo, appends the point mass with the next free `W<n>` id, and
    /// rebuilds (which recomputes composite mass, CG, and Izz via parallel
    /// axis theorem).
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let id = crate::io::next_point_mass_id(&bp.bodies);
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.push(crate::io::PointMassJson {
            id,
            label: None,
            mass,
            local_pos,
        });
```

**3f. Docs (the repo rule: every code change ships its doc update).**

Find in `docs/ai/02-system.yaml`:

```yaml
    sign. The Python reference solvers still omit Q_v (BL-028) (BL-027).
```

Replace with:

```yaml
    sign. The Python reference solvers still omit Q_v (BL-028) (BL-027).
  - point_mass_ids_unique_per_mechanism — every blueprint point mass
    (weight) has a non-blank id unique across the mechanism
    (PointMassJson { id, label?, mass, local_pos }). AppState::load_from_json_str
    calls io::assign_point_mass_ids (blank/duplicate ids -> smallest unused
    W<n>, bodies sorted by id, list order); add_point_mass uses
    io::next_point_mass_id. Address weights by (body_id, weight_id), never
    by list index.
  - point_mass_loader_validation — the loader skips (does not apply) a point
    mass on ground, a mass that is not a positive finite number, or a
    non-finite position (io::point_mass_skip_reason). Skipped weights stay in
    the blueprint (saved back unchanged); load_from_json_str pushes
    io::point_mass_warnings into error_log and opens the error panel.
    io::from_json::apply_point_masses is the only place weights reach the
    physics (the loader and sync_live_mass_props both call it), so a
    no-rebuild mass edit skips exactly what a rebuild skips.
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - undo.rs
    - dxf_import.rs
```

Replace with:

```yaml
    - undo.rs
    - dxf_import.rs
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids; extend it instead of re-implementing fixtures per module)
```

Find in `docs/architecture/ARCHITECTURE.md`:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body.
```

Add after it:

```markdown
In the Rust JSON schema a point mass ("weight") lives in its body's `point_masses` list as `{ "id": "W1", "label": "Robot torso", "mass": 50.0, "local_pos": [0.3, 0.0] }` (`label` optional). The `id` is unique across the mechanism; files written before ids existed load unchanged and get `W<n>` ids on load (smallest unused number; bodies sorted by id, list order). The loader skips, and reports in the error panel, a point mass on ground, a mass that is not a positive finite number, and a non-finite position; skipped weights stay in the file so a save writes them back unchanged.
```

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Add after it:

```markdown
## 2026-09-29 — Payload weights Task 1: point-mass ids, labels and loader validation
- Spec: `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`
  (Track 2, section 1 "Data model").
- `io/schema.rs`: `PointMassJson` gains `id: String` (`#[serde(default)]`) and
  `label: Option<String>` (skipped when absent). `assign_point_mass_ids` gives
  blank/duplicate ids the smallest unused `W<n>` (bodies sorted by id, list
  order; idempotent); `next_point_mass_id` for new weights.
- `io/from_json.rs`: `point_mass_skip_reason` rejects weights on ground,
  non-positive/non-finite masses (a negative mass used to be subtracted) and
  non-finite positions; the loader skips them; `point_mass_warnings` lists them.
  `apply_point_masses` is the one place weights reach the physics: the loader
  and BL-024's `sync_live_mass_props` both call it, so a base-mass edit skips
  the same weights a rebuild skips.
- `gui/state/file_io.rs::load_from_json_str` assigns ids and pushes warnings
  into `error_log` (error panel opens). `add_point_mass` assigns the next id.
- New `gui/test_support.rs` (`#[cfg(test)]`): `sorted_link_ids`, the shared
  "sorted non-ground link ids" fixture helper for GUI module tests.
- Tests: `io::schema::point_mass_id_tests`,
  `io::from_json::point_mass_validation_tests`, and in `gui/state/tests.rs`
  old-file load gets ids, ids + labels round-trip through save and share URL,
  invalid weights skipped and reported, and
  `set_body_mass_skips_point_masses_the_loader_rejects` (mutation: applying
  every weight in `sync_live_mass_props` turns it red); BL-023 round-trip
  helper now compares ids and labels too.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib point_mass
```

Expected: `test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 729 filtered out`. That is the 17 new tests (7 in `io::schema::point_mass_id_tests`, 5 in `io::from_json::point_mass_validation_tests`, and in `gui::state::tests` `saved_file_carries_point_mass_ids`, `old_file_without_point_mass_ids_gets_ids_on_load`, `point_mass_ids_and_labels_round_trip_through_save_and_share`, `load_skips_and_reports_invalid_point_masses` and `set_body_mass_skips_point_masses_the_loader_rejects`) plus the 21 existing tests whose names contain `point_mass` (`core::body`, BL-023, BL-024, BL-025 and the basic add/remove/persist tests). The BL-023 `point_masses_not_double_counted_*` tests passing shows the loader change still applies each weight exactly once.

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: `test result: ok. 767 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` (750 before this task plus the 17 new tests).

Mutation check for the new BL-024 test (run once, by hand). Stage `blueprint_ops.rs` so it can be restored, put the old apply-every-weight loop back in `sync_live_mass_props`, and confirm the test goes red:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload
git add linkage-sim-rs/src/gui/state/blueprint_ops.rs
sed -i 's|        crate::io::from_json::apply_point_masses(&mut composite, &bp_body.point_masses);|        for pm in \&bp_body.point_masses {\n            composite.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));\n        }|' linkage-sim-rs/src/gui/state/blueprint_ops.rs
git diff --stat -- linkage-sim-rs/src/gui/state/blueprint_ops.rs
cd linkage-sim-rs && cargo test --lib set_body_mass_skips_point_masses_the_loader_rejects
```

Expected: `git diff --stat` lists `blueprint_ops.rs` as changed (the mutation, relative to the staged copy), and the test fails with ``live mass should be new base + the valid 2 kg weight only, got 1`` (the rejected -3 kg weight is subtracted again) and `test result: FAILED. 0 passed; 1 failed`. Then restore the staged version and confirm green (`git checkout` rewrites the file, so cargo sees a fresh mtime):

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload
git checkout -- linkage-sim-rs/src/gui/state/blueprint_ops.rs
git diff --stat -- linkage-sim-rs/src/gui/state/blueprint_ops.rs
cd linkage-sim-rs && cargo test --lib set_body_mass_skips_point_masses_the_loader_rejects
```

Expected: `git diff --stat` prints nothing (the working file equals the staged copy again), and `test result: ok. 1 passed; 0 failed`.

Then the full gate:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected: the last line is `GATE PASS` (`cargo test --all`, `cargo clippy --all-targets` and the WASM check all pass). The test run rewrites the PNGs in `docs/chebyshev_lambda/`; restore them so they stay out of the commit:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/gui/test_support.rs linkage-sim-rs/src/gui/mod.rs linkage-sim-rs/src/io/schema.rs linkage-sim-rs/src/io/from_json.rs linkage-sim-rs/src/io/mod.rs linkage-sim-rs/src/gui/state/file_io.rs linkage-sim-rs/src/gui/state/blueprint_ops.rs linkage-sim-rs/src/gui/state/tests.rs docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md docs/architecture/ARCHITECTURE.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 1: weight identity in the data model

PointMassJson gains id + optional label; assign_point_mass_ids /
next_point_mass_id give every weight a unique W<n> id (on load and on
add). The loader skips and reports weights on ground and non-positive or
non-finite masses/positions (point_mass_skip_reason, point_mass_warnings);
load_from_json_str surfaces the warnings in the error panel.

apply_point_masses is now the single place weights reach the physics:
the loader and BL-024's sync_live_mass_props both call it, so a base-mass
edit skips exactly the weights a rebuild skips. New #[cfg(test)]
gui/test_support.rs holds the shared sorted_link_ids fixture helper.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 2: Id-addressed weight editing API

Task 1 gave every weight (point mass) an `id` and a `label`. This task moves every weight edit off list indices. The GUI adds, moves, edits and removes a weight by `(body_id, weight_id)`. Each committed edit is exactly one validated undo step, and undo snapshots now carry the editable weight list, so undo and redo restore ids and labels instead of baking the weights into the link mass. The only visible changes are that the weight mass field no longer clamps a loaded out-of-range mass on an idle frame, and that Place Mass uses the last mass used (`last_point_mass_kg`, default 1 kg) instead of a hard-coded 1 kg.

Spec: Track 2 section 1 ("every weight is addressable by id rather than by list index") and the undo requirement behind known defect 4 (BL-025).

Design points the code follows:

- **One shared edit path.** Mass edits, label edits and same-body moves all go through the private `edit_point_mass` (clone the weight, apply the closure, reject a result the loader would skip, no-op when nothing changed, else one `mutate_and_rebuild`). Adding, reattaching and removing a weight validate the same way, so no editor can create a weight that the loader later skips (`io::point_mass_skip_reason`).
- **Validate first, then one undo entry.** `mutate_and_rebuild` pushes the undo snapshot and rebuilds. It is only reached when the edit is valid and changes the blueprint, so an invalid or no-op request never pushes an entry or rebuilds. Callers invoke each edit once per commit (drag-stop, typed value, click), never per drag frame.
- **Undo keeps the editable list.** `mechanism_to_json` bakes point masses into the composite mass and emits no list. The extracted `overlay_blueprint_point_masses` writes the blueprint's base mass/CG/Izz plus the weight list back over it. `serialize_to_json_string` (file, URL, autosave) and `take_snapshot` (undo/redo) share it, so the two writers cannot drift apart (BL-023).
- **Test fixtures take link ids from the shared helper.** Every new fixture calls `crate::gui::test_support::sorted_link_ids` (added in Task 1): `weight_clicks::setup` in `canvas/mod.rs`, `idle_frames_do_not_clamp_out_of_range_weight_masses` in `property_panel/mod.rs`, and `four_bar_with_weight` in `pending_edits.rs`. Do not re-inline the sorted non-ground key list, and do not import `GROUND_ID` just for it.

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/mod.rs` (`last_point_mass_kg` field, the two point-mass mode fields, `Default`)
- Modify: `linkage-sim-rs/src/gui/state/blueprint_ops.rs` (imports, three private helpers, the id-addressed weight API replacing the index-based one)
- Modify: `linkage-sim-rs/src/gui/state/file_io.rs` (new `overlay_blueprint_point_masses`; its call in `serialize_to_json_string`)
- Modify: `linkage-sim-rs/src/gui/state/undo_ops.rs` (`take_snapshot`)
- Modify: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (`PendingPropertyEdit` variants, `apply_pending` arms, new test module)
- Modify: `linkage-sim-rs/src/gui/property_panel/mod.rs` (Point Masses section, new test module)
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs` (Move to Link handler, Reposition handler, `handle_place_mass`)
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (new `weight_clicks` module inside `mod tests`)
- Modify: `linkage-sim-rs/src/gui/state/tests.rs` (`remove_point_mass_modifies_blueprint`, BL-023 `dst3` removals, the BL-025 section)
- Modify: `docs/ai/02-system.yaml`
- Modify: `docs/ai/05-update-tracker.md`

**Interfaces:**
- Consumes (Task 1):
  - `crate::io::PointMassJson { pub id: String, pub label: Option<String>, pub mass: f64, pub local_pos: [f64; 2] }` (derives `Debug, Clone, PartialEq, Serialize, Deserialize`)
  - `crate::io::next_point_mass_id(bodies: &HashMap<String, BodyJson>) -> String` (smallest unused `W<n>` across all bodies)
  - `crate::io::point_mass_skip_reason(body_id: &str, pm: &PointMassJson) -> Option<&'static str>` (`Some` for a weight on ground, a mass that is not positive and finite, or a non-finite position)
  - `crate::io::schema::is_blank_point_mass_id(id: &str) -> bool` (`pub(crate)`)
  - ids and labels assigned and kept by `load_from_json_str`; `crate::gui::test_support::sorted_link_ids(state: &AppState) -> Vec<String>` (`#[cfg(test)]`, `pub(crate)`)
  - existing: `AppState::mutate_and_rebuild<F: FnOnce(&mut Self)>(&mut self, op: F)` (push undo, run `op`, rebuild), `AppState::world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2]`, `undo_history.undo_count() -> usize`
  - test helpers already in `gui/state/tests.rs`: `built_mass_props`, `blueprint_mass_props`, `assert_mass_props_match`, `point_mass_ids`
- Produces:
  - `AppState::last_point_mass_kg: f64` (pub, default `1.0`; set by `add_point_mass`)
  - `AppState::reassigning_point_mass: Option<(String, String)>` and `AppState::repositioning_point_mass: Option<(String, String)>`, each `(body_id, weight_id)` (they were `(String, usize)`)
  - `pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>` (was `()`)
  - `pub fn find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson>`
  - `pub fn move_point_mass(&mut self, body_id: &str, weight_id: &str, target_body: &str, local_pos: [f64; 2]) -> bool`
  - `pub fn set_point_mass_mass(&mut self, body_id: &str, weight_id: &str, mass: f64) -> bool`
  - `pub fn set_point_mass_label(&mut self, body_id: &str, weight_id: &str, label: Option<String>) -> bool`
  - `pub fn remove_point_mass_by_id(&mut self, body_id: &str, weight_id: &str) -> bool`
  - `pub(crate) fn overlay_blueprint_point_masses(&self, json_struct: &mut crate::io::MechanismJson)`
  - Semantics: validate first; exactly one undo entry (via `mutate_and_rebuild`) per effective change. An invalid request (missing weight or body, blank id, ground target, mass not positive and finite, non-finite position) returns `false` / `None` with no undo entry and no rebuild. A no-op request returns `true` with no undo entry. `move_point_mass` with `target_body == body_id` repositions in place (list order kept); another body reattaches the weight (appended) keeping its id, label and mass. `set_point_mass_label` trims and treats a blank label as `None`. Undo and redo restore the weight list with ids and labels.
  - Removed: `remove_point_mass(body, index)`, `update_point_mass(...)`, `move_point_mass_to_body(...)`, private `point_mass_exists`.
  - `PendingPropertyEdit` (`pub(super)`): `SetPointMassMass { body_id, weight_id, mass }`, `SetPointMassPosition { body_id, weight_id, local_pos }`, `RemovePointMass { body_id, weight_id }`, `ReassignPointMass { body_id, weight_id }`, `RepositionPointMass { body_id, weight_id }` (`UpdatePointMass` removed).
  - Private helpers in `blueprint_ops.rs`: `point_mass_position`, `point_mass_mut`, `take_point_mass`, and `AppState::edit_point_mass`.

- [ ] **Step 1: Write the failing tests**

Every block in this step is a test change. The production code in Step 3 is what makes them compile and pass. Convention for the blocks below: "Add after it" inserts the new block after the Find text, separated by one blank line, unless the prose says "directly after".

**1a. `gui/state/tests.rs`: move the existing tests onto the id API.**

In `remove_point_mass_modifies_blueprint`:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        state.remove_point_mass(&body_id, 0);
```

Replace with:

```rust
        let weight_id = state.add_point_mass(&body_id, 0.5, [0.01, 0.0]).expect("add should succeed");
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        assert!(state.remove_point_mass_by_id(&body_id, &weight_id));
```


In `point_masses_not_double_counted_by_serialize_load_bl023` (the loader assigned `W1` and `W2` to the two weights, so the removals name them by id):

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        dst3.remove_point_mass(&heavy_body, 1);
        dst3.remove_point_mass(&heavy_body, 0);
```

Replace with:

```rust
        assert!(dst3.remove_point_mass_by_id(&heavy_body, "W2"));
        assert!(dst3.remove_point_mass_by_id(&heavy_body, "W1"));
```


**1b. `gui/state/tests.rs`: the BL-025 section.** The header comment and fixture now assert the id, and a new `blueprint_weights` helper snapshots every body's weight list:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
    // ── BL-025: point-mass edits are single, undoable steps ──────────────
    //
    // Undo snapshots bake point masses into the composite mass/CG/Izz and drop
    // the editable list (see BL-023 notes), so "restores prior state" is
    // asserted on the built composite properties, which is what the physics sees.

    /// Four-bar with one 2 kg point mass on the first (sorted) non-ground body.
    /// Returns the state, that body id, and a second non-ground body id.
    fn four_bar_with_one_point_mass() -> (AppState, String, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);
        state.add_point_mass(&ids[0], 2.0, [0.03, 0.02]);
        (state, ids[0].clone(), ids[1].clone())
    }
```

Replace with:

```rust
    // ── Weight editing: id-addressed, one undo step per edit (BL-025) ──────
    //
    // Undo snapshots carry the editable weight list (ids, labels, base mass),
    // so "restores prior state" is asserted on both the built composite
    // properties (what the physics sees) and the blueprint weight lists.

    /// Four-bar with one 2 kg weight "W1" at (0.03, 0.02) on the first
    /// (sorted) non-ground body. Returns the state, that body id, and a
    /// second non-ground body id.
    fn four_bar_with_one_point_mass() -> (AppState, String, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);
        assert_eq!(state.add_point_mass(&ids[0], 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        (state, ids[0].clone(), ids[1].clone())
    }

    /// Every blueprint body's weight list (ids, labels, masses, positions).
    fn blueprint_weights(state: &AppState) -> std::collections::BTreeMap<String, Vec<crate::io::PointMassJson>> {
        state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .iter()
            .map(|(id, b)| (id.clone(), b.point_masses.clone()))
            .collect()
    }
```


`assert_edit_is_one_undoable_step` now also proves that one `undo()` restores the weight lists. Its head:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
    /// Run `edit` and assert it recorded exactly one undo entry, that it
    /// really changed the built composite mass properties (guards against a
    /// vacuous test), and that a single `undo()` restores the prior composite
    /// mass properties and the prior undo depth.
    fn assert_edit_is_one_undoable_step(
        what: &str,
        state: &mut AppState,
        edit: impl FnOnce(&mut AppState),
    ) {
        let depth_before = state.undo_history.undo_count();
        let props_before = built_mass_props(state);

        edit(state);
```

Replace with:

```rust
    /// Run `edit` and assert it recorded exactly one undo entry, that it
    /// really changed the built composite mass properties (guards against a
    /// vacuous test), and that a single `undo()` restores the prior composite
    /// mass properties, the prior weight lists and the prior undo depth.
    fn assert_edit_is_one_undoable_step(
        what: &str,
        state: &mut AppState,
        edit: impl FnOnce(&mut AppState),
    ) {
        let depth_before = state.undo_history.undo_count();
        let props_before = built_mass_props(state);
        let weights_before = blueprint_weights(state);

        edit(state);
```


and its tail:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        assert_mass_props_match(&format!("{what}: after undo"), &props_before, &built_mass_props(state));
    }
```

Replace with:

```rust
        assert_mass_props_match(&format!("{what}: after undo"), &props_before, &built_mass_props(state));
        assert_eq!(blueprint_weights(state), weights_before, "{what}: undo must restore the weight lists");
    }
```


The four existing one-undo-step tests call the id API. In `point_mass_numeric_mass_edit_is_one_undo_step_bl025`:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
            s.update_point_mass(&body, 0, 5.0, [0.03, 0.02]);
```

Replace with:

```rust
            assert!(s.set_point_mass_mass(&body, "W1", 5.0));
```


In `point_mass_numeric_position_edit_is_one_undo_step_bl025`:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
            s.update_point_mass(&body, 0, 2.0, [-0.04, 0.05]);
```

Replace with:

```rust
            assert!(s.move_point_mass(&body, "W1", &body, [-0.04, 0.05]));
```


In `point_mass_reposition_is_one_undo_step_bl025`:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        // Mirrors the canvas Reposition click: world point -> body-local -> update.
        assert_edit_is_one_undoable_step("reposition", &mut state, |s| {
            let [lx, ly] = s.world_to_body_local(&body, 0.07, 0.06);
            s.update_point_mass(&body, 0, 2.0, [lx, ly]);
```

Replace with:

```rust
        // Mirrors the canvas Reposition click: world point -> body-local -> move.
        assert_edit_is_one_undoable_step("reposition", &mut state, |s| {
            let [lx, ly] = s.world_to_body_local(&body, 0.07, 0.06);
            assert!(s.move_point_mass(&body, "W1", &body, [lx, ly]));
```


In `point_mass_move_to_link_is_one_undo_step_bl025` (the weight keeps its id):

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
            s.move_point_mass_to_body(&from_body, 0, &to_body, [0.04, -0.01]);

            // The mass left the old body and landed, unchanged, on the new one.
            let bp = s.blueprint.as_ref().unwrap();
            assert!(bp.bodies[&from_body].point_masses.is_empty(), "old body should lose the point mass");
            let moved = &bp.bodies[&to_body].point_masses;
            assert_eq!(moved.len(), 1, "new body should gain exactly one point mass");
```

Replace with:

```rust
            assert!(s.move_point_mass(&from_body, "W1", &to_body, [0.04, -0.01]));

            // The weight left the old body and landed, unchanged, on the new one.
            let bp = s.blueprint.as_ref().unwrap();
            assert!(bp.bodies[&from_body].point_masses.is_empty(), "old body should lose the point mass");
            let moved = &bp.bodies[&to_body].point_masses;
            assert_eq!(moved.len(), 1, "new body should gain exactly one point mass");
            assert_eq!(moved[0].id, "W1", "the weight keeps its id");
```


**1c. `gui/state/tests.rs`: new API tests.** These cover one-undo-step for remove and add, the id and last-mass contract of `add_point_mass`, invalid input, in-place reposition versus reattach, label trimming, and undo/redo restoring the weight list. Insert them after `point_mass_move_to_link_is_one_undo_step_bl025` (before `point_mass_invalid_targets_create_no_undo_entry_bl025`):

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
            assert!((moved[0].local_pos[1] + 0.01).abs() < 1e-12);
        });
    }
```

Add after it:

```rust
    #[test]
    fn remove_point_mass_by_id_is_one_undo_step() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("remove", &mut state, |s| {
            assert!(s.remove_point_mass_by_id(&body, "W1"));
            assert!(s.find_point_mass(&body, "W1").is_none());
        });
    }

    #[test]
    fn add_point_mass_is_one_undo_step() {
        let (mut state, _, other) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("add", &mut state, |s| {
            assert_eq!(s.add_point_mass(&other, 3.0, [0.02, 0.01]).as_deref(), Some("W2"));
        });
    }

    #[test]
    fn last_point_mass_kg_defaults_to_one_kilogram() {
        assert_eq!(AppState::default().last_point_mass_kg, 1.0);
    }

    #[test]
    fn add_point_mass_returns_the_new_id_and_remembers_the_mass() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert_eq!(state.last_point_mass_kg, 2.0, "the fixture's add sets the last mass");

        assert_eq!(state.add_point_mass(&other, 7.5, [0.01, 0.0]).as_deref(), Some("W2"));
        assert_eq!(state.last_point_mass_kg, 7.5);
        assert_eq!(state.add_point_mass(&body, 0.25, [0.0, 0.01]).as_deref(), Some("W3"));
        assert_eq!(
            state.find_point_mass(&body, "W3"),
            Some(&crate::io::PointMassJson {
                id: "W3".to_string(),
                label: None,
                mass: 0.25,
                local_pos: [0.0, 0.01],
            })
        );

        // A freed number is handed out again (smallest unused).
        assert!(state.remove_point_mass_by_id(&other, "W2"));
        assert_eq!(state.add_point_mass(&other, 1.0, [0.0, 0.0]).as_deref(), Some("W2"));
    }

    #[test]
    fn add_point_mass_rejects_invalid_input_without_undo_entry() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        state.last_point_mass_kg = 4.0;
        let depth = state.undo_history.undo_count();
        let weights = blueprint_weights(&state);

        let b = body.as_str();
        for (what, target, mass, pos) in [
            ("ground", GROUND_ID, 1.0, [0.0, 0.0]),
            ("missing body", "no_such_body", 1.0, [0.0, 0.0]),
            ("zero mass", b, 0.0, [0.0, 0.0]),
            ("negative mass", b, -1.0, [0.0, 0.0]),
            ("NaN mass", b, f64::NAN, [0.0, 0.0]),
            ("infinite mass", b, f64::INFINITY, [0.0, 0.0]),
            ("NaN position", b, 1.0, [f64::NAN, 0.0]),
            ("infinite position", b, 1.0, [0.0, f64::INFINITY]),
        ] {
            assert_eq!(state.add_point_mass(target, mass, pos), None, "{what}");
        }

        assert_eq!(state.undo_history.undo_count(), depth, "rejected adds must not push undo entries");
        assert_eq!(blueprint_weights(&state), weights, "rejected adds must not change the model");
        assert_eq!(state.last_point_mass_kg, 4.0, "a rejected add must not change the last mass");
    }

    #[test]
    fn move_point_mass_on_the_same_body_repositions_in_place() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_eq!(state.add_point_mass(&body, 3.0, [0.05, 0.0]).as_deref(), Some("W2"));
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));

        assert!(state.move_point_mass(&body, "W1", &body, [-0.02, 0.04]));

        let pms = &state.blueprint.as_ref().unwrap().bodies[&body].point_masses;
        assert_eq!(pms.len(), 2);
        assert_eq!(
            pms[0],
            crate::io::PointMassJson {
                id: "W1".to_string(),
                label: Some("Torso".to_string()),
                mass: 2.0,
                local_pos: [-0.02, 0.04],
            },
            "same slot, same id/label/mass, new position"
        );
        assert_eq!(pms[1].id, "W2");
    }

    #[test]
    fn move_point_mass_to_another_body_keeps_id_label_and_mass() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));

        assert!(state.move_point_mass(&body, "W1", &other, [0.01, -0.02]));

        assert!(state.find_point_mass(&body, "W1").is_none());
        assert_eq!(
            state.find_point_mass(&other, "W1"),
            Some(&crate::io::PointMassJson {
                id: "W1".to_string(),
                label: Some("Torso".to_string()),
                mass: 2.0,
                local_pos: [0.01, -0.02],
            })
        );
    }

    #[test]
    fn set_point_mass_label_is_one_undo_step_and_trims_or_clears() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        let label = |s: &AppState| s.find_point_mass(&body, "W1").expect("W1 must survive").label.clone();
        let depth = state.undo_history.undo_count();

        assert!(state.set_point_mass_label(&body, "W1", Some("  Robot torso ".to_string())));
        assert_eq!(label(&state).as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        assert!(state.set_point_mass_label(&body, "W1", Some("   ".to_string())));
        assert_eq!(label(&state), None, "a blank label clears it");
        assert_eq!(state.undo_history.undo_count(), depth + 2);

        state.undo();
        assert_eq!(label(&state).as_deref(), Some("Robot torso"), "undo restores the label");
        state.undo();
        assert_eq!(label(&state), None);
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn undo_and_redo_restore_the_editable_weight_list() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));
        let weights_before = blueprint_weights(&state);
        let base = blueprint_mass_props(&state);
        let built_before = built_mass_props(&state);

        assert!(state.move_point_mass(&body, "W1", &other, [0.04, -0.01]));
        let weights_after = blueprint_weights(&state);
        let built_after = built_mass_props(&state);

        state.undo();
        assert_eq!(blueprint_weights(&state), weights_before, "undo puts W1 (id, label) back on its body");
        assert_mass_props_match("undo: blueprint keeps base mass", &base, &blueprint_mass_props(&state));
        assert_mass_props_match("undo: built composite", &built_before, &built_mass_props(&state));

        state.redo();
        assert_eq!(blueprint_weights(&state), weights_after, "redo re-applies the move");
        assert_mass_props_match("redo: blueprint keeps base mass", &base, &blueprint_mass_props(&state));
        assert_mass_props_match("redo: built composite", &built_after, &built_mass_props(&state));
    }
```


**1d. `gui/state/tests.rs`: the invalid-target test and two more tests.** `point_mass_invalid_targets_create_no_undo_entry_bl025` now exercises every id-API entry point with bad input and compares the weight lists. Its body:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        let props = built_mass_props(&state);

        state.remove_point_mass(&body, 1); // index past the end
        state.remove_point_mass(&body, usize::MAX);
        state.remove_point_mass(&other, 0); // body has no point masses
        state.remove_point_mass("no_such_body", 0);
        state.update_point_mass(&body, 1, 3.0, [0.0, 0.0]);
        state.update_point_mass("no_such_body", 0, 3.0, [0.0, 0.0]);
        state.move_point_mass_to_body(&body, 1, &other, [0.0, 0.0]); // bad index
        state.move_point_mass_to_body(&body, 0, "no_such_body", [0.0, 0.0]); // bad destination
```

Replace with:

```rust
        let props = built_mass_props(&state);
        let weights = blueprint_weights(&state);

        let origin = [0.0, 0.0];
        assert!(!state.move_point_mass(&body, "W9", &body, origin), "unknown id");
        assert!(!state.move_point_mass(&body, "", &body, origin), "blank id");
        assert!(!state.move_point_mass(&other, "W1", &other, origin), "weight is on another body");
        assert!(!state.move_point_mass("no_such_body", "W1", &body, origin), "missing source body");
        assert!(!state.move_point_mass(&body, "W1", GROUND_ID, origin), "ground target");
        assert!(!state.move_point_mass(&body, "W1", "no_such_body", origin), "missing target body");
        assert!(!state.move_point_mass(&body, "W1", &body, [f64::NAN, 0.0]), "NaN position");
        assert!(!state.move_point_mass(&body, "W1", &other, [0.0, f64::INFINITY]), "infinite position");
        assert!(!state.set_point_mass_mass(&body, "W1", 0.0), "zero mass");
        assert!(!state.set_point_mass_mass(&body, "W1", -2.0), "negative mass");
        assert!(!state.set_point_mass_mass(&body, "W1", f64::NAN), "NaN mass");
        assert!(!state.set_point_mass_mass(&body, "W9", 3.0), "unknown id");
        assert!(!state.set_point_mass_label(&body, "W9", Some("x".to_string())), "unknown id");
        assert!(!state.remove_point_mass_by_id(&body, "W9"), "unknown id");
        assert!(!state.remove_point_mass_by_id(&body, ""), "blank id");
        assert!(!state.remove_point_mass_by_id(&other, "W1"), "weight is on another body");
        assert!(!state.remove_point_mass_by_id("no_such_body", "W1"), "missing body");
```


and its closing assertions, followed by two new tests: no-op edits succeed without an undo entry, and weights the loader skipped (on ground, zero mass) stay addressable and can be repaired:

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        assert_mass_props_match("invalid targets leave the model untouched", &props, &built_mass_props(&state));
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body].point_masses.len(),
            1,
            "a bad destination must not drop the source point mass"
        );
    }
```

Replace with:

```rust
        assert_mass_props_match("invalid targets leave the model untouched", &props, &built_mass_props(&state));
        assert_eq!(blueprint_weights(&state), weights, "invalid targets must not drop or change the weight");
    }

    #[test]
    fn no_op_weight_edits_succeed_without_undo_entry() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        let depth = state.undo_history.undo_count();

        assert!(state.move_point_mass(&body, "W1", &body, [0.03, 0.02]), "same position");
        assert!(state.set_point_mass_mass(&body, "W1", 2.0), "same mass");
        assert!(state.set_point_mass_label(&body, "W1", None), "same (absent) label");
        assert!(state.set_point_mass_label(&body, "W1", Some("  ".to_string())), "blank = absent");

        assert_eq!(state.undo_history.undo_count(), depth, "no-op edits must not push undo entries");
    }

    #[test]
    fn weights_skipped_by_the_loader_can_be_repaired() {
        let (src, body, _) = four_bar_with_one_point_mass();
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][GROUND_ID]["point_masses"] =
            serde_json::json!([{"id": "on_ground", "mass": 5.0, "local_pos": [0.0, 0.0]}]);
        v["bodies"][body.as_str()]["point_masses"][0]["mass"] = serde_json::json!(0.0);
        let mut state = AppState::default();
        state.load_from_json_str(&v.to_string()).unwrap();
        let base = blueprint_mass_props(&state)[&body].mass;
        let built = |s: &AppState| built_mass_props(s)[&body].mass;
        assert!((built(&state) - base).abs() < 1e-12, "both weights start skipped");

        // A skipped zero-mass weight stays addressable; a valid mass applies it.
        assert!(state.set_point_mass_mass(&body, "W1", 2.0));
        assert!((built(&state) - (base + 2.0)).abs() < 1e-12);

        // A weight on ground can be moved onto a link, where it applies.
        assert!(state.move_point_mass(GROUND_ID, "on_ground", &body, [0.0, 0.0]));
        assert!((built(&state) - (base + 7.0)).abs() < 1e-12);
        assert!(state.find_point_mass(GROUND_ID, "on_ground").is_none());
    }
```


**1e. `gui/property_panel/pending_edits.rs`: `apply_pending` by weight id.** Weight edits apply by id as one undo step each, Move to Link / Reposition arm the canvas modes with the weight id, and a stale edit (the weight already moved away) is a no-op.

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
            PendingPropertyEdit::ConvertActuatorToLinearDriver { index } => {
                state.convert_actuator_to_linear_driver(index);
            }
        }
    }
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::EditorTool;
    use crate::gui::test_support::sorted_link_ids;

    /// Four-bar with one 2 kg weight on the first (sorted) link. Returns the
    /// state, that link's id, a second link's id and the weight id.
    fn four_bar_with_weight() -> (AppState, String, String, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let links = sorted_link_ids(&state);
        let weight = state.add_point_mass(&links[0], 2.0, [0.03, 0.02]).unwrap();
        (state, links[0].clone(), links[1].clone(), weight)
    }

    #[test]
    fn weight_edits_apply_by_id_as_one_undo_step_each() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassMass {
                body_id: body.clone(),
                weight_id: w.clone(),
                mass: 5.0,
            }),
        );
        assert_eq!(state.find_point_mass(&body, &w).unwrap().mass, 5.0);
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassPosition {
                body_id: body.clone(),
                weight_id: w.clone(),
                local_pos: [-0.01, 0.04],
            }),
        );
        assert_eq!(state.find_point_mass(&body, &w).unwrap().local_pos, [-0.01, 0.04]);
        assert_eq!(state.undo_history.undo_count(), depth + 2);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::RemovePointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert!(state.find_point_mass(&body, &w).is_none());
        assert_eq!(state.undo_history.undo_count(), depth + 3);
    }

    #[test]
    fn reassign_and_reposition_arm_canvas_modes_by_weight_id() {
        let (mut state, body, _, w) = four_bar_with_weight();
        state.active_tool = EditorTool::PlaceMass;

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::ReassignPointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert_eq!(state.reassigning_point_mass, Some((body.clone(), w.clone())));
        assert_eq!(state.repositioning_point_mass, None);
        assert_eq!(state.active_tool, EditorTool::Select);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::RepositionPointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert_eq!(state.repositioning_point_mass, Some((body, w)));
        assert_eq!(state.reassigning_point_mass, None);
    }

    #[test]
    fn stale_weight_edit_is_a_no_op() {
        let (mut state, body, other, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();
        // The weight moved away (e.g. by an earlier edit) before this one applied.
        assert!(state.move_point_mass(&body, &w, &other, [0.0, 0.0]));
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassMass { body_id: body, weight_id: w.clone(), mass: 9.0 }),
        );
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }
}
```


**1f. `gui/property_panel/mod.rs`: idle frames must not clamp a loaded weight mass.** A 1500 kg weight and a 0 kg weight (only possible from a file) sit outside the mass field's `0.001..=1000` kg range. egui's `DragValue` clamps an out-of-range bound value and reports the response as changed, which the panel would commit as a silent edit plus an undo entry. Two idle frames must change nothing.

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
            if response.lost_focus() && (scale_pct - 100.0).abs() > 0.01 {
                *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: scale_pct / 100.0 });
                // Reset to 100% after applying
                ui.data_mut(|d| *d.get_temp_mut_or(id, 100.0) = 100.0);
            }
        });
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::test_support::sorted_link_ids;

    /// Run one frame of the property panel with no user input at all.
    fn one_idle_frame(state: &mut AppState) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| draw_property_panel(ui, state));
        });
    }

    /// Weight masses outside the mass field's 0.001..=1000 kg range (a heavy
    /// payload, or a zero mass the loader skipped) must survive idle frames.
    /// egui's DragValue clamps an out-of-range bound value and reports it as
    /// changed, which the panel would commit as a silent edit + undo entry.
    #[test]
    fn idle_frames_do_not_clamp_out_of_range_weight_masses() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let links = sorted_link_ids(&state);
        let body = links[0].clone();
        let heavy = state.add_point_mass(&body, 2.0, [0.03, 0.0]).unwrap();
        let skipped = state.add_point_mass(&body, 2.0, [-0.03, 0.0]).unwrap();
        // Out-of-range masses only arrive from files; write them directly.
        {
            let pms = &mut state.blueprint.as_mut().unwrap().bodies.get_mut(&body).unwrap().point_masses;
            pms[0].mass = 1500.0;
            pms[1].mass = 0.0;
        }
        state.rebuild();
        state.link_editor_body = Some(body.clone());
        let depth = state.undo_history.undo_count();

        // Two frames: the panel applies at most one pending edit per frame.
        one_idle_frame(&mut state);
        one_idle_frame(&mut state);

        assert_eq!(state.find_point_mass(&body, &heavy).unwrap().mass, 1500.0);
        assert_eq!(state.find_point_mass(&body, &skipped).unwrap().mass, 0.0);
        assert_eq!(state.undo_history.undo_count(), depth, "idle frames must not record edits");
    }
}
```


**1g. `gui/canvas/mod.rs`: headless canvas clicks.** Three tests drive the real `draw_canvas` with pointer events: a Reposition click, a Move to Link click (world position, id and mass survive), and a Place Mass click (uses `last_point_mass_kg`, next id `W2`). Add this module directly after the last test in `mod tests` (`fill_template_linear_actuator`), before the closing `}` of `mod tests`:

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            _ => panic!("expected LinearActuator"),
        }
    }
```

Add after it:

```rust
    /// Headless canvas clicks driving the weight (point-mass) handlers.
    mod weight_clicks {
        use eframe::egui::{self, Pos2};
        use nalgebra::Vector2;

        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::{AppState, EditorTool};
        use crate::gui::test_support::sorted_link_ids;
        use super::super::draw_canvas;

        fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw_canvas(ui, state));
            });
        }

        fn click(ctx: &egui::Context, state: &mut AppState, pos: Pos2) {
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            frame(ctx, state, vec![egui::Event::PointerMoved(pos)]);
            frame(ctx, state, vec![button(true)]);
            frame(ctx, state, vec![button(false)]);
        }

        /// Four-bar with a 2 kg weight "W1" on the first sorted link, after one
        /// idle frame (which applies the pending fit-to-view, so `state.view`
        /// is final). Returns the context, state, that link and a second link.
        fn setup() -> (egui::Context, AppState, String, String) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::FourBar);
            let links = sorted_link_ids(&state);
            assert_eq!(state.add_point_mass(&links[0], 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            (ctx, state, links[0].clone(), links[1].clone())
        }

        fn world_of(state: &AppState, body: &str, local: [f64; 2]) -> [f64; 2] {
            let mech = state.mechanism.as_ref().unwrap();
            let p = mech.state().body_point_global(body, &Vector2::new(local[0], local[1]), &state.q);
            [p.x, p.y]
        }

        fn screen_of(state: &AppState, world: [f64; 2]) -> Pos2 {
            let [x, y] = state.view.world_to_screen(world[0], world[1]);
            Pos2::new(x, y)
        }

        /// The body-local point under screen position `pos` on `body`.
        fn local_under(state: &AppState, body: &str, pos: Pos2) -> [f64; 2] {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            state.world_to_body_local(body, wx, wy)
        }

        fn assert_close(what: &str, got: [f64; 2], want: [f64; 2]) {
            assert!(
                (got[0] - want[0]).abs() < 1e-9 && (got[1] - want[1]).abs() < 1e-9,
                "{what}: {got:?} vs {want:?}"
            );
        }

        #[test]
        fn reposition_click_moves_the_weight_by_id_as_one_undo_step() {
            let (ctx, mut state, body, _) = setup();
            let target = Pos2::new(120.0, 700.0); // empty canvas
            let expected = local_under(&state, &body, target);
            let depth = state.undo_history.undo_count();

            state.repositioning_point_mass = Some((body.clone(), "W1".to_string()));
            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W1").expect("W1 keeps its id and link");
            assert_close("reposition", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.repositioning_point_mass.is_none(), "the mode ends after one click");
        }

        #[test]
        fn move_to_link_click_keeps_world_position_id_and_mass() {
            let (ctx, mut state, body, other) = setup();
            let world = world_of(&state, &body, [0.03, 0.02]);
            let expected = state.world_to_body_local(&other, world[0], world[1]);
            // Click the middle of the other link's bar.
            let mech = state.mechanism.as_ref().unwrap();
            let pts: Vec<Vector2<f64>> = mech.bodies()[&other].attachment_points.values().copied().collect();
            assert_eq!(pts.len(), 2, "fixture: a two-pin link");
            let a = world_of(&state, &other, [pts[0].x, pts[0].y]);
            let b = world_of(&state, &other, [pts[1].x, pts[1].y]);
            let mid = screen_of(&state, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
            let depth = state.undo_history.undo_count();

            state.reassigning_point_mass = Some((body.clone(), "W1".to_string()));
            click(&ctx, &mut state, mid);

            assert!(state.find_point_mass(&body, "W1").is_none(), "W1 left the old link");
            let pm = state.find_point_mass(&other, "W1").expect("W1 is on the clicked link");
            assert_close("reattach keeps the world position", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.reassigning_point_mass.is_none(), "the mode ends after one click");
        }

        #[test]
        fn place_mass_click_uses_the_last_point_mass() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            let expected = local_under(&state, &body, target);

            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W2").expect("the new weight gets the next id");
            assert_eq!(pm.mass, 3.5);
            assert_close("placement", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
        }
    }
```


- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib -- point_mass weight idle_frames
```

Expected: the test build fails to compile (nothing runs). In the reference run the last line was ``error: could not compile `linkage-sim-rs` (lib test) due to 86 previous errors``. The headline errors are:

- ``error[E0599]: no method named `move_point_mass` found for struct `gui::state::AppState` `` (also `find_point_mass`, `set_point_mass_mass`, `set_point_mass_label`, `remove_point_mass_by_id`)
- ``error[E0609]: no field `last_point_mass_kg` on type `gui::state::AppState` ``
- ``error[E0599]: no method named `as_deref` found for unit type `()` `` (also `unwrap` and `expect`), because `add_point_mass` still returns `()`
- ``error[E0599]: no variant named `SetPointMassMass` found for enum `pending_edits::PendingPropertyEdit` `` (also `SetPointMassPosition`)
- ``error[E0559]: variant `pending_edits::PendingPropertyEdit::RemovePointMass` has no field named `weight_id` `` (also `ReassignPointMass`, `RepositionPointMass`)
- ``error[E0308]: mismatched types`` where a `(String, String)` is assigned or compared to `reassigning_point_mass` / `repositioning_point_mass` (`expected usize, found String`)

- [ ] **Step 3: Implement**

**3a. `gui/state/mod.rs`: the fields.** `last_point_mass_kg` is new. The two mode fields carry the weight id instead of a list index.

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
    /// Point mass being reassigned to a different link. (body_id, index)
    /// When Some, next link click moves the mass to that body.
    pub reassigning_point_mass: Option<(String, usize)>,
    /// Point mass being repositioned via mouse click. (body_id, index)
    /// When Some, next canvas click updates the mass position.
    pub repositioning_point_mass: Option<(String, usize)>,
```

Replace with:

```rust
    /// Mass (kg) of the most recently added weight; the default for the next
    /// placement. Set by `add_point_mass`.
    pub last_point_mass_kg: f64,
    /// Point mass being reassigned to a different link. (body_id, weight_id)
    /// When Some, next link click moves the mass to that body.
    pub reassigning_point_mass: Option<(String, String)>,
    /// Point mass being repositioned via mouse click. (body_id, weight_id)
    /// When Some, next canvas click updates the mass position.
    pub repositioning_point_mass: Option<(String, String)>,
```


and in `impl Default for AppState`:

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
            place_mass_body: None,
            reassigning_point_mass: None,
```

Replace with:

```rust
            place_mass_body: None,
            last_point_mass_kg: 1.0,
            reassigning_point_mass: None,
```


**3b. `gui/state/blueprint_ops.rs`: imports and private helpers.** The helpers locate a weight by id inside a body, so every public method below shares one lookup rule (a blank id never matches).

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
use crate::io::{
    load_mechanism_unbuilt_from_json,
    DriverJson, JointJson, MechanismJson,
};
```

Replace with:

```rust
use crate::io::{
    load_mechanism_unbuilt_from_json, next_point_mass_id, point_mass_skip_reason,
    BodyJson, DriverJson, JointJson, MechanismJson, PointMassJson,
};
```


Insert the helpers after `seed_q_by_body_id`:

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
    q
}

/// Extract (body_i, point_i, body_j, point_j) from a JointJson.
```

Replace with:

```rust
    q
}

/// Position of weight `weight_id` in `body`'s point-mass list. Blank ids
/// never match (they are not addressable; see `io::assign_point_mass_ids`).
fn point_mass_position(body: &BodyJson, weight_id: &str) -> Option<usize> {
    if crate::io::schema::is_blank_point_mass_id(weight_id) {
        return None;
    }
    body.point_masses.iter().position(|pm| pm.id == weight_id)
}

/// Mutable access to weight `weight_id` on body `body_id` of `bp`.
fn point_mass_mut<'a>(bp: &'a mut MechanismJson, body_id: &str, weight_id: &str) -> Option<&'a mut PointMassJson> {
    let body = bp.bodies.get_mut(body_id)?;
    let i = point_mass_position(body, weight_id)?;
    body.point_masses.get_mut(i)
}

/// Remove weight `weight_id` from body `body_id` of `bp` and return it.
fn take_point_mass(bp: &mut MechanismJson, body_id: &str, weight_id: &str) -> Option<PointMassJson> {
    let body = bp.bodies.get_mut(body_id)?;
    let i = point_mass_position(body, weight_id)?;
    Some(body.point_masses.remove(i))
}

/// Extract (body_i, point_i, body_j, point_j) from a JointJson.
```


**3c. `gui/state/blueprint_ops.rs`: the id-addressed weight API.** This replaces `add_point_mass` (Task 1 form), `point_mass_exists`, `remove_point_mass`, `move_point_mass_to_body` and `update_point_mass` in one block:

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
    /// Add a point mass to a body in the blueprint.
    ///
    /// Pushes undo, appends the point mass with the next free `W<n>` id, and
    /// rebuilds (which recomputes composite mass, CG, and Izz via parallel
    /// axis theorem).
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let id = crate::io::next_point_mass_id(&bp.bodies);
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.push(crate::io::PointMassJson {
            id,
            label: None,
            mass,
            local_pos,
        });
        self.rebuild();
    }

    /// True if the blueprint has `body_id` with a point mass at `index`.
    fn point_mass_exists(&self, body_id: &str, index: usize) -> bool {
        self.blueprint
            .as_ref()
            .and_then(|bp| bp.bodies.get(body_id))
            .is_some_and(|b| index < b.point_masses.len())
    }

    /// Remove a point mass from a body in the blueprint by index.
    ///
    /// Pushes undo, removes the point mass, and rebuilds.  No-op (no undo
    /// entry) if the body or `index` does not exist.
    pub fn remove_point_mass(&mut self, body_id: &str, index: usize) {
        if !self.point_mass_exists(body_id, index) {
            return;
        }
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.remove(index);
        self.rebuild();
    }

    /// Move a point mass to another body, keeping its mass and placing it at
    /// `new_local_pos` in the new body's frame.
    ///
    /// One undo entry and one rebuild (`remove_point_mass` + `add_point_mass`
    /// would record two of each).  No-op (no undo entry, mass kept in place)
    /// if the source point mass or the destination body does not exist.
    pub fn move_point_mass_to_body(
        &mut self,
        old_body_id: &str,
        index: usize,
        new_body_id: &str,
        new_local_pos: [f64; 2],
    ) {
        if !self.point_mass_exists(old_body_id, index) {
            return;
        }
        if !self.blueprint.as_ref().is_some_and(|bp| bp.bodies.contains_key(new_body_id)) {
            return;
        }
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(old_body) = bp.bodies.get_mut(old_body_id) else { return };
        let mut point_mass = old_body.point_masses.remove(index);
        point_mass.local_pos = new_local_pos;
        let Some(new_body) = bp.bodies.get_mut(new_body_id) else { return };
        new_body.point_masses.push(point_mass);
        self.rebuild();
    }

    /// Update a point mass on a body in the blueprint by index.
    ///
    /// One committed edit = one undo entry, so callers must invoke this once
    /// per committed edit (drag-stop / typed value / Reposition click), not on
    /// every frame of a drag.  Updates mass and/or local_pos, then rebuilds.
    /// No-op (no undo entry) if the body or `index` does not exist.
    pub fn update_point_mass(
        &mut self,
        body_id: &str,
        index: usize,
        mass: f64,
        local_pos: [f64; 2],
    ) {
        if !self.point_mass_exists(body_id, index) {
            return;
        }
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        let Some(pm) = body.point_masses.get_mut(index) else { return };
        pm.mass = mass;
        pm.local_pos = local_pos;
        self.rebuild();
    }
```

Replace with:

```rust
    // ── Weights (point masses), addressed by (body_id, weight_id) ─────────
    //
    // Every edit validates first and records exactly one undo entry (via
    // `mutate_and_rebuild`) only when it changes the blueprint. A weight
    // the edit would leave in a state the loader skips
    // (`io::point_mass_skip_reason`: on ground, mass not positive and
    // finite, non-finite position) is rejected. Call each once per committed
    // edit (drag-stop / typed value / click), never per drag frame.

    /// Add a weight to body `body_id` and return its new id (the smallest
    /// unused `W<n>`).
    ///
    /// One undo entry and one rebuild (which recomputes composite mass, CG
    /// and Izz). Remembers `mass` in `last_point_mass_kg`, the default for
    /// the next placement. Returns `None` (no undo entry, nothing changed)
    /// when there is no blueprint, the body does not exist, or the loader
    /// would skip the weight.
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String> {
        let bp = self.blueprint.as_ref()?;
        if !bp.bodies.contains_key(body_id) {
            return None;
        }
        let weight = PointMassJson { id: next_point_mass_id(&bp.bodies), label: None, mass, local_pos };
        if point_mass_skip_reason(body_id, &weight).is_some() {
            return None;
        }
        let id = weight.id.clone();
        self.mutate_and_rebuild(|s| {
            if let Some(body) = s.blueprint.as_mut().and_then(|bp| bp.bodies.get_mut(body_id)) {
                body.point_masses.push(weight);
            }
        });
        self.last_point_mass_kg = mass;
        Some(id)
    }

    /// The weight `weight_id` on body `body_id`, if both exist.
    pub fn find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson> {
        let body = self.blueprint.as_ref()?.bodies.get(body_id)?;
        body.point_masses.get(point_mass_position(body, weight_id)?)
    }

    /// Move weight `weight_id` from `body_id` to `local_pos` in
    /// `target_body`'s frame.
    ///
    /// `target_body == body_id` repositions the weight in place (list order
    /// kept); another body reattaches it there (appended), keeping its id,
    /// label and mass. Returns `false` (no undo entry, nothing changed) when
    /// the weight or the target body does not exist, or the loader would skip
    /// the weight at the target (ground target, non-finite position). Returns
    /// `true` without an undo entry when the weight is already there.
    pub fn move_point_mass(
        &mut self,
        body_id: &str,
        weight_id: &str,
        target_body: &str,
        local_pos: [f64; 2],
    ) -> bool {
        if target_body == body_id {
            return self.edit_point_mass(body_id, weight_id, |pm| pm.local_pos = local_pos);
        }
        let Some(current) = self.find_point_mass(body_id, weight_id) else { return false };
        let moved = PointMassJson { local_pos, ..current.clone() };
        let target_exists = self.blueprint.as_ref().is_some_and(|bp| bp.bodies.contains_key(target_body));
        if !target_exists || point_mass_skip_reason(target_body, &moved).is_some() {
            return false;
        }
        self.mutate_and_rebuild(|s| {
            let Some(bp) = s.blueprint.as_mut() else { return };
            take_point_mass(bp, body_id, weight_id);
            if let Some(target) = bp.bodies.get_mut(target_body) {
                target.point_masses.push(moved);
            }
        });
        true
    }

    /// Set the mass (kg) of weight `weight_id` on `body_id`. Returns `false`
    /// (nothing recorded) for a missing weight or a mass that is not positive
    /// and finite; `true` without an undo entry when unchanged.
    pub fn set_point_mass_mass(&mut self, body_id: &str, weight_id: &str, mass: f64) -> bool {
        self.edit_point_mass(body_id, weight_id, |pm| pm.mass = mass)
    }

    /// Set the display label of weight `weight_id` on `body_id`. Surrounding
    /// whitespace is trimmed and a blank label clears it (`None`: the id is
    /// shown). Returns `false` for a missing weight; `true` without an undo
    /// entry when unchanged.
    pub fn set_point_mass_label(&mut self, body_id: &str, weight_id: &str, label: Option<String>) -> bool {
        let label = label.map(|l| l.trim().to_string()).filter(|l| !l.is_empty());
        self.edit_point_mass(body_id, weight_id, |pm| pm.label = label)
    }

    /// Remove weight `weight_id` from `body_id`. Returns `false` (no undo
    /// entry) when it does not exist.
    pub fn remove_point_mass_by_id(&mut self, body_id: &str, weight_id: &str) -> bool {
        if self.find_point_mass(body_id, weight_id).is_none() {
            return false;
        }
        self.mutate_and_rebuild(|s| {
            if let Some(bp) = s.blueprint.as_mut() {
                take_point_mass(bp, body_id, weight_id);
            }
        });
        true
    }

    /// Apply `edit` to weight `weight_id` on `body_id` in place: the shared
    /// path of every same-body weight edit. Returns `false` (no undo entry)
    /// when the weight is missing or the edited weight would be skipped by
    /// the loader; `true` without an undo entry when `edit` changes nothing.
    fn edit_point_mass(
        &mut self,
        body_id: &str,
        weight_id: &str,
        edit: impl FnOnce(&mut PointMassJson),
    ) -> bool {
        let Some(current) = self.find_point_mass(body_id, weight_id) else { return false };
        let mut updated = current.clone();
        edit(&mut updated);
        if point_mass_skip_reason(body_id, &updated).is_some() {
            return false;
        }
        if updated == *current {
            return true;
        }
        self.mutate_and_rebuild(|s| {
            if let Some(pm) = s.blueprint.as_mut().and_then(|bp| point_mass_mut(bp, body_id, weight_id)) {
                *pm = updated;
            }
        });
        true
    }
```


**3d. `gui/state/file_io.rs`: extract `overlay_blueprint_point_masses`.** The inline BL-023 block in `serialize_to_json_string` becomes a method, so `take_snapshot` can share it. First add the method above `serialize_to_json_string`:

Find in `linkage-sim-rs/src/gui/state/file_io.rs`:

```rust
    /// Serialize the current mechanism to a pretty-printed JSON string.
    ///
    /// Includes load cases, mounting angle, and blueprint point masses.
```

Replace with:

```rust
    /// Put the blueprint's editable weights into `json_struct`, a
    /// `mechanism_to_json` dump of the live mechanism.
    ///
    /// `mechanism_to_json` bakes point masses into the composite
    /// (point-mass-inclusive) mass/CG/Izz and emits no list, while load
    /// re-applies the list on top of the stored values. So for every body
    /// with point masses, write the blueprint's BASE mass/CG/Izz plus its
    /// point-mass list (ids, labels): each weight is then applied exactly
    /// once (BL-023) and stays editable. Used by file/URL serialization and
    /// by undo snapshots.
    pub(crate) fn overlay_blueprint_point_masses(&self, json_struct: &mut crate::io::MechanismJson) {
        let Some(bp) = &self.blueprint else { return };
        for (body_id, bp_body) in &bp.bodies {
            if bp_body.point_masses.is_empty() {
                continue;
            }
            if let Some(json_body) = json_struct.bodies.get_mut(body_id) {
                json_body.point_masses = bp_body.point_masses.clone();
                json_body.mass = bp_body.mass;
                json_body.cg_local = bp_body.cg_local;
                json_body.izz_cg = bp_body.izz_cg;
            }
        }
    }

    /// Serialize the current mechanism to a pretty-printed JSON string.
    ///
    /// Includes load cases, mounting angle, and blueprint point masses.
```


then replace the inline block with a call:

Find in `linkage-sim-rs/src/gui/state/file_io.rs`:

```rust
        json_struct.mounting_angle = self.mounting_angle;
        // Preserve blueprint point masses (baked into mass/CG/Izz at build time).
        // `mechanism_to_json` emitted the composite (point-mass-inclusive)
        // mass/CG/Izz, and load re-applies the point masses on top of the
        // stored values, so for these bodies write the blueprint's BASE
        // mass/CG/Izz instead or every point mass is counted twice (BL-023).
        if let Some(ref bp) = self.blueprint {
            for (body_id, bp_body) in &bp.bodies {
                if bp_body.point_masses.is_empty() {
                    continue;
                }
                if let Some(json_body) = json_struct.bodies.get_mut(body_id) {
                    json_body.point_masses = bp_body.point_masses.clone();
                    json_body.mass = bp_body.mass;
                    json_body.cg_local = bp_body.cg_local;
                    json_body.izz_cg = bp_body.izz_cg;
                }
            }
        }
        // Persist GUI sweep state (mode + trajectory severity) so reload
```

Replace with:

```rust
        json_struct.mounting_angle = self.mounting_angle;
        self.overlay_blueprint_point_masses(&mut json_struct);
        // Persist GUI sweep state (mode + trajectory severity) so reload
```


**3e. `gui/state/undo_ops.rs`: undo snapshots keep the editable weight list.**

Find in `linkage-sim-rs/src/gui/state/undo_ops.rs`:

```rust
        let json = mechanism_to_json(mech).ok()?;
        let json_str = serde_json::to_string(&json).ok()?;
```

Replace with:

```rust
        let mut json = mechanism_to_json(mech).ok()?;
        // Keep the editable weights (ids, labels, base mass) so undo/redo
        // restores them instead of baking them into the composite mass.
        self.overlay_blueprint_point_masses(&mut json);
        let json_str = serde_json::to_string(&json).ok()?;
```


**3f. `gui/property_panel/pending_edits.rs`: the pending-edit variants and their arms.**

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
    UpdatePointMass { body_id: String, index: usize, mass: f64, local_pos: [f64; 2] },
    RemovePointMass { body_id: String, index: usize },
    /// Enter mode to reassign a point mass to a different body.
    ReassignPointMass { body_id: String, index: usize },
    /// Enter mode to reposition a point mass via mouse click.
    RepositionPointMass { body_id: String, index: usize },
```

Replace with:

```rust
    /// Commit a typed / drag-stopped weight mass (kg).
    SetPointMassMass { body_id: String, weight_id: String, mass: f64 },
    /// Commit a typed / drag-stopped body-local X or Y of a weight.
    SetPointMassPosition { body_id: String, weight_id: String, local_pos: [f64; 2] },
    RemovePointMass { body_id: String, weight_id: String },
    /// Enter mode to reassign a point mass to a different body.
    ReassignPointMass { body_id: String, weight_id: String },
    /// Enter mode to reposition a point mass via mouse click.
    RepositionPointMass { body_id: String, weight_id: String },
```


and in `apply_pending`:

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
            PendingPropertyEdit::UpdatePointMass { body_id, index, mass, local_pos } => {
                state.update_point_mass(&body_id, index, mass, local_pos);
            }
            PendingPropertyEdit::RemovePointMass { body_id, index } => {
                state.remove_point_mass(&body_id, index);
            }
            PendingPropertyEdit::ReassignPointMass { body_id, index } => {
                state.reassigning_point_mass = Some((body_id, index));
                state.repositioning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::RepositionPointMass { body_id, index } => {
                state.repositioning_point_mass = Some((body_id, index));
                state.reassigning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
```

Replace with:

```rust
            PendingPropertyEdit::SetPointMassMass { body_id, weight_id, mass } => {
                state.set_point_mass_mass(&body_id, &weight_id, mass);
            }
            PendingPropertyEdit::SetPointMassPosition { body_id, weight_id, local_pos } => {
                state.move_point_mass(&body_id, &weight_id, &body_id, local_pos);
            }
            PendingPropertyEdit::RemovePointMass { body_id, weight_id } => {
                state.remove_point_mass_by_id(&body_id, &weight_id);
            }
            PendingPropertyEdit::ReassignPointMass { body_id, weight_id } => {
                state.reassigning_point_mass = Some((body_id, weight_id));
                state.repositioning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::RepositionPointMass { body_id, weight_id } => {
                state.repositioning_point_mass = Some((body_id, weight_id));
                state.reassigning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
```


**3g. `gui/property_panel/mod.rs`: the Point Masses section addresses weights by `pm.id`.** No layout change. The mass field gains `.clamp_existing_to_range(false)` so a loaded out-of-range mass is not clamped, reported as changed and committed on an idle frame.

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                            .range(0.001..=1000.0)
                                                            .prefix("m: ")
                                                            .suffix(" kg"),
                                                    ).on_hover_text("Point mass magnitude in kg");
                                                    if mr.drag_stopped() || (mr.changed() && !mr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: mass_val,
                                                            local_pos: pm.local_pos,
                                                        });
```

Replace with:

```rust
                                                            .range(0.001..=1000.0)
                                                            // Never clamp a loaded mass on an idle frame:
                                                            // egui would report it as changed and the
                                                            // panel would commit a silent edit.
                                                            .clamp_existing_to_range(false)
                                                            .prefix("m: ")
                                                            .suffix(" kg"),
                                                    ).on_hover_text("Point mass magnitude in kg");
                                                    if mr.drag_stopped() || (mr.changed() && !mr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            mass: mass_val,
                                                        });
```


The X and Y fields commit through `SetPointMassPosition`:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                    if xr.drag_stopped() || (xr.changed() && !xr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: pm.mass,
                                                            local_pos: [units.length_to_si(x_display), pm.local_pos[1]],
                                                        });
```

Replace with:

```rust
                                                    if xr.drag_stopped() || (xr.changed() && !xr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassPosition {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            local_pos: [units.length_to_si(x_display), pm.local_pos[1]],
                                                        });
```


Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                    if yr.drag_stopped() || (yr.changed() && !yr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: pm.mass,
                                                            local_pos: [pm.local_pos[0], units.length_to_si(y_display)],
                                                        });
```

Replace with:

```rust
                                                    if yr.drag_stopped() || (yr.changed() && !yr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassPosition {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            local_pos: [pm.local_pos[0], units.length_to_si(y_display)],
                                                        });
```


Remove, Move to Link and Reposition carry the weight id:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                        pending = Some(PendingPropertyEdit::RemovePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
```

Replace with:

```rust
                                                        pending = Some(PendingPropertyEdit::RemovePointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
```


Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                        pending = Some(PendingPropertyEdit::ReassignPointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
```

Replace with:

```rust
                                                        pending = Some(PendingPropertyEdit::ReassignPointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
```


Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                                                        pending = Some(PendingPropertyEdit::RepositionPointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
```

Replace with:

```rust
                                                        pending = Some(PendingPropertyEdit::RepositionPointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
```


**3h. `gui/canvas/interaction.rs`: Move to Link, Reposition and Place Mass.** Move to Link looks the weight up by id instead of walking `bp.bodies` / `point_masses.get(index)`, and moves it with one `move_point_mass` call (one undo entry):

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
                let (old_body_id, pm_index) = state.reassigning_point_mass.take().unwrap();
                // Get the mass value and world position of the existing point mass
                if let Some(bp) = &state.blueprint {
                    if let Some(body) = bp.bodies.get(&old_body_id) {
                        if let Some(pm) = body.point_masses.get(pm_index) {
                            let [wx, wy] = {
                                let lx = pm.local_pos[0];
                                let ly = pm.local_pos[1];
                                // Convert old body-local to world
                                if old_body_id == "ground" {
                                    [lx, ly]
                                } else if let Some(mech) = &state.mechanism {
                                    if let Ok(idx) = mech.state().get_index(&old_body_id) {
                                        let bx = state.q[idx.q_start];
                                        let by = state.q[idx.q_start + 1];
                                        let theta = state.q[idx.q_start + 2];
                                        let ct = theta.cos();
                                        let st = theta.sin();
                                        [bx + ct * lx - st * ly, by + st * lx + ct * ly]
                                    } else { [lx, ly] }
                                } else { [lx, ly] }
                            };
                            // Move from old body to new body in one undoable step
                            let [nlx, nly] = state.world_to_body_local(&new_body_id, wx, wy);
                            state.move_point_mass_to_body(&old_body_id, pm_index, &new_body_id, [nlx, nly]);
                        }
                    }
                }
```

Replace with:

```rust
                let (old_body_id, weight_id) = state.reassigning_point_mass.take().unwrap();
                // Get the world position of the existing point mass
                if let Some([lx, ly]) = state.find_point_mass(&old_body_id, &weight_id).map(|pm| pm.local_pos) {
                    let [wx, wy] = {
                        // Convert old body-local to world
                        if old_body_id == "ground" {
                            [lx, ly]
                        } else if let Some(mech) = &state.mechanism {
                            if let Ok(idx) = mech.state().get_index(&old_body_id) {
                                let bx = state.q[idx.q_start];
                                let by = state.q[idx.q_start + 1];
                                let theta = state.q[idx.q_start + 2];
                                let ct = theta.cos();
                                let st = theta.sin();
                                [bx + ct * lx - st * ly, by + st * lx + ct * ly]
                            } else { [lx, ly] }
                        } else { [lx, ly] }
                    };
                    // Move from old body to new body in one undoable step
                    let [nlx, nly] = state.world_to_body_local(&new_body_id, wx, wy);
                    state.move_point_mass(&old_body_id, &weight_id, &new_body_id, [nlx, nly]);
                }
```


Reposition no longer reads the current mass back (the mass is untouched by a move):

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
            let (body_id, pm_index) = state.repositioning_point_mass.take().unwrap();
            let [lx, ly] = state.world_to_body_local(&body_id, wx as f64, wy as f64);
            // Get current mass value
            let mass_val = state.blueprint.as_ref()
                .and_then(|bp| bp.bodies.get(&body_id))
                .and_then(|b| b.point_masses.get(pm_index))
                .map(|pm| pm.mass)
                .unwrap_or(1.0);
            state.update_point_mass(&body_id, pm_index, mass_val, [lx, ly]);
```

Replace with:

```rust
            let (body_id, weight_id) = state.repositioning_point_mass.take().unwrap();
            let [lx, ly] = state.world_to_body_local(&body_id, wx as f64, wy as f64);
            state.move_point_mass(&body_id, &weight_id, &body_id, [lx, ly]);
```


Place Mass uses the last mass used:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
                state.add_point_mass(&body_id, 1.0, [lx, ly]); // 1 kg default
```

Replace with:

```rust
                state.add_point_mass(&body_id, state.last_point_mass_kg, [lx, ly]);
```


**3i. Docs.** `docs/ai/02-system.yaml`: the `point_mass_ops_one_undo_step_per_commit` invariant now describes the id-addressed API, and the `blueprint_mass_is_base_live_mass_is_composite` invariant names `overlay_blueprint_point_masses` (used by file/URL serialization and by `take_snapshot`) next to the `sync_live_mass_props` rule for `set_body_mass` / `set_body_izz`:

Find in `docs/ai/02-system.yaml`:

```yaml
  - point_mass_ops_one_undo_step_per_commit — remove_point_mass /
    update_point_mass / move_point_mass_to_body each push exactly one undo
    entry, and only after validating the body + index (invalid target = no
    entry, no rebuild). Call update_point_mass once per committed edit
    (drag-stop / typed value / Reposition click), never per drag frame. Move
    to Link must use move_point_mass_to_body, not remove + add (two entries)
    (BL-025).
  - blueprint_mass_is_base_live_mass_is_composite — state.blueprint bodies
    hold BASE mass/CG/Izz plus a point_masses list; the built Mechanism
    holds the composite (base + point masses). Anything that writes a
    file/URL from the live mechanism (AppState::serialize_to_json_string)
    must write the blueprint BASE values for bodies that have point_masses,
    because load re-applies the list on top. Writing composite + list
    double-counts every point mass (BL-023). mechanism_to_json alone (undo
    snapshots) bakes point masses in and emits an empty list, which is
    self-consistent but loses the editable list. set_body_mass /
    set_body_izz edit the blueprint BASE value and then re-derive the live
```

Replace with:

```yaml
  - point_mass_ops_one_undo_step_per_commit — the id-addressed weight API
    (AppState::add_point_mass -> Option<id>, move_point_mass (same body =
    reposition, other body = reattach keeping id/label/mass),
    set_point_mass_mass, set_point_mass_label, remove_point_mass_by_id;
    find_point_mass to read) validates first and pushes exactly one undo
    entry via mutate_and_rebuild. Invalid target (missing weight/body, ground
    target, mass not positive+finite, non-finite position) = false/None, no
    entry, no rebuild; a no-op edit = true, no entry. Call once per committed
    edit (drag-stop / typed value / click), never per drag frame. Move to
    Link is one move_point_mass, not remove + add (BL-025).
  - blueprint_mass_is_base_live_mass_is_composite — state.blueprint bodies
    hold BASE mass/CG/Izz plus a point_masses list; the built Mechanism
    holds the composite (base + point masses). Anything that writes a
    file/URL from the live mechanism (AppState::serialize_to_json_string)
    must write the blueprint BASE values for bodies that have point_masses,
    because load re-applies the list on top. Writing composite + list
    double-counts every point mass (BL-023). mechanism_to_json alone bakes
    point masses in and emits an empty list, so every writer (file/URL
    serialization AND undo snapshots, take_snapshot) calls
    AppState::overlay_blueprint_point_masses; undo/redo then restores the
    editable weight list with ids and labels. set_body_mass /
    set_body_izz edit the blueprint BASE value and then re-derive the live
```


and a lesson about `DragValue::range` next to the existing egui lesson (add the three lines directly after that line, no blank line between):

Find in `docs/ai/02-system.yaml`:

```yaml
    or typed edits, or bind a non-stepped slider (BL-020, BL-021).
```

Add after it:

```yaml
    DragValue::range also marks the response changed() when it clamps an
    out-of-range stored value on an idle frame; use
    .clamp_existing_to_range(false) on committed fields (weight mass field).
```


`docs/ai/05-update-tracker.md`: add this entry as the newest one, above the Task 1 entry:

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Add after it:

```markdown
## 2026-09-29 — Payload weights Task 2: id-addressed weight editing API
- `gui/state/blueprint_ops.rs`: index-based `remove_point_mass` /
  `update_point_mass` / `move_point_mass_to_body` replaced by
  `add_point_mass -> Option<String>` (sets `last_point_mass_kg`),
  `find_point_mass`, `move_point_mass` (same body = reposition, other body =
  reattach), `set_point_mass_mass`, `set_point_mass_label` (trimmed, blank
  clears), `remove_point_mass_by_id`. Each validates with
  `io::point_mass_skip_reason` and records exactly one undo entry through
  `mutate_and_rebuild`; invalid = no entry; no-op = no entry.
- Undo fidelity: `take_snapshot` now calls the extracted
  `overlay_blueprint_point_masses` (shared with `serialize_to_json_string`),
  so undo/redo restores the weight list with ids and labels instead of baking
  weights into the link mass (gap noted in the BL-025 entry below).
- Callers migrated with no visible UI change: property panel pending edits
  (`SetPointMassMass`, `SetPointMassPosition`, `RemovePointMass`,
  `ReassignPointMass`, `RepositionPointMass` carry `weight_id`), canvas Move
  to Link / Reposition (`reassigning_point_mass` / `repositioning_point_mass`
  are `(body_id, weight_id)`), Place Mass uses `last_point_mass_kg`
  (`AppState` field, default 1.0). The weight mass DragValue no longer clamps
  a loaded out-of-range mass (e.g. 1500 kg, or 0 kg) on an idle frame.
- Tests: `gui/state/tests.rs` (BL-025 tests on the id API, now also asserting
  undo restores the weight lists; add/move/label/remove, invalid targets,
  no-op edits, undo/redo list restore, repairing loader-skipped weights),
  `property_panel::tests` (idle-frame clamp), `property_panel::pending_edits::tests`,
  `canvas::tests::weight_clicks` (headless Reposition, Move to Link, Place Mass clicks).
  Their fixtures take link ids from `gui::test_support::sorted_link_ids`.
```


- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib -- point_mass weight idle_frames
```

Expected: `test result: ok. 56 passed; 0 failed; 0 ignored; 0 measured; 729 filtered out`. (The filter also picks up the Task 1 loader and schema tests whose names contain `point_mass`.)

Then the whole lib suite:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: `test result: ok. 785 passed; 0 failed`.

Then the gate:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected: the last line is `GATE PASS` (its `cargo test --all` shows the same 785 lib tests). The clippy warning count must not go up. The reference run went from 394 to 392, and the new code adds none. The test run regenerates the images in `docs/chebyshev_lambda/`. Restore them so they do not enter the commit:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Optional mutation checks (each verified red against the finished commit, restore the line afterwards):

- Delete the `self.overlay_blueprint_point_masses(&mut json);` line in `take_snapshot`. Eight tests fail (`point_mass_numeric_mass_edit_is_one_undo_step_bl025`, `point_mass_numeric_position_edit_is_one_undo_step_bl025`, `point_mass_reposition_is_one_undo_step_bl025`, `point_mass_move_to_link_is_one_undo_step_bl025`, `add_point_mass_is_one_undo_step`, `remove_point_mass_by_id_is_one_undo_step`, `set_point_mass_label_is_one_undo_step_and_trims_or_clears`, `undo_and_redo_restore_the_editable_weight_list`).
- Delete `.clamp_existing_to_range(false)` in the weight mass field. `idle_frames_do_not_clamp_out_of_range_weight_masses` fails.
- Change `if updated == *current {` to `if false && updated == *current {` in `edit_point_mass`. `no_op_weight_edits_succeed_without_undo_entry` fails.

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/gui/state/blueprint_ops.rs linkage-sim-rs/src/gui/state/mod.rs linkage-sim-rs/src/gui/state/file_io.rs linkage-sim-rs/src/gui/state/undo_ops.rs linkage-sim-rs/src/gui/state/tests.rs linkage-sim-rs/src/gui/property_panel/mod.rs linkage-sim-rs/src/gui/property_panel/pending_edits.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/canvas/mod.rs docs/ai/02-system.yaml docs/ai/05-update-tracker.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 2: id-addressed weight editing API

Replace the index-based point-mass API with add_point_mass -> Option<id>,
find_point_mass, move_point_mass (reposition / reattach),
set_point_mass_mass, set_point_mass_label and remove_point_mass_by_id,
each one validated undo step via mutate_and_rebuild (invalid or no-op =
no entry). Undo snapshots now carry the editable weight list, so undo
restores ids and labels. Property panel and canvas callers address weights
by id with no visible UI change; the weight mass field no longer clamps a
loaded out-of-range mass on an idle frame; Place Mass uses
last_point_mass_kg. Test fixtures take link ids from the shared
gui::test_support::sorted_link_ids.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 3: Gravity breakdown physics module

**Files:**
- Create: `linkage-sim-rs/src/analysis/gravity_breakdown.rs` (module doc, implementation, unit tests)
- Create: `linkage-sim-rs/tests/gravity_breakdown_reference.rs` (energy reference on the two actuator samples)
- Modify: `linkage-sim-rs/src/analysis/mod.rs` (register the module, right after `pub mod force_breakdown;`)
- Modify: `docs/ai/02-system.yaml` (analysis responsibility; new invariant before `point_mass_loader_validation`), `docs/ai/03-structure.yaml` (analysis files and a note), `docs/ai/05-update-tracker.md` (new top entry), `docs/architecture/ARCHITECTURE.md` (PointMass section, after the JSON-schema paragraph)

**Interfaces:**
- Consumes (Tasks 1-2): `crate::io::point_mass_skip_reason(body_id: &str, pm: &PointMassJson) -> Option<&'static str>` (the loader's own skip rule, re-exported from `io::from_json`); `PointMassJson { id: String, label: Option<String>, mass: f64, local_pos: [f64; 2] }`; `BodyJson { mass, cg_local, label, point_masses, .. }`; `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>` (integration test only). Existing: `State::body_point_velocity`, `State::body_point_global`, `State::is_ground`, `State::get_index`, `Mechanism::forces`, `core::state::GROUND_ID`.
- Produces (`crate::analysis::gravity_breakdown`):
  - `pub const EPS_REL_LDOT: f64 = 0.01; pub const BRAKE_TOL_REL: f64 = 1e-6; pub const NEUTRAL_REL: f64 = 0.01;`
  - `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)] pub struct WeightSource { pub id: String, pub name: String, pub body_id: String, pub local_pos: [f64; 2], pub mass: f64, pub is_link_self_weight: bool }`
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Classification { Helping, Hurting, Neutral }`
  - `pub fn weight_sources(bp: &MechanismJson) -> Vec<WeightSource>`
  - `pub fn gravity_powers(mech: &Mechanism, sources: &[WeightSource], q: &DVector<f64>, q_dot: &DVector<f64>, g: [f64; 2]) -> Vec<f64>`
  - `pub fn gravity_vector(mech: &Mechanism) -> [f64; 2]` (the sweep reads g from the mechanism's element, mounting angle included)
  - `pub fn max_abs_finite(values: &[f64]) -> f64`, `pub fn classify(p_g: f64, max_abs_p_g: f64) -> Classification`, `pub fn force_share(p_g: f64, rate: f64, max_abs_rate: f64) -> f64`, `pub fn is_braking(p_act: f64, max_abs_p_act: f64) -> bool` (the spec's named rules, used by Task 4 and the canvas)

**Physics in one paragraph.** Gravity is linear in mass, and a built body's composite mass and CG are the mass-weighted sum of its base mass and its point masses. So the model's gravity load is exactly the sum of the loads of its weight sources: one link self-weight per body (base mass at the base CG) plus every point mass the loader applies. `weight_sources` filters point masses with `io::point_mass_skip_reason`, the same rule `io::from_json::apply_point_masses` uses, so the list always matches the physics. At one pose each source's gravity power is `P_g,i = m_i g . v_i` (positive = the weight is coming down and helps the actuator). `g` comes from the mechanism's `Gravity` element, never from `gravity_magnitude`, because the element carries the mounting angle.

- [ ] **Step 1: Write the failing test**

Register the module in `linkage-sim-rs/src/analysis/mod.rs`.

Find in `linkage-sim-rs/src/analysis/mod.rs`:

```rust
pub mod force_breakdown;
```

Add after it:

```rust
pub mod gravity_breakdown;
```

Create `linkage-sim-rs/src/analysis/gravity_breakdown.rs` starting with this module doc:

```rust
//! Per-weight gravity power: which weights help and which hurt the actuator.
//!
//! A *weight source* is either a link's own mass (the blueprint's base
//! `mass` at its base `cg_local`) or a point mass the user attached to a
//! link. Gravity loads are linear in mass, and a built body's composite
//! mass and CG are the mass-weighted sum of its base mass and its point
//! masses (`core::body::Body::add_point_mass`), so the model's gravity load
//! is exactly the sum of the sources' gravity loads.
//!
//! At one pose with velocity `q_dot` (from the constant-rate velocity
//! solve) each source's gravity power is
//!
//! ```text
//! P_g,i = m_i * (g . v_i)     [W]
//! ```
//!
//! where `v_i` is the world velocity of the source's point
//! (`State::body_point_velocity`). `P_g,i > 0` means gravity does positive
//! work on that weight (it is coming down): the weight is **helping** the
//! actuator. The rules that turn powers into shares and labels live here
//! too, next to their named thresholds:
//!
//! - force share `-P_g,i / rate` ([`force_share`]; NaN near stroke
//!   reversal, [`EPS_REL_LDOT`]); the power share is `-P_g,i`,
//! - helping / hurting / neutral per weight ([`classify`], [`NEUTRAL_REL`]),
//! - braking where actuator power is negative ([`is_braking`],
//!   [`BRAKE_TOL_REL`]).
//!
//! The sweep (`gui::sweep`) applies them per sample. Spec:
//! docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md
//! (Track 2, section 2).
```

Then append the test module to the same file. Step 3 inserts the implementation between the module doc and `#[cfg(test)]`, so leave a blank line after the doc for now.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::elements::{evaluate_gravity, ForceElement, GravityElement};
    use crate::io::{load_mechanism_unbuilt_from_json, MechanismJson, SCHEMA_VERSION};
    use crate::solver::kinematics::{solve_position, solve_velocity};
    use nalgebra::{DVector, Vector2};
    use serde_json::json;

    const G: f64 = 9.81;

    fn build(bp: &MechanismJson) -> Mechanism {
        let mut mech = load_mechanism_unbuilt_from_json(bp).expect("blueprint loads");
        mech.build().expect("mechanism builds");
        mech
    }

    /// Ground plus one bar pinned at the origin (body frame at the pivot, so
    /// its pose is (0, 0, theta)), driven at `omega` rad/s from theta = 0.
    /// Bar: 2 kg base mass at (0.5, 0); weights W1 = 3 kg at the tip (1, 0)
    /// and W2 "Robot" = 1.5 kg at (0.6, 0). Gravity (0, -9.81).
    fn single_bar_blueprint(omega: f64) -> MechanismJson {
        serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {"O": [0.0, 0.0]},
                           "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "bar": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                        "mass": 2.0, "cg_local": [0.5, 0.0], "izz_cg": 0.2,
                        "point_masses": [
                            {"id": "W1", "mass": 3.0, "local_pos": [1.0, 0.0]},
                            {"id": "W2", "label": "Robot", "mass": 1.5, "local_pos": [0.6, 0.0]}
                        ]}
            },
            "joints": {
                "J1": {"type": "revolute", "body_i": "ground", "point_i": "O",
                       "body_j": "bar", "point_j": "A"}
            },
            "drivers": {
                "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "bar",
                       "omega": omega, "theta_0": 0.0}
            },
            "forces": [{"type": "Gravity", "g_vector": [0.0, -G]}]
        }))
        .expect("single-bar blueprint parses")
    }

    /// Sources and gravity powers of the single bar at `theta`, driven at
    /// `omega` (pose solved at driver time theta / omega).
    fn single_bar_powers(theta: f64, omega: f64) -> (Vec<WeightSource>, Vec<f64>) {
        let bp = single_bar_blueprint(omega);
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("bar", &mut q, 0.0, 0.0, theta);
        let t = theta / omega;
        let solved = solve_position(&mech, &q, t, 1e-12, 50).expect("position solve");
        assert!(solved.converged, "bar pose at {theta} rad did not converge");
        let q_dot = solve_velocity(&mech, &solved.q, t).expect("velocity solve");
        let powers = gravity_powers(&mech, &sources, &solved.q, &q_dot, gravity_vector(&mech));
        (sources, powers)
    }

    /// Hand calculation for a bar pivoted at the origin: a point at
    /// body-local (x, 0) moves with v = omega * x * (-sin theta, cos theta),
    /// so with g = (0, -9.81) its gravity power is
    /// P = -m * 9.81 * omega * x * cos(theta). Covers the bar rising
    /// (cos theta > 0, P < 0), falling (cos theta < 0, P > 0) and moving
    /// horizontally (theta = 90, 270 deg, P = 0).
    #[test]
    fn gravity_power_single_bar_matches_hand_calculation() {
        let omega = 2.0;
        // (mass, x) of link:bar, W1, W2 in weight_sources order.
        let points = [(2.0, 0.5), (3.0, 1.0), (1.5, 0.6)];
        for deg in [0.0_f64, 30.0, 60.0, 90.0, 120.0, 180.0, 210.0, 270.0, 300.0, 330.0] {
            let theta = deg.to_radians();
            let (sources, powers) = single_bar_powers(theta, omega);
            assert_eq!(sources.len(), points.len());
            for (i, &(m, x)) in points.iter().enumerate() {
                let hand = -m * G * omega * x * theta.cos();
                assert!(
                    (powers[i] - hand).abs() <= 1e-9 * m * G * omega * x,
                    "{} at {deg} deg: P_g {} vs hand {hand}",
                    sources[i].id,
                    powers[i]
                );
            }
        }
    }

    /// Same pose, opposite direction: going up the weights cost power
    /// (hurting, P_g < 0); coming down gravity gives exactly that power back
    /// (helping, P_g > 0). The key physics statement of the spec.
    #[test]
    fn gravity_power_flips_sign_between_ascending_and_descending() {
        let theta = 30.0_f64.to_radians();
        let (sources, up) = single_bar_powers(theta, 2.0);
        let (_, down) = single_bar_powers(theta, -2.0);
        for i in 0..sources.len() {
            assert!(up[i] < 0.0, "{} rising: P_g {} should be < 0", sources[i].id, up[i]);
            assert!(down[i] > 0.0, "{} falling: P_g {} should be > 0", sources[i].id, down[i]);
            assert!((up[i] + down[i]).abs() <= 1e-12 * up[i].abs(), "{}: |P_g| differs", sources[i].id);
            assert_eq!(classify(up[i], up[i].abs()), Classification::Hurting);
            assert_eq!(classify(down[i], down[i].abs()), Classification::Helping);
        }
    }

    const FOURBAR_OMEGA: f64 = 1.5;
    const MOUNT_DEG: f64 = 30.0;

    /// Crank-rocker 4-bar (crank 1, coupler 3, rocker 2.5, ground 4 m) with
    /// base masses on every link, W1 "Robot" (5 kg, off the coupler line)
    /// and W2 (2 kg at the rocker tip), gravity rotated by a 30 deg mounting
    /// angle so both gravity components matter.
    fn fourbar_blueprint() -> MechanismJson {
        let mount = MOUNT_DEG.to_radians();
        let g = [-G * mount.sin(), -G * mount.cos()];
        serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {"O2": [0.0, 0.0], "O4": [4.0, 0.0]},
                           "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "crank": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                          "mass": 1.0, "cg_local": [0.5, 0.0], "izz_cg": 0.1},
                "coupler": {"attachment_points": {"B": [0.0, 0.0], "C": [3.0, 0.0]},
                            "mass": 3.0, "cg_local": [1.5, 0.0], "izz_cg": 2.25,
                            "point_masses": [
                                {"id": "W1", "label": "Robot", "mass": 5.0, "local_pos": [1.0, 0.4]}
                            ]},
                "rocker": {"attachment_points": {"D": [0.0, 0.0], "C": [2.5, 0.0]},
                           "mass": 2.0, "cg_local": [1.25, 0.0], "izz_cg": 1.0,
                           "point_masses": [
                               {"id": "W2", "mass": 2.0, "local_pos": [2.5, 0.0]}
                           ]}
            },
            "joints": {
                "J1": {"type": "revolute", "body_i": "ground", "point_i": "O2",
                       "body_j": "crank", "point_j": "A"},
                "J2": {"type": "revolute", "body_i": "crank", "point_i": "B",
                       "body_j": "coupler", "point_j": "B"},
                "J3": {"type": "revolute", "body_i": "coupler", "point_i": "C",
                       "body_j": "rocker", "point_j": "C"},
                "J4": {"type": "revolute", "body_i": "ground", "point_i": "O4",
                       "body_j": "rocker", "point_j": "D"}
            },
            "drivers": {
                "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "crank",
                       "omega": FOURBAR_OMEGA, "theta_0": 0.0}
            },
            "forces": [{"type": "Gravity", "g_vector": g}]
        }))
        .expect("4-bar blueprint parses")
    }

    /// Newton solve of the 4-bar at crank angle `theta` from `guess`.
    fn solve_fourbar_near(mech: &Mechanism, guess: &DVector<f64>, theta: f64) -> DVector<f64> {
        let solved = solve_position(mech, guess, theta / FOURBAR_OMEGA, 1e-12, 50).expect("position solve");
        assert!(solved.converged, "4-bar pose at {theta} rad did not converge");
        solved.q
    }

    /// Solved elbow-up pose of the 4-bar at crank angle `theta`, seeded from
    /// the closed-form loop closure.
    fn fourbar_q(mech: &Mechanism, theta: f64) -> DVector<f64> {
        let (bx, by) = (theta.cos(), theta.sin());
        let (dx, dy) = (4.0 - bx, -by);
        let d = (dx * dx + dy * dy).sqrt();
        // Distance from B towards O4 to the foot of C, and C's height above B->O4.
        let a = (d * d + 3.0 * 3.0 - 2.5 * 2.5) / (2.0 * d);
        let h = (3.0 * 3.0 - a * a).sqrt();
        let (ux, uy) = (dx / d, dy / d);
        let (cx, cy) = (bx + a * ux - h * uy, by + a * uy + h * ux);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("crank", &mut q, 0.0, 0.0, theta);
        st.set_pose("coupler", &mut q, bx, by, (cy - by).atan2(cx - bx));
        st.set_pose("rocker", &mut q, 4.0, 0.0, cy.atan2(cx - 4.0));
        solve_fourbar_near(mech, &q, theta)
    }

    /// Potential energy `-m g . r` of source `s` at pose `q`.
    fn potential_energy(mech: &Mechanism, s: &WeightSource, q: &DVector<f64>, g: [f64; 2]) -> f64 {
        let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
        let r = mech.state().body_point_global(&s.body_id, &local, q);
        -s.mass * (g[0] * r.x + g[1] * r.y)
    }

    /// Independent reference: P_g is minus the rate of change of each
    /// weight's potential energy, finite-differenced over a small crank step
    /// (positions only, no velocity solve), for every link self-weight and
    /// point mass of the 4-bar under tilted gravity.
    #[test]
    fn gravity_power_matches_potential_energy_finite_difference() {
        let bp = fourbar_blueprint();
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        assert_eq!(sources.len(), 5, "3 link self-weights + 2 point masses");
        let g = gravity_vector(&mech);
        let h = 1e-3;
        let mut rows: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        for deg in (0..360).step_by(20) {
            let theta = (deg as f64).to_radians();
            let q = fourbar_q(&mech, theta);
            let q_dot = solve_velocity(&mech, &q, theta / FOURBAR_OMEGA).expect("velocity solve");
            let q_plus = solve_fourbar_near(&mech, &q, theta + h);
            let q_minus = solve_fourbar_near(&mech, &q, theta - h);
            let powers = gravity_powers(&mech, &sources, &q, &q_dot, g);
            let fd: Vec<f64> = sources
                .iter()
                .map(|s| {
                    let d_pe = potential_energy(&mech, s, &q_plus, g) - potential_energy(&mech, s, &q_minus, g);
                    -FOURBAR_OMEGA * d_pe / (2.0 * h)
                })
                .collect();
            rows.push((deg, powers, fd));
        }
        for (i, s) in sources.iter().enumerate() {
            let peak = rows.iter().fold(0.0_f64, |m, r| m.max(r.1[i].abs()));
            assert!(peak > 1.0, "{}: implausibly small peak gravity power {peak}", s.id);
            for (deg, powers, fd) in &rows {
                assert!(
                    (powers[i] - fd[i]).abs() <= 1e-5 * peak,
                    "{} at {deg} deg: P_g {} vs energy finite difference {}",
                    s.id,
                    powers[i],
                    fd[i]
                );
            }
        }
    }

    /// The link self-weights plus the point masses reproduce the built
    /// mechanism's total gravity power `Q_gravity . q_dot` (composite masses
    /// and CGs), and the point masses are a real part of that sum.
    #[test]
    fn link_self_weights_and_point_masses_sum_to_model_gravity_power() {
        let bp = fourbar_blueprint();
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        let g = gravity_vector(&mech);
        let gravity = GravityElement { g_vector: g };
        let mut point_masses_matter = false;
        for deg in (0..360).step_by(15) {
            let theta = (deg as f64).to_radians();
            let q = fourbar_q(&mech, theta);
            let q_dot = solve_velocity(&mech, &q, theta / FOURBAR_OMEGA).expect("velocity solve");
            let powers = gravity_powers(&mech, &sources, &q, &q_dot, g);
            let model = evaluate_gravity(&gravity, mech.state(), mech.bodies(), &q).dot(&q_dot);
            let sum: f64 = powers.iter().sum();
            let scale: f64 = powers.iter().map(|p| p.abs()).sum::<f64>().max(1e-12);
            assert!(
                (sum - model).abs() <= 1e-9 * scale,
                "{deg} deg: sum of weight powers {sum} vs model gravity power {model}"
            );
            let links_only: f64 = sources
                .iter()
                .zip(&powers)
                .filter(|(s, _)| s.is_link_self_weight)
                .map(|(_, p)| p)
                .sum();
            point_masses_matter |= (links_only - model).abs() > 0.1 * scale;
        }
        assert!(point_masses_matter, "the point masses never changed the total: test is vacuous");
    }

    #[test]
    fn weight_sources_lists_link_self_weights_then_applied_point_masses() {
        let bp: MechanismJson = serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {}, "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0,
                           "point_masses": [{"id": "W9", "mass": 1.0, "local_pos": [0.0, 0.0]}]},
                "bar_d": {"attachment_points": {}, "mass": -1.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "bar_c": {"attachment_points": {}, "mass": 1.0, "cg_local": [0.1, 0.2], "izz_cg": 0.0,
                          "point_masses": [
                              {"id": "W4", "mass": 0.0, "local_pos": [0.0, 0.0]},
                              {"id": "W5", "label": "  ", "mass": 2.0, "local_pos": [1.0, 0.0]}
                          ]},
                "bar_a": {"attachment_points": {}, "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0,
                          "point_masses": [{"id": "W3", "mass": 4.0, "local_pos": [0.5, 0.5]}]},
                "bar_b": {"attachment_points": {}, "mass": 2.5, "cg_local": [0.5, 0.0], "izz_cg": 0.0,
                          "label": "Arm",
                          "point_masses": [
                              {"id": "W2", "label": "Robot", "mass": 50.0, "local_pos": [1.0, 0.0]},
                              {"id": "W1", "mass": 3.0, "local_pos": [0.2, 0.0]}
                          ]}
            },
            "joints": {}
        }))
        .expect("blueprint parses");
        let sources = weight_sources(&bp);
        let summary: Vec<(&str, &str, &str, bool)> = sources
            .iter()
            .map(|s| (s.id.as_str(), s.name.as_str(), s.body_id.as_str(), s.is_link_self_weight))
            .collect();
        // Link self-weights first (bodies sorted by id; massless, negative-mass
        // and ground bodies have none), then applied point masses (bodies
        // sorted by id, list order; ground and zero-mass weights skipped).
        assert_eq!(
            summary,
            vec![
                ("link:bar_b", "Arm", "bar_b", true),
                ("link:bar_c", "bar_c", "bar_c", true),
                ("W3", "W3", "bar_a", false),
                ("W2", "Robot", "bar_b", false),
                ("W1", "W1", "bar_b", false),
                ("W5", "W5", "bar_c", false),
            ]
        );
        assert_eq!((sources[1].local_pos, sources[1].mass), ([0.1, 0.2], 1.0), "self-weight at base CG");
        assert_eq!((sources[3].local_pos, sources[3].mass), ([1.0, 0.0], 50.0), "point mass as stored");
    }

    #[test]
    fn gravity_power_is_zero_on_ground_and_nan_for_a_body_missing_from_the_mechanism() {
        let bp = single_bar_blueprint(2.0);
        let mech = build(&bp);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("bar", &mut q, 0.0, 0.0, 0.3);
        let q_dot = solve_velocity(&mech, &q, 0.15).expect("velocity solve");
        let source = |body: &str| WeightSource {
            id: "X".to_string(),
            name: "X".to_string(),
            body_id: body.to_string(),
            local_pos: [1.0, 0.0],
            mass: 1.0,
            is_link_self_weight: false,
        };
        let powers = gravity_powers(&mech, &[source("ground"), source("ghost"), source("bar")], &q, &q_dot, [0.0, -G]);
        assert_eq!(powers[0], 0.0, "ground does not move");
        assert!(powers[1].is_nan(), "a body the mechanism lacks gives NaN, not a panic");
        assert!(powers[2].is_finite() && powers[2] != 0.0);
    }

    #[test]
    fn gravity_vector_reads_the_mechanism_gravity_elements() {
        let mut mech = build(&fourbar_blueprint());
        let mount = MOUNT_DEG.to_radians();
        let g = gravity_vector(&mech);
        assert!((g[0] + G * mount.sin()).abs() < 1e-12 && (g[1] + G * mount.cos()).abs() < 1e-12, "{g:?}");

        mech.add_force(ForceElement::Gravity(GravityElement { g_vector: [1.0, 2.0] }));
        let summed = gravity_vector(&mech);
        assert!((summed[0] - (g[0] + 1.0)).abs() < 1e-12 && (summed[1] - (g[1] + 2.0)).abs() < 1e-12);

        let mut no_gravity = single_bar_blueprint(2.0);
        no_gravity.forces.clear();
        assert_eq!(gravity_vector(&build(&no_gravity)), [0.0, 0.0]);
    }

    #[test]
    fn classify_uses_a_one_percent_neutral_band() {
        use Classification::*;
        assert_eq!(classify(5.0, 100.0), Helping);
        assert_eq!(classify(-5.0, 100.0), Hurting);
        assert_eq!(classify(1.1, 100.0), Helping, "just above the 1 % band");
        assert_eq!(classify(-1.1, 100.0), Hurting, "just below minus the band");
        assert_eq!(classify(0.9, 100.0), Neutral, "inside the band");
        assert_eq!(classify(-0.9, 100.0), Neutral, "inside the band");
        assert_eq!(classify(0.0, 0.0), Neutral, "a weight that never moves vertically");
        assert_eq!(classify(f64::NAN, 100.0), Neutral);
        assert_eq!(classify(f64::INFINITY, 100.0), Neutral);
    }

    #[test]
    fn force_share_divides_by_rate_and_is_nan_near_reversal() {
        assert_eq!(force_share(10.0, 2.0, 4.0), -5.0);
        let above_band = force_share(-10.0, -0.05, 4.0);
        assert!((above_band - (-200.0)).abs() < 1e-9, "|rate| 0.05 >= 1 % of 4: {above_band}");
        assert!(force_share(10.0, 0.03, 4.0).is_nan(), "|rate| < 1 % of max: near reversal");
        assert!(force_share(10.0, -0.03, 4.0).is_nan());
        assert!(force_share(10.0, 0.0, 0.0).is_nan(), "no motion at all");
        assert!(force_share(10.0, f64::NAN, 4.0).is_nan());
        assert!(force_share(10.0, f64::INFINITY, 4.0).is_nan());
        assert!(force_share(f64::NAN, 2.0, 4.0).is_nan());
    }

    #[test]
    fn is_braking_needs_power_below_the_relative_tolerance() {
        assert!(is_braking(-1.0, 100.0));
        assert!(!is_braking(-1e-5, 100.0), "within 1e-6 * 100 of zero");
        assert!(!is_braking(0.0, 100.0));
        assert!(!is_braking(5.0, 100.0));
        assert!(!is_braking(f64::NAN, 100.0));
    }

    #[test]
    fn max_abs_finite_skips_non_finite_values() {
        assert_eq!(max_abs_finite(&[1.0, -3.0, f64::NAN, 2.0, f64::INFINITY]), 3.0);
        assert_eq!(max_abs_finite(&[]), 0.0);
        assert_eq!(max_abs_finite(&[f64::NAN, f64::NEG_INFINITY]), 0.0);
    }
}
```

Create `linkage-sim-rs/tests/gravity_breakdown_reference.rs`:

```rust
//! Payload weights (spec Track 2, section 2): independent reference for the
//! per-weight gravity power on the two actuator samples.
//!
//! Probe method from the spec: each weight's `P_g = m g . v` (velocity from
//! the solver's velocity solve) must equal minus the rate of change of its
//! potential energy `-m g . r`, finite-differenced over a small crank step
//! from positions alone. The link self-weights plus the point masses must
//! also add up to the built mechanism's total gravity power
//! `Q_gravity . q_dot` (composite masses; the massless compound actuator
//! bodies a rebuild adds must not break the sum).

use linkage_sim_rs::analysis::gravity_breakdown::{
    gravity_powers, gravity_vector, weight_sources, WeightSource,
};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{evaluate_gravity, GravityElement};
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;
use linkage_sim_rs::solver::kinematics::{solve_position, solve_velocity};
use nalgebra::{DVector, Vector2};

/// Crank step for the central difference (rad).
const H: f64 = 1e-3;

/// Load `sample` and add `weights` (body, kg, body-local position in m);
/// each add rebuilds the mechanism from the blueprint.
fn sample_with_weights(sample: SampleMechanism, weights: &[(&str, f64, [f64; 2])]) -> AppState {
    let mut state = AppState::default();
    state.load_sample(sample);
    for &(body, mass, pos) in weights {
        state.add_point_mass(body, mass, pos).expect("weight added");
    }
    state.sync_gravity();
    state
}

/// Newton solve at driver angle `theta` from `guess`.
fn solve_at(state: &AppState, mech: &Mechanism, guess: &DVector<f64>, theta: f64) -> DVector<f64> {
    let t = (theta - state.driver_theta_0()) / state.driver_omega();
    let solved = solve_position(mech, guess, t, 1e-12, 50).expect("position solve");
    assert!(solved.converged, "no pose at {theta} rad");
    solved.q
}

/// Potential energy `-m g . r` of source `s` at pose `q`.
fn potential_energy(mech: &Mechanism, s: &WeightSource, q: &DVector<f64>, g: [f64; 2]) -> f64 {
    let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
    let r = mech.state().body_point_global(&s.body_id, &local, q);
    -s.mass * (g[0] * r.x + g[1] * r.y)
}

struct Row {
    deg: usize,
    powers: Vec<f64>,
    energy_fd: Vec<f64>,
    model_total: f64,
}

fn assert_matches_energy_reference(sample: SampleMechanism, weights: &[(&str, f64, [f64; 2])]) {
    let state = sample_with_weights(sample, weights);
    let mech = state.mechanism.as_ref().expect("mechanism built");
    let sources = weight_sources(state.blueprint.as_ref().expect("blueprint"));
    assert_eq!(
        sources.iter().filter(|s| !s.is_link_self_weight).count(),
        weights.len(),
        "{sample:?}: every added weight is a source"
    );
    assert!(sources.iter().any(|s| s.is_link_self_weight), "{sample:?}: links have mass");
    let g = gravity_vector(mech);
    assert!(g[0].abs() < 1e-12 && (g[1] + 9.81).abs() < 1e-12, "{sample:?}: gravity {g:?}");
    let gravity = GravityElement { g_vector: g };
    let omega = state.driver_omega();

    // Odd multiples of 5 deg: the parallelogram is a change point (all links
    // collinear, velocity not unique) at exactly 0 and 180 deg.
    let mut q = state.q.clone();
    let mut rows = Vec::new();
    for deg in (5..360).step_by(10) {
        let theta = (deg as f64).to_radians();
        q = solve_at(&state, mech, &q, theta);
        let t = (theta - state.driver_theta_0()) / omega;
        let q_dot = solve_velocity(mech, &q, t).expect("velocity solve");
        let q_plus = solve_at(&state, mech, &q, theta + H);
        let q_minus = solve_at(&state, mech, &q, theta - H);
        let energy_fd = sources
            .iter()
            .map(|s| {
                let d_pe = potential_energy(mech, s, &q_plus, g) - potential_energy(mech, s, &q_minus, g);
                -omega * d_pe / (2.0 * H)
            })
            .collect();
        rows.push(Row {
            deg,
            powers: gravity_powers(mech, &sources, &q, &q_dot, g),
            energy_fd,
            model_total: evaluate_gravity(&gravity, mech.state(), mech.bodies(), &q).dot(&q_dot),
        });
    }

    for (i, s) in sources.iter().enumerate() {
        let peak = rows.iter().fold(0.0_f64, |m, r| m.max(r.powers[i].abs()));
        assert!(peak > 0.0, "{sample:?} {}: weight never moves vertically", s.id);
        for r in &rows {
            assert!(
                (r.powers[i] - r.energy_fd[i]).abs() <= 1e-5 * peak,
                "{sample:?} {} at {} deg: P_g {} vs energy finite difference {}",
                s.id,
                r.deg,
                r.powers[i],
                r.energy_fd[i]
            );
        }
    }
    for r in &rows {
        let sum: f64 = r.powers.iter().sum();
        let scale: f64 = r.powers.iter().map(|p| p.abs()).sum::<f64>().max(1e-12);
        assert!(
            (sum - r.model_total).abs() <= 1e-9 * scale,
            "{sample:?} at {} deg: sum of weight powers {sum} vs model gravity power {}",
            r.deg,
            r.model_total
        );
    }
}

#[test]
fn parallelogram_actuator_gravity_power_matches_energy_reference() {
    assert_matches_energy_reference(
        SampleMechanism::ParallelogramActuator,
        &[("rocker", 50.0, [0.0, 0.0]), ("coupler", 20.0, [2.0, 0.5])],
    );
}

#[test]
fn chebyshev_lambda_actuator_gravity_power_matches_energy_reference() {
    assert_matches_energy_reference(
        SampleMechanism::ChebyshevLambdaActuator,
        &[("coupler", 50.0, [0.1838, 0.0]), ("rocker", 5.0, [0.0, 0.02])],
    );
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib gravity_breakdown
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --test gravity_breakdown_reference
```

Expected: neither compiles, because none of the items exist yet. rustc stops at import resolution, so the lib test prints only ``error[E0432]: unresolved import `Classification` `` (from the function-local `use Classification::*;` in `classify_uses_a_one_percent_neutral_band`), and the integration test prints `error[E0432]: unresolved imports` naming `gravity_powers`, `gravity_vector`, `weight_sources` and `WeightSource`. There are no E0412 or E0425 errors at this stage: they appear only after the imports resolve.

- [ ] **Step 3: Implement**

In `linkage-sim-rs/src/analysis/gravity_breakdown.rs`, insert between the module doc (ending `//! (Track 2, section 2).`) and `#[cfg(test)]`, keeping one blank line on each side:

```rust
use nalgebra::{DVector, Vector2};
use serde::{Deserialize, Serialize};

use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::io::{point_mass_skip_reason, MechanismJson};

/// Force shares are NaN where `|rate| < EPS_REL_LDOT * max|rate|` over the
/// sweep: near stroke reversal the actuator barely moves, so `-P_g / rate`
/// blows up although the force needed to hold the load does not.
pub const EPS_REL_LDOT: f64 = 0.01;

/// The actuator is braking (the load drives it) where its power is below
/// `-BRAKE_TOL_REL * max|P_act|` over the sweep; the tolerance keeps
/// round-off around zero power from flipping the label.
pub const BRAKE_TOL_REL: f64 = 1e-6;

/// A weight is neutral while `|P_g,i| <= NEUTRAL_REL * max|P_g,i|` over the
/// sweep for that weight (it is momentarily moving horizontally).
pub const NEUTRAL_REL: f64 = 0.01;

/// One weight the breakdown reports on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightSource {
    /// `"link:<body_id>"` for a link self-weight, else the point-mass id.
    pub id: String,
    /// Display name: the body label (or id) for a link self-weight, the
    /// point-mass label (or id) for a point mass.
    pub name: String,
    /// Body the weight rides on.
    pub body_id: String,
    /// Position in the body's local frame (m).
    pub local_pos: [f64; 2],
    /// Mass (kg), positive and finite.
    pub mass: f64,
    /// True for the link's own mass, false for a point mass.
    pub is_link_self_weight: bool,
}

/// Whether gravity on a weight helps or hurts the actuator at one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// Coming down: gravity does positive work (`P_g` above the band).
    Helping,
    /// Going up: the actuator lifts it (`P_g` below minus the band).
    Hurting,
    /// Moving (nearly) horizontally, or no finite power.
    Neutral,
}

/// `label` when it has visible text, else `fallback`.
fn display_name(label: Option<&String>, fallback: &str) -> String {
    match label {
        Some(text) if !text.trim().is_empty() => text.clone(),
        _ => fallback.to_string(),
    }
}

/// Every weight in blueprint `bp`, in a fixed order: first one link
/// self-weight per non-ground body whose base mass is positive and finite
/// (bodies sorted by id), then every point mass the loader applies (bodies
/// sorted by id, list order).
///
/// Point masses the loader skips (`io::point_mass_skip_reason`: on ground,
/// mass not positive and finite, non-finite position) are left out, so the
/// sources' gravity loads add up to the gravity load of the mechanism built
/// from `bp`; pass that mechanism to [`gravity_powers`]. A negative base
/// mass is invalid input and gets no source; whatever gravity load it adds
/// to the built mechanism shows up in the breakdown's non-gravity remainder.
pub fn weight_sources(bp: &MechanismJson) -> Vec<WeightSource> {
    let mut body_ids: Vec<&String> = bp.bodies.keys().filter(|id| id.as_str() != GROUND_ID).collect();
    body_ids.sort();

    let mut sources = Vec::new();
    for &body_id in &body_ids {
        let body = &bp.bodies[body_id];
        if body.mass.is_finite() && body.mass > 0.0 {
            sources.push(WeightSource {
                id: format!("link:{body_id}"),
                name: display_name(body.label.as_ref(), body_id),
                body_id: body_id.clone(),
                local_pos: body.cg_local,
                mass: body.mass,
                is_link_self_weight: true,
            });
        }
    }
    for &body_id in &body_ids {
        for pm in &bp.bodies[body_id].point_masses {
            if point_mass_skip_reason(body_id, pm).is_some() {
                continue;
            }
            sources.push(WeightSource {
                id: pm.id.clone(),
                name: display_name(pm.label.as_ref(), &pm.id),
                body_id: body_id.clone(),
                local_pos: pm.local_pos,
                mass: pm.mass,
                is_link_self_weight: false,
            });
        }
    }
    sources
}

/// Gravity power `m_i * (g . v_i)` (W) of each source at pose `q` with
/// velocity `q_dot`; positive = gravity does positive work (helping).
///
/// `mech` must be built from the blueprint the sources came from, and `g`
/// is the gravity vector (m/s^2), normally [`gravity_vector`] of `mech`. A
/// source on ground gets 0 (ground does not move); a source whose body is
/// not in `mech` (stale blueprint) gets NaN instead of a panic.
pub fn gravity_powers(
    mech: &Mechanism,
    sources: &[WeightSource],
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    g: [f64; 2],
) -> Vec<f64> {
    let state = mech.state();
    let g = Vector2::new(g[0], g[1]);
    sources
        .iter()
        .map(|s| {
            if !state.is_ground(&s.body_id) && state.get_index(&s.body_id).is_err() {
                return f64::NAN;
            }
            let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
            let v = state.body_point_velocity(&s.body_id, &local, q, q_dot);
            s.mass * g.dot(&v)
        })
        .collect()
}

/// The mechanism's gravity vector (m/s^2): the sum of its `Gravity` force
/// elements (the GUI keeps one, rotated by the mounting angle in
/// `AppState::sync_gravity`); zero when gravity is off.
pub fn gravity_vector(mech: &Mechanism) -> [f64; 2] {
    mech.forces().iter().fold([0.0, 0.0], |sum, force| match force {
        ForceElement::Gravity(g) => [sum[0] + g.g_vector[0], sum[1] + g.g_vector[1]],
        _ => sum,
    })
}

/// Largest finite `|value|` in `values`; 0 when none is finite.
pub fn max_abs_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0, |max, v| max.max(v.abs()))
}

/// Classify one weight at one sample from its gravity power `p_g` and the
/// largest `|P_g|` of the same weight over the sweep: helping above the
/// neutral band `NEUTRAL_REL * max_abs_p_g`, hurting below minus the band,
/// neutral inside it or when `p_g` is not finite.
pub fn classify(p_g: f64, max_abs_p_g: f64) -> Classification {
    if !p_g.is_finite() {
        return Classification::Neutral;
    }
    let band = NEUTRAL_REL * max_abs_p_g;
    if p_g > band {
        Classification::Helping
    } else if p_g < -band {
        Classification::Hurting
    } else {
        Classification::Neutral
    }
}

/// Force share `-p_g / rate` of one weight at one sample.
///
/// `rate` is the actuator extension rate dL/dt (m/s), or the driver rate
/// when the shares are driver-torque shares; `max_abs_rate` is its largest
/// finite `|value|` over the sweep. NaN near stroke reversal
/// (`|rate| < EPS_REL_LDOT * max_abs_rate`) and at a zero or non-finite
/// rate.
pub fn force_share(p_g: f64, rate: f64, max_abs_rate: f64) -> f64 {
    if !rate.is_finite() || rate == 0.0 || rate.abs() < EPS_REL_LDOT * max_abs_rate {
        return f64::NAN;
    }
    -p_g / rate
}

/// True when actuator power `p_act` is negative beyond the tolerance
/// `BRAKE_TOL_REL * max_abs_p_act` (the load drives the actuator); false
/// for NaN.
pub fn is_braking(p_act: f64, max_abs_p_act: f64) -> bool {
    p_act < -BRAKE_TOL_REL * max_abs_p_act
}
```

Docs. Update the project docs in the same commit.

Find in `docs/ai/02-system.yaml`:

```yaml
      virtual work, crank selection, force breakdown
```

Replace with:

```yaml
      virtual work, crank selection, force breakdown, per-weight gravity
      breakdown (gravity_breakdown.rs)
```

Find in `docs/ai/02-system.yaml`:

```yaml
    io::next_point_mass_id. Address weights by (body_id, weight_id), never
    by list index.
```

Add after it (the new invariant goes directly before `- point_mass_loader_validation`):

```yaml
  - gravity_breakdown_sources_sum_to_model_gravity — analysis::gravity_breakdown::weight_sources
    lists one link self-weight per non-ground body with positive finite base
    mass (base mass at base cg_local, id "link:<body>") plus every point mass
    the loader applies (the io::point_mass_skip_reason filter that
    io::from_json::apply_point_masses uses). Gravity is linear and the
    composite body is the mass-weighted sum, so the sources' gravity powers
    m g.v add up to the built mechanism's Q_gravity.q_dot (tested). Read g
    from the mechanism (gravity_vector), never recompute it from
    gravity_magnitude: the element carries the mounting angle.
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    files: [validation, grashof, transmission, coupler, energy, envelopes, motor_sizing, virtual_work, crank_selection, force_breakdown]
```

Replace with:

```yaml
    files: [validation, grashof, transmission, coupler, energy, envelopes, motor_sizing, virtual_work, crank_selection, force_breakdown, gravity_breakdown]
    notes:
      gravity_breakdown: per-weight gravity power P_g,i = m_i g.v_i (weight_sources from the blueprint = link self-weights + applied point masses; gravity_powers; gravity_vector; classify / force_share / is_braking with EPS_REL_LDOT, NEUTRAL_REL, BRAKE_TOL_REL). Pure, no GUI deps.
```

Find in `docs/ai/05-update-tracker.md`:

```markdown
## 2026-09-29 — Payload weights Task 2: id-addressed weight editing API
```

Replace with (the new entry goes above the Task 2 entry, separated by a blank line):

```markdown
## 2026-09-29 — Payload weights Task 3: gravity breakdown physics module
- New pure module `analysis/gravity_breakdown.rs` (spec Track 2, section 2):
  `weight_sources(&MechanismJson)` (link self-weights `link:<body>` at base
  mass/CG for positive finite base mass, then every point mass the loader
  applies; bodies sorted by id), `gravity_powers` (`P_g,i = m_i g . v_i`
  from `State::body_point_velocity`; 0 on ground, NaN for a body missing
  from the mechanism), `gravity_vector` (sum of the mechanism's Gravity
  elements, so the mounting angle is respected), and the named rules
  `classify` (`NEUTRAL_REL` = 0.01), `force_share` (NaN when
  `|rate| < EPS_REL_LDOT * max|rate|`, `EPS_REL_LDOT` = 0.01),
  `is_braking` (`BRAKE_TOL_REL` = 1e-6), `max_abs_finite`.
- Tests: `analysis::gravity_breakdown::tests` (single bar hand calculation
  rising / falling / horizontal; sign flip with direction; potential-energy
  finite difference on a 4-bar under 30 deg tilted gravity; sources sum to
  the built `Q_gravity . q_dot`; source ordering and skip rules; rule edge
  cases) and `tests/gravity_breakdown_reference.rs` (energy finite
  difference + sum check on ParallelogramActuator and ChebyshevLambdaActuator
  with two weights each, compound actuator bodies included).
- Mutation check done: negating `g` inside `gravity_powers` fails 4 unit
  tests and both integration tests.

## 2026-09-29 — Payload weights Task 2: id-addressed weight editing API
```

Find in `docs/architecture/ARCHITECTURE.md`:

```markdown
In the Rust JSON schema a point mass ("weight") lives in its body's `point_masses` list as `{ "id": "W1", "label": "Robot torso", "mass": 50.0, "local_pos": [0.3, 0.0] }` (`label` optional). The `id` is unique across the mechanism; files written before ids existed load unchanged and get `W<n>` ids on load (smallest unused number; bodies sorted by id, list order). The loader skips, and reports in the error panel, a point mass on ground, a mass that is not a positive finite number, and a non-finite position; skipped weights stay in the file so a save writes them back unchanged.
```

Add after it (as its own paragraph, with a blank line between the two):

```markdown
Because gravity is linear in mass and the composite body is the mass-weighted sum, the gravity load of a body is exactly the sum of its link self-weight (base mass at the base CG) and its point masses. `analysis::gravity_breakdown` uses this to report, per weight, the gravity power `P_g = m g . v` (positive = the weight is coming down and helps the actuator); the per-weight shares always add up to the mechanism's total gravity load.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib gravity_breakdown
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --test gravity_breakdown_reference
```

Expected: `test result: ok. 11 passed; 0 failed` for the lib filter and `test result: ok. 2 passed; 0 failed` for the integration test.

Manual mutation check (do not commit it): in `gravity_powers` change `let g = Vector2::new(g[0], g[1]);` to `let g = -Vector2::new(g[0], g[1]);` and rerun both commands. Expected: 4 lib tests FAIL (`gravity_power_single_bar_matches_hand_calculation`, `gravity_power_flips_sign_between_ascending_and_descending`, `gravity_power_matches_potential_energy_finite_difference`, `link_self_weights_and_point_masses_sum_to_model_gravity_power`) and both integration tests FAIL. Restore the line and rerun: all pass again.

Then the full gate:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Expected: `GATE PASS`, with the lib suite at `test result: ok. 796 passed` (the 11 new tests on top of the previous total) and no new clippy warnings (the lib-test clippy warning count stays at 301). The checkout restores the PNGs the test run rewrites (BL-012); `git -C /c/Users/Cole/source/repos/linkage_simulation-payload status --short` should then list only the files this task changes.

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/analysis/gravity_breakdown.rs linkage-sim-rs/src/analysis/mod.rs linkage-sim-rs/tests/gravity_breakdown_reference.rs docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md docs/architecture/ARCHITECTURE.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 3: gravity breakdown physics module

New pure module analysis::gravity_breakdown. weight_sources lists link
self-weights (base mass at base CG) and every point mass the loader
applies. It filters with io::point_mass_skip_reason, the same rule
io::from_json::apply_point_masses uses, so the list matches the physics.
gravity_powers computes P_g,i = m_i g . v_i. gravity_vector reads g from
the mechanism's Gravity elements, so the mounting angle is included. The
named rules classify / force_share / is_braking use the constants
EPS_REL_LDOT, NEUTRAL_REL and BRAKE_TOL_REL.

Tests: a single-bar hand calculation (rising, falling, horizontal), a
potential-energy finite difference on a 4-bar under tilted gravity and on
the Parallelogram/Chebyshev actuator samples with two weights each, and a
check that the sources sum to the built mechanism's gravity power. A
mutation check (negated g) fails 4 unit and 2 integration tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 4: Per-weight gravity breakdown in the sweep

**Files:**
- Create: `linkage-sim-rs/src/gui/sweep/weights.rs` (`WeightBreakdown`, `ShareBasis`, `WeightBreakdownBuilder`, `required_totals`, tests)
- Modify: `linkage-sim-rs/src/gui/sweep/mod.rs` (module and re-exports at the top; `SweepData::weight_breakdown` after `pose_snapshots`; `compute_sweep_data` becomes a wrapper of the new `compute_sweep_data_with_weights`; builder setup after `has_force_zones`; one push per sample in the loop; `push_nan_row` gains a `breakdown` parameter; `weight_breakdown: None` in every other `SweepData` literal; one new test at the end of `mod tests`)
- Modify: `linkage-sim-rs/src/gui/state/blueprint_ops.rs` (imports; `AppState::compute_sweep` passes the weight sources, around line 1419)
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (`set_actuator_stored_force`)
- Modify: `linkage-sim-rs/src/gui/export/csv.rs` (2 test literals), `linkage-sim-rs/src/gui/export/raster.rs` (1 test literal): add `weight_breakdown: None,`
- Modify: `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`

**Interfaces:**
- Consumes (Task 3, `crate::analysis::gravity_breakdown`):
  - `pub struct WeightSource { pub id: String, pub name: String, pub body_id: String, pub local_pos: [f64; 2], pub mass: f64, pub is_link_self_weight: bool }` (`Debug, Clone, PartialEq, Serialize, Deserialize`)
  - `pub enum Classification { Helping, Hurting, Neutral }` (`Debug, Clone, Copy, PartialEq, Eq`)
  - `pub fn weight_sources(bp: &MechanismJson) -> Vec<WeightSource>`
  - `pub fn gravity_powers(mech: &Mechanism, sources: &[WeightSource], q: &DVector<f64>, q_dot: &DVector<f64>, g: [f64; 2]) -> Vec<f64>`
  - `pub fn gravity_vector(mech: &Mechanism) -> [f64; 2]`
  - `pub fn classify(p_g: f64, max_abs_p_g: f64) -> Classification`
  - `pub fn force_share(p_g: f64, rate: f64, max_abs_rate: f64) -> f64`
  - `pub fn is_braking(p_act: f64, max_abs_p_act: f64) -> bool`
  - `pub fn max_abs_finite(values: &[f64]) -> f64`
  - `pub const EPS_REL_LDOT: f64 = 0.01`
- Consumes (Task 2): `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>` (tests).
- Consumes (already on main): `solver::reactions::required_actuator_force(driver_torque: f64, driver_omega: f64, dl_dt: f64, applied_force: f64) -> Option<f64>` (BL-026: the sweep's `actuator_forces` is this required force), `AppState::set_body_mass(&mut self, body_id: &str, mass: f64)` with `sync_live_mass_props` (BL-024: a mass edit keeps the point masses in the live body), `AppState::{compute_sweep, driver_omega, sweep_dirty, mounting_angle}`, `SweepData` and `push_nan_row` in `gui/sweep/mod.rs`, and the pass-1 statics helper `solve_reactions_with_actuator` (its `driver_torque`).
- Produces (`crate::gui::sweep`):
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)] pub enum ShareBasis { ActuatorForce, DriverTorque }`
  - `#[derive(Debug, Clone, Serialize, Deserialize)] pub struct WeightBreakdown { pub sources: Vec<WeightSource>, pub basis: ShareBasis, pub gravity_power: Vec<Vec<f64>>, pub force_share: Vec<Vec<f64>>, pub power_share: Vec<Vec<f64>>, pub other_force: Vec<f64>, pub other_power: Vec<f64>, pub total_force: Vec<f64>, pub total_power: Vec<f64>, pub braking: Vec<bool> }`
  - `impl WeightBreakdown { pub fn classification(&self, source: usize, sample: usize) -> Classification }`
  - `SweepData { #[serde(default, skip_serializing_if = "Option::is_none")] pub weight_breakdown: Option<WeightBreakdown>, .. }`
  - `pub(crate) fn compute_sweep_data_with_weights(mech: &Mechanism, q_start: &DVector<f64>, omega: f64, theta_0: f64, gravity_magnitude: f64, sweep_range: Option<(f64, f64)>, weight_sources: Vec<WeightSource>) -> (SweepData, DVector<f64>)` (`compute_sweep_data` keeps its signature and yields `weight_breakdown: None`)
  - private to `gui::sweep`: `pub(super) struct WeightBreakdownBuilder` (`new`, `push_sample`, `push_nan`, `finish`) and `pub(super) fn required_totals(basis: ShareBasis, driver_torque: f64, omega: f64, rate: f64, actuator_force: f64, stored_force: f64) -> (f64, f64)`
  - `#[cfg(test)] pub(crate) fn set_actuator_stored_force(state: &mut AppState, force: f64)` in `gui/test_support.rs`

**Design notes.**
- Per weight `i` the sweep computes the gravity power `P_g,i = m_i g . v_i` (Task 3's `gravity_powers`, with g from `gravity_vector(mech)`: the mechanism's own `Gravity` element, mounting angle included, not the `gravity_magnitude` parameter). From it come the force share `-P_g,i / (dL/dt)` (N, NaN within 1 % of stroke reversal), the power share `-P_g,i` (W, positive = the actuator spends power lifting the weight), the non-gravity remainder `total - sum(shares)` (force zones, springs, external loads, end stops) and `braking`, true where the total power is below minus a small tolerance.
- **The totals are the required force, taken as is.** Since BL-026 the sweep's `actuator_forces` is already `required_actuator_force(...)`, the force needed to drive the load, in sizing mode (stored force 0) and stored-force mode alike. So `required_totals` returns `total_force = actuator_force` unchanged; adding the stored force on top (the pre-BL-026 workaround) would double count it. `total_power` stays `driver_torque * omega + stored_force * dL/dt`: the same balance multiplied through by `dL/dt`. It equals the plotted `F * dL/dt` wherever that is finite, and it stays finite at stroke reversal, where `required_actuator_force` returns `None` (NaN force, NaN plotted power). That keeps the power shares and braking defined there.
- A failed statics solve pushes a 0 driver torque into `driver_torques` (the plot shows 0). The breakdown reports NaN totals for that sample instead (`pending_reactions.is_some()` is the statics-ok flag). A failed position solve goes through `push_nan_row` and a failed velocity solve through the existing else branch; both push a NaN row, so every series stays aligned with `angles_deg`.
- Basis: with a `LinearActuator` the shares are shares of the actuator force (`ShareBasis::ActuatorForce`, rate = `actuator_speeds`). Without one they are shares of the driver effort (`ShareBasis::DriverTorque`, rate = the driver rate `omega`): N*m for a revolute driver, N for a linear driver in a stroke-mode sweep. The driver basis (and its stroke-mode variant, one test) goes beyond the actuator-only scope the spec describes for v1; it is the same code path. Trajectory mode never fills the breakdown (`None`).
- `weight_sources(blueprint)` reads the blueprint while the sweep reads the built mechanism. A Mass edit on a weighted link goes through `set_body_mass` and `sync_live_mass_props` (BL-024, on main), which re-derive the composite body (base mass plus point masses) without a rebuild, so the two stay consistent. One test guards the sum invariant after such an edit.
- The energy check `sweep_gravity_power_matches_energy_change_between_adjacent_samples` covers both actuator samples and every source, link self-weights included. It rebuilds positions only from the recorded traces plus `body_angles` (no velocity). The tolerances are named constants based on measured truncation error of the central difference: worst relative error 5.1e-5 on the parallelogram (`(1 deg)^2 / 6`, exactly the expected value for circular motion) and 1.2e-3 on the Chebyshev. Doubling the step quadruples both (ratio 3.99 to 4.00), so this is truncation error and not a physics mismatch.
- The test helper `set_actuator_stored_force` lives in `gui/test_support.rs` so later tasks reuse it instead of copying the loop.
- Tests, in `gui::sweep::weights::tests` unless noted:
  - Parallelogram and Chebyshev sizing sweeps with two weights: the shares plus the remainder equal the totals, the totals equal the plotted actuator force and power, and the remainder is about 0 when gravity is the only load.
  - `stored_force_mode_and_sizing_mode_give_identical_breakdowns`: on both actuator samples, a stored-force sweep and a sizing sweep must agree on total and other force and power, on every per-source `gravity_power`, `force_share` and `power_share`, on the classification at every sample and on the braking vector, and in both `total_force` equals the plotted actuator force exactly.
  - `required_totals_take_the_required_force_as_is_with_a_reversal_safe_power` builds its inputs with `required_actuator_force` (sizing, a 3 N stored force, and the reversal case where the force is `None` but the power is still `T * omega`).
  - Helping, hurting and neutral at known poses (parallelogram) and from the traced motion (Chebyshev); braking follows the net gravity power; force shares are NaN exactly near stroke reversal; forced solver failures keep every series aligned (actuator and driver basis); the FourBar driver-torque basis; the mounting angle; hand-checkable builder rows; no sources gives no breakdown.
  - The mass-edit test (BL-024) and, in `gui::sweep::tests`, the stroke-mode linear-driver test.

- [ ] **Step 1: Write the failing tests**

The tests need four things: the module registered, the shared stored-force setter, the new `weights.rs` (module doc plus test module), and one stroke-mode test in `sweep/mod.rs`. The file `weights.rs` is created in two pieces because Step 3 puts the implementation between them, so the file reads doc comment, implementation, tests.

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
mod fourbar;
mod motion_profile;
```

Replace with:

```rust
mod fourbar;
mod motion_profile;
mod weights;
```

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
use crate::core::state::GROUND_ID;
```

Add after it:

```rust
use crate::forces::elements::ForceElement;
```

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
    ids.sort();
    ids
}
```

Add after it (after a blank line, at the end of the file):

```rust
/// Set every LinearActuator's stored force in the blueprint and rebuild
/// (0 = sizing mode). Since BL-026 the sweep reports the required actuator
/// force in both modes.
pub(crate) fn set_actuator_stored_force(state: &mut AppState, force: f64) {
    for element in &mut state.blueprint.as_mut().expect("blueprint").forces {
        if let ForceElement::LinearActuator(la) = element {
            la.force = force;
        }
    }
    state.rebuild();
}
```

Create `linkage-sim-rs/src/gui/sweep/weights.rs` with the module doc comment (Step 3 adds the implementation below it):

```rust
//! Per-weight gravity breakdown of a sweep (payload weights, spec Track 2
//! section 2): which weight helps or hurts the actuator at each sample, by
//! how much in actuator force and in actuator power, and where the load
//! drives the actuator (braking).
//!
//! `compute_sweep_data_with_weights` pushes one row per sample into a
//! [`WeightBreakdownBuilder`]; the rules that need sweep-wide maxima
//! (near-reversal NaN, braking tolerance) run in
//! [`WeightBreakdownBuilder::finish`]. The physics and the named thresholds
//! live in `analysis::gravity_breakdown`.
```

Add after it (after a blank line, at the end of the file) the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{gravity_vector, max_abs_finite, EPS_REL_LDOT};
    use crate::forces::elements::ForceElement;
    use crate::gui::samples::{build_sample, SampleMechanism};
    use crate::gui::state::AppState;
    use crate::gui::sweep::{compute_sweep_data, compute_sweep_data_with_weights, SweepData};
    use crate::gui::test_support::set_actuator_stored_force;
    use crate::solver::reactions::required_actuator_force;
    use Classification::{Helping, Hurting, Neutral};

    /// Tolerance factor for identities that hold up to round-off.
    const ROUND_OFF: f64 = 1e-9;
    /// Tolerance factor for "the weights explain the whole load" when
    /// gravity is the only load: exact by the power balance up to solver
    /// precision, amplified by at most 1 / EPS_REL_LDOT.
    const GRAVITY_ONLY: f64 = 1e-6;
    /// Tolerances of the adjacent-sample energy check, relative to each
    /// source's peak gravity power. The central difference over +-1 deg has
    /// a truncation error of ~(1 deg)^2 / 6 = 5.1e-5 times the squared
    /// harmonic order of the motion. Measured worst cases: 5.1e-5 on the
    /// parallelogram (pure circles) and 1.2e-3 on the Chebyshev (the
    /// coupler's fast return stroke). Doubling the step to 2 deg quadruples
    /// both, so they are truncation error, not a physics mismatch.
    const PARALLELOGRAM_FD_TOL: f64 = 2e-4;
    const CHEBYSHEV_FD_TOL: f64 = 2.5e-3;
    /// Tolerance factor when comparing two separate sweeps: at the
    /// parallelogram's change points (0/180/360 deg, singular Jacobian) the
    /// statics solve amplifies round-off to ~1e-8 relative.
    const CROSS_SWEEP: f64 = 1e-6;

    /// Weights to add: (body, kg, body-local position in m).
    type Weights = &'static [(&'static str, f64, [f64; 2])];

    /// Parallelogram weights: W1 on the rocker tip (rocker-local C, traced by
    /// the sweep as "rocker.C"), W2 on coupler point P (traced "coupler.P").
    const PARALLELOGRAM_WEIGHTS: Weights =
        &[("rocker", 50.0, [0.0, 0.0]), ("coupler", 20.0, [2.0, 0.0])];
    /// Chebyshev weights: W1 on coupler point M (the straight-line tracer,
    /// traced "coupler.M"), W2 off the rocker line.
    const CHEBYSHEV_WEIGHTS: Weights =
        &[("coupler", 50.0, [0.1838, 0.0]), ("rocker", 5.0, [0.0, 0.02])];
    /// FourBar (no actuator) weights.
    const FOURBAR_WEIGHTS: Weights =
        &[("coupler", 1.0, [0.02, 0.005]), ("rocker", 0.5, [0.0, 0.0])];

    fn add_weights(state: &mut AppState, weights: &[(&str, f64, [f64; 2])]) {
        for &(body, mass, pos) in weights {
            state.add_point_mass(body, mass, pos).expect("weight added");
        }
    }

    /// Load `sample`, optionally put its actuator in sizing mode (stored
    /// force 0), add `weights` (body, kg, body-local m), sweep 0..=360 deg.
    fn swept(sample: SampleMechanism, sizing: bool, weights: &[(&str, f64, [f64; 2])]) -> AppState {
        let mut state = AppState::default();
        state.load_sample(sample);
        if sizing {
            set_actuator_stored_force(&mut state, 0.0);
        }
        add_weights(&mut state, weights);
        state.compute_sweep();
        state
    }

    fn breakdown(state: &AppState) -> (&SweepData, &WeightBreakdown) {
        let data = state.sweep_data.as_ref().expect("sweep computed");
        (data, data.weight_breakdown.as_ref().expect("breakdown computed"))
    }

    fn source_ids(b: &WeightBreakdown) -> Vec<&str> {
        b.sources.iter().map(|s| s.id.as_str()).collect()
    }

    fn source_index(b: &WeightBreakdown, id: &str) -> usize {
        b.sources.iter().position(|s| s.id == id).unwrap_or_else(|| panic!("no source {id}"))
    }

    fn sample_at(data: &SweepData, deg: f64) -> usize {
        data.angles_deg
            .iter()
            .position(|&a| (a - deg).abs() < 1e-9)
            .unwrap_or_else(|| panic!("no sample at {deg} deg"))
    }

    fn assert_close(what: &str, data: &SweepData, k: usize, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{what} at {} deg: {got} vs {want} (tol {tol})",
            data.angles_deg[k]
        );
    }

    /// Every per-sample series has one entry per sweep sample.
    fn assert_aligned(data: &SweepData, b: &WeightBreakdown) {
        let n = data.angles_deg.len();
        assert!(n > 0, "empty sweep");
        for (name, per_source) in [
            ("gravity_power", &b.gravity_power),
            ("force_share", &b.force_share),
            ("power_share", &b.power_share),
        ] {
            assert_eq!(per_source.len(), b.sources.len(), "{name}: one series per source");
            for (i, series) in per_source.iter().enumerate() {
                assert_eq!(series.len(), n, "{name}[{}]", b.sources[i].id);
            }
        }
        for (name, len) in [
            ("other_force", b.other_force.len()),
            ("other_power", b.other_power.len()),
            ("total_force", b.total_force.len()),
            ("total_power", b.total_power.len()),
            ("braking", b.braking.len()),
        ] {
            assert_eq!(len, n, "{name}");
        }
    }

    /// Sum invariants of a sweep where gravity is the only load: shares plus
    /// remainder give the totals at every valid sample, and the weights
    /// explain the whole load (remainder ~ 0).
    fn assert_gravity_only_sum_invariants(data: &SweepData, b: &WeightBreakdown) {
        let n = data.angles_deg.len();
        let valid_force: Vec<usize> = (0..n)
            .filter(|&k| b.total_force[k].is_finite() && b.force_share.iter().all(|s| s[k].is_finite()))
            .collect();
        assert!(valid_force.len() > n / 2, "only {} of {n} samples have finite force shares", valid_force.len());
        let peak_force = valid_force.iter().fold(0.0_f64, |m, &k| m.max(b.total_force[k].abs()));
        let peak_power = max_abs_finite(&b.total_power);
        assert!(peak_force > 0.0 && peak_power > 0.0, "the weights load the driver");
        for &k in &valid_force {
            let sum: f64 = b.force_share.iter().map(|s| s[k]).sum();
            assert_close("sum(F_i) + F_other", data, k, sum + b.other_force[k], b.total_force[k], ROUND_OFF * peak_force);
            assert_close("F_other (gravity is the only load)", data, k, b.other_force[k], 0.0, GRAVITY_ONLY * peak_force);
        }
        let mut n_power = 0;
        for k in (0..n).filter(|&k| b.total_power[k].is_finite()) {
            n_power += 1;
            let sum: f64 = b.power_share.iter().map(|s| s[k]).sum();
            assert_close("sum(P_i) + P_other", data, k, sum + b.other_power[k], b.total_power[k], ROUND_OFF * peak_power);
            assert_close("P_other (gravity is the only load)", data, k, b.other_power[k], 0.0, GRAVITY_ONLY * peak_power);
        }
        assert!(n_power >= valid_force.len(), "power is defined wherever force shares are");
    }

    /// The breakdown totals are the plotted actuator series (the required
    /// force in sizing and stored-force mode alike since BL-026).
    fn assert_totals_match_actuator_plot(data: &SweepData, b: &WeightBreakdown) {
        assert_eq!(b.basis, ShareBasis::ActuatorForce);
        let forces = data.actuator_forces.as_ref().expect("actuator forces");
        let powers = data.actuator_power.as_ref().expect("actuator power");
        let peak_power = max_abs_finite(powers);
        for k in 0..data.angles_deg.len() {
            if forces[k].is_finite() {
                assert_eq!(b.total_force[k], forces[k], "total_force at {} deg", data.angles_deg[k]);
            }
            if powers[k].is_finite() {
                assert_close("total_power vs actuator_power", data, k, b.total_power[k], powers[k], ROUND_OFF * peak_power);
            }
        }
    }

    #[test]
    fn parallelogram_sizing_breakdown_sums_to_actuator_force_and_power() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W2", "W1"]);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    #[test]
    fn chebyshev_sizing_breakdown_sums_to_actuator_force_and_power() {
        let state = swept(SampleMechanism::ChebyshevLambdaActuator, true, CHEBYSHEV_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W1", "W2"]);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    /// Stored-force mode and sizing mode move the same way under the same
    /// loads, and since BL-026 the sweep reports the REQUIRED actuator force
    /// in both, so the two breakdowns must be identical: totals, shares,
    /// remainders, braking and classification. In both, the total is the
    /// plotted actuator force exactly. Adding the stored force back on top
    /// (the pre-BL-026 workaround) would double count it and fail here.
    #[test]
    fn stored_force_mode_and_sizing_mode_give_identical_breakdowns() {
        for (sample, weights) in [
            (SampleMechanism::ParallelogramActuator, PARALLELOGRAM_WEIGHTS),
            (SampleMechanism::ChebyshevLambdaActuator, CHEBYSHEV_WEIGHTS),
        ] {
            let stored = swept(sample, false, weights);
            let sizing = swept(sample, true, weights);
            let stored_force = stored
                .mechanism
                .as_ref()
                .unwrap()
                .forces()
                .iter()
                .find_map(|f| match f {
                    ForceElement::LinearActuator(la) => Some(la.force),
                    _ => None,
                })
                .expect("actuator");
            assert!(stored_force.abs() > 1.0, "{sample:?} ships with a stored force");
            let (data, b) = breakdown(&stored);
            let (sizing_data, sizing_b) = breakdown(&sizing);
            assert_eq!(data.angles_deg, sizing_data.angles_deg, "{sample:?}: same samples");
            assert_eq!(source_ids(b), source_ids(sizing_b), "{sample:?}: same sources");
            assert_aligned(data, b);
            assert_totals_match_actuator_plot(data, b);
            assert_gravity_only_sum_invariants(data, b);

            let force_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.total_force).max(stored_force.abs());
            let power_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.total_power);
            let name = |series: &str| format!("{sample:?} {series}");
            assert_series(&name("total_force"), &b.total_force, &sizing_b.total_force, force_tol);
            assert_series(&name("other_force"), &b.other_force, &sizing_b.other_force, force_tol);
            assert_series(&name("total_power"), &b.total_power, &sizing_b.total_power, power_tol);
            assert_series(&name("other_power"), &b.other_power, &sizing_b.other_power, power_tol);
            for (i, s) in b.sources.iter().enumerate() {
                let p_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.gravity_power[i]);
                let f_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.force_share[i]);
                let id = &s.id;
                assert_series(&name(&format!("gravity_power[{id}]")), &b.gravity_power[i], &sizing_b.gravity_power[i], p_tol);
                assert_series(&name(&format!("force_share[{id}]")), &b.force_share[i], &sizing_b.force_share[i], f_tol);
                assert_series(&name(&format!("power_share[{id}]")), &b.power_share[i], &sizing_b.power_share[i], p_tol);
                for k in 0..data.angles_deg.len() {
                    let deg = data.angles_deg[k];
                    assert_eq!(b.classification(i, k), sizing_b.classification(i, k), "{sample:?} {id} at {deg} deg");
                }
            }
            assert_eq!(b.braking, sizing_b.braking, "{sample:?}: braking");
        }
    }

    /// Parallelogram: the rocker turns with the crank, so the rocker tip W1
    /// is at O4 + 2 (cos t, sin t) and, with omega > 0, rises while
    /// cos t > 0. At 135 deg every link and weight is coming down, so
    /// gravity drives the actuator (braking); at 45 deg the actuator lifts
    /// them all (motoring); at 90 deg they all move horizontally.
    #[test]
    fn parallelogram_classification_and_braking_at_known_poses() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        let tip = source_index(b, "W1");
        let (k45, k90, k135) = (sample_at(data, 45.0), sample_at(data, 90.0), sample_at(data, 135.0));
        for i in 0..b.sources.len() {
            let id = &b.sources[i].id;
            assert_eq!(b.classification(i, k45), Hurting, "{id} rising at 45 deg");
            assert_eq!(b.classification(i, k90), Neutral, "{id} moving horizontally at 90 deg");
            assert_eq!(b.classification(i, k135), Helping, "{id} coming down at 135 deg");
        }
        assert!(!b.braking[k45], "the actuator lifts the load at 45 deg");
        assert!(b.braking[k135], "the load drives the actuator at 135 deg");
        assert!(b.power_share[tip][k45] > 0.0, "lifting W1 costs actuator power");
        assert!(b.power_share[tip][k135] < 0.0, "lowering W1 gives power back");
    }

    /// Chebyshev: W1 rides on coupler point M, whose path the sweep traces
    /// as "coupler.M". Wherever M clearly moves up (central difference of the
    /// trace) W1 is hurting, wherever it clearly moves down W1 is helping
    /// (M runs nearly level along its straight-line stretch, so only part of
    /// the cycle qualifies). With gravity the only load, the actuator brakes
    /// exactly where the weights' net gravity power is positive.
    #[test]
    fn chebyshev_classification_follows_the_weight_motion_and_braking_follows_net_gravity_power() {
        let state = swept(SampleMechanism::ChebyshevLambdaActuator, true, CHEBYSHEV_WEIGHTS);
        let (data, b) = breakdown(&state);
        let w1 = source_index(b, "W1");
        let trace = &data.coupler_traces["coupler.M"];
        let n = trace.len();
        let rise: Vec<f64> = (0..n)
            .map(|k| if k == 0 || k + 1 == n { f64::NAN } else { trace[k + 1][1] - trace[k - 1][1] })
            .collect();
        let max_rise = max_abs_finite(&rise);
        let mut checked = 0;
        for k in (0..n).filter(|&k| rise[k].abs() > 0.05 * max_rise) {
            let expected = if rise[k] > 0.0 { Hurting } else { Helping };
            assert_eq!(b.classification(w1, k), expected, "W1 at {} deg (rise {})", data.angles_deg[k], rise[k]);
            checked += 1;
        }
        assert!(checked > n / 4, "only {checked} samples with clear vertical motion");

        let net: Vec<f64> = (0..n).map(|k| b.gravity_power.iter().map(|s| s[k]).sum()).collect();
        let max_net = max_abs_finite(&net);
        let (mut n_braking, mut n_motoring) = (0, 0);
        for k in (0..n).filter(|&k| net[k].abs() > 1e-3 * max_net) {
            assert_eq!(b.braking[k], net[k] > 0.0, "braking at {} deg, net gravity power {}", data.angles_deg[k], net[k]);
            if b.braking[k] {
                n_braking += 1;
            } else {
                n_motoring += 1;
            }
        }
        assert!(n_braking > 0 && n_motoring > 0, "a lift cycle both motors and brakes");
    }

    /// Near stroke reversal (|dL/dt| < 1 % of its sweep maximum) the force
    /// shares and the force remainder are NaN; the power shares, total power
    /// and power remainder stay defined because they do not divide by dL/dt.
    #[test]
    fn force_share_is_nan_exactly_near_stroke_reversal_and_power_share_stays_defined() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        let speeds = data.actuator_speeds.as_ref().expect("actuator speeds");
        let max_speed = max_abs_finite(speeds);
        let mut n_reversal = 0;
        for (k, &ldot) in speeds.iter().enumerate() {
            assert!(ldot.is_finite(), "every parallelogram sample solves (dL/dt at {} deg)", data.angles_deg[k]);
            let near_reversal = ldot.abs() < EPS_REL_LDOT * max_speed;
            n_reversal += usize::from(near_reversal);
            for i in 0..b.sources.len() {
                let id = &b.sources[i].id;
                let deg = data.angles_deg[k];
                assert_eq!(b.force_share[i][k].is_nan(), near_reversal, "{id} force share at {deg} deg (dL/dt {ldot})");
                assert!(b.power_share[i][k].is_finite(), "{id} power share at {deg} deg");
            }
            assert_eq!(b.other_force[k].is_nan(), near_reversal, "other_force at {} deg", data.angles_deg[k]);
            assert!(b.total_power[k].is_finite() && b.other_power[k].is_finite());
        }
        assert!(n_reversal > 0, "the actuator reverses twice per revolution");
    }

    /// Stretch the ground link (move O4) until the crank cannot pass the far
    /// side, so a band of samples fails to solve (O4 is also moved off the
    /// x axis where needed: from the parallelogram's collinear start pose a
    /// symmetric target never leaves the axis). Every breakdown series still
    /// has one entry per sample: NaN, neutral and not braking where the
    /// sample failed, finite gravity power everywhere else.
    fn assert_forced_failures_stay_aligned(
        sample: SampleMechanism,
        o4: [f64; 2],
        sizing: bool,
        weights: &[(&str, f64, [f64; 2])],
        basis: ShareBasis,
    ) {
        let mut state = AppState::default();
        state.load_sample(sample);
        if sizing {
            set_actuator_stored_force(&mut state, 0.0);
        }
        state
            .blueprint
            .as_mut()
            .unwrap()
            .bodies
            .get_mut("ground")
            .unwrap()
            .attachment_points
            .insert("O4".to_string(), o4);
        state.rebuild();
        assert!(state.solver_status.converged, "{sample:?}: 0 deg must still assemble");
        add_weights(&mut state, weights);
        state.compute_sweep();
        let (data, b) = breakdown(&state);
        assert_eq!(b.basis, basis);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        let n = data.angles_deg.len();
        let failed: Vec<usize> = (0..n).filter(|&k| data.body_angles["crank"][k].is_nan()).collect();
        assert!(failed.len() > 90, "{sample:?}: expected an unreachable band, {} samples failed", failed.len());
        for &k in &failed {
            let deg = data.angles_deg[k];
            for i in 0..b.sources.len() {
                assert!(b.gravity_power[i][k].is_nan(), "{sample:?} gravity_power at {deg} deg");
                assert!(b.force_share[i][k].is_nan(), "{sample:?} force_share at {deg} deg");
                assert!(b.power_share[i][k].is_nan(), "{sample:?} power_share at {deg} deg");
                assert_eq!(b.classification(i, k), Neutral, "{sample:?} classification at {deg} deg");
            }
            for (name, v) in [
                ("other_force", b.other_force[k]),
                ("other_power", b.other_power[k]),
                ("total_force", b.total_force[k]),
                ("total_power", b.total_power[k]),
            ] {
                assert!(v.is_nan(), "{sample:?} {name} at {deg} deg");
            }
            assert!(!b.braking[k], "{sample:?} braking at {deg} deg");
        }
        for k in (0..n).filter(|k| !failed.contains(k)) {
            assert!(
                b.gravity_power.iter().all(|s| s[k].is_finite()),
                "{sample:?}: converged sample at {} deg has a non-finite gravity power",
                data.angles_deg[k]
            );
        }
    }

    #[test]
    fn forced_solver_failures_keep_the_breakdown_aligned_actuator_basis() {
        // |O4| = 5.46: |B - O4| <= coupler 4 + rocker 2 only for crank angles
        // of about -87..104 deg, so about 105..272 deg is unreachable.
        assert_forced_failures_stay_aligned(
            SampleMechanism::ParallelogramActuator,
            [5.4, 0.8],
            true,
            PARALLELOGRAM_WEIGHTS,
            ShareBasis::ActuatorForce,
        );
    }

    #[test]
    fn forced_solver_failures_keep_the_breakdown_aligned_driver_basis() {
        // Crank 0.01 + ground 0.065 > coupler 0.04 + rocker 0.03: about 117..243 deg unreachable.
        assert_forced_failures_stay_aligned(
            SampleMechanism::FourBar,
            [0.065, 0.0],
            false,
            FOURBAR_WEIGHTS,
            ShareBasis::DriverTorque,
        );
    }

    /// No actuator: the shares are driver-torque shares `-P_g / omega`, the
    /// total is the plotted driver torque, and with the constant driver rate
    /// no share is NaN.
    #[test]
    fn fourbar_without_actuator_breaks_down_the_driver_torque() {
        let state = swept(SampleMechanism::FourBar, false, FOURBAR_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(b.basis, ShareBasis::DriverTorque);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W1", "W2"]);
        assert_aligned(data, b);
        assert_gravity_only_sum_invariants(data, b);
        let torques = data.driver_torques.as_ref().expect("driver torques");
        let omega = state.driver_omega();
        let peak_power = max_abs_finite(&b.total_power);
        for (k, &torque) in torques.iter().enumerate() {
            assert_eq!(b.total_force[k], torque, "total = driver torque at {} deg", data.angles_deg[k]);
            assert_close("total_power = T * omega", data, k, b.total_power[k], torque * omega, ROUND_OFF * peak_power);
            for i in 0..b.sources.len() {
                let share = b.force_share[i][k];
                assert!(share.is_finite(), "{} torque share at {} deg", b.sources[i].id, data.angles_deg[k]);
                assert_eq!(share, -b.gravity_power[i][k] / omega);
            }
        }
    }

    /// Gravity comes from the mechanism's element, which carries the
    /// mounting angle: tilted 30 deg, the weights still explain the whole
    /// actuator load, and at 90 deg the rocker tip (moving in -x) is helped
    /// by the now horizontal gravity component instead of being neutral.
    #[test]
    fn breakdown_uses_the_mounting_angle_through_the_gravity_element() {
        let level = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let mut tilted = AppState::default();
        tilted.load_sample(SampleMechanism::ParallelogramActuator);
        set_actuator_stored_force(&mut tilted, 0.0);
        tilted.mounting_angle = 30.0_f64.to_radians();
        add_weights(&mut tilted, PARALLELOGRAM_WEIGHTS);
        tilted.compute_sweep();
        let g = gravity_vector(tilted.mechanism.as_ref().unwrap());
        assert!((g[0] + 9.81 * 0.5).abs() < 1e-9, "gravity tilted by the mounting angle: {g:?}");

        let (data, b) = breakdown(&tilted);
        assert_gravity_only_sum_invariants(data, b);
        let (level_data, level_b) = breakdown(&level);
        let tip = source_index(b, "W1");
        assert_eq!(level_b.classification(tip, sample_at(level_data, 90.0)), Neutral);
        assert_eq!(b.classification(tip, sample_at(data, 90.0)), Helping);
    }

    /// World position of `source` at every sweep sample, rebuilt only from
    /// positions the sweep recorded (no velocity solve): the trace of one
    /// attachment point of the source's body plus that body's angle.
    fn traced_positions(state: &AppState, data: &SweepData, source: &WeightSource) -> Vec<[f64; 2]> {
        let body = &state.blueprint.as_ref().expect("blueprint").bodies[&source.body_id];
        let (point, anchor) = body
            .attachment_points
            .iter()
            .min_by(|a, b| a.0.cmp(b.0))
            .unwrap_or_else(|| panic!("{} has no attachment point", source.body_id));
        let trace = &data.coupler_traces[&format!("{}.{point}", source.body_id)];
        let angles_deg = &data.body_angles[&source.body_id];
        let offset = [source.local_pos[0] - anchor[0], source.local_pos[1] - anchor[1]];
        trace
            .iter()
            .zip(angles_deg)
            .map(|(p, deg)| {
                let (sin, cos) = deg.to_radians().sin_cos();
                [p[0] + cos * offset[0] - sin * offset[1], p[1] + sin * offset[0] + cos * offset[1]]
            })
            .collect()
    }

    /// The sweep feeds each sample's own pose and velocity to the physics:
    /// on both actuator samples, every source (link self-weights and point
    /// masses) has gravity power equal to minus the rate of change of its
    /// potential energy between adjacent sweep samples, a central difference
    /// over +-1 deg of the positions the sweep recorded. On the
    /// parallelogram, samples within 1 deg of its change points (0/180/360
    /// deg, all links collinear) are skipped: the velocity there is not
    /// unique and Newton only reaches the pose to ~sqrt(tolerance). The
    /// Chebyshev crank-rocker has no change point.
    #[test]
    fn sweep_gravity_power_matches_energy_change_between_adjacent_samples() {
        // (sample, weights, change points to skip, tolerance relative to the
        // source's peak |P_g|).
        let cases: [(SampleMechanism, Weights, &[f64], f64); 2] = [
            (SampleMechanism::ParallelogramActuator, PARALLELOGRAM_WEIGHTS, &[0.0, 180.0, 360.0], PARALLELOGRAM_FD_TOL),
            (SampleMechanism::ChebyshevLambdaActuator, CHEBYSHEV_WEIGHTS, &[], CHEBYSHEV_FD_TOL),
        ];
        for (sample, weights, change_points, rel_tol) in cases {
            let state = swept(sample, true, weights);
            let (data, b) = breakdown(&state);
            assert_eq!(b.sources.iter().filter(|s| !s.is_link_self_weight).count(), weights.len());
            let g = gravity_vector(state.mechanism.as_ref().unwrap());
            let dt = 1.0_f64.to_radians() / state.driver_omega(); // one degree per sample
            let n = data.angles_deg.len();
            for (i, s) in b.sources.iter().enumerate() {
                let r = traced_positions(&state, data, s);
                let peak = max_abs_finite(&b.gravity_power[i]);
                assert!(peak > 0.0, "{sample:?} {}: never moves vertically", s.id);
                let mut checked = 0;
                for k in 1..n - 1 {
                    let deg = data.angles_deg[k];
                    if change_points.iter().any(|&c| (deg - c).abs() <= 1.0) {
                        continue;
                    }
                    let (before, after) = (r[k - 1], r[k + 1]);
                    let energy_fd = s.mass * (g[0] * (after[0] - before[0]) + g[1] * (after[1] - before[1])) / (2.0 * dt);
                    let what = format!("{sample:?} {}", s.id);
                    assert_close(&what, data, k, b.gravity_power[i][k], energy_fd, rel_tol * peak);
                    checked += 1;
                }
                assert!(checked > 300, "{sample:?} {}: only {checked} samples checked", s.id);
            }
        }
    }

    /// BL-024: a Mass edit on a link that carries weights re-derives the
    /// live composite body (base + point masses) without a rebuild, so the
    /// breakdown recomputed after the edit still has the weights explain
    /// the whole actuator load, with the new base mass in that link's
    /// self-weight.
    #[test]
    fn breakdown_sum_invariant_holds_after_a_mass_edit_on_a_weighted_link_without_rebuild() {
        let mut state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (_, before) = breakdown(&state);
        let old_mass = before.sources[source_index(before, "link:rocker")].mass;
        let new_mass = old_mass + 7.5;
        assert!(before.sources.iter().any(|s| s.id == "W1" && s.body_id == "rocker"), "the rocker carries W1");

        state.set_body_mass("rocker", new_mass);
        assert!(state.sweep_dirty, "the edit marks the sweep stale");
        // What the update loop does once the debounce expires; no rebuild.
        state.compute_sweep();

        let (data, b) = breakdown(&state);
        assert_eq!(b.sources[source_index(b, "link:rocker")].mass, new_mass);
        assert_eq!(b.sources[source_index(b, "W1")].mass, 50.0);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    #[test]
    fn sweep_without_weight_sources_has_no_breakdown() {
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let omega = 2.0 * std::f64::consts::PI;
        let (plain, _) = compute_sweep_data(&mech, &q0, omega, 0.0, 9.81, None);
        assert!(plain.weight_breakdown.is_none());
        let (empty, _) = compute_sweep_data_with_weights(&mech, &q0, omega, 0.0, 9.81, None, Vec::new());
        assert!(empty.weight_breakdown.is_none());
        assert_eq!(plain.angles_deg, empty.angles_deg);
    }

    fn source(id: &str) -> WeightSource {
        WeightSource {
            id: id.to_string(),
            name: id.to_string(),
            body_id: "bar".to_string(),
            local_pos: [0.0, 0.0],
            mass: 1.0,
            is_link_self_weight: false,
        }
    }

    /// `got` equals `want` sample by sample: NaN at the same samples, finite
    /// values within the absolute tolerance `tol`.
    fn assert_series(name: &str, got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{name} length");
        for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
            let same = (g.is_nan() && w.is_nan()) || (g - w).abs() <= tol;
            assert!(same, "{name}[{k}]: {g} vs {w} (tol {tol})");
        }
    }

    /// Hand-checkable rows: rates 2, 0.01, -1, then a failed sample. The
    /// sweep-wide max |rate| is 2, so 0.01 (< 1 % of it) is a near-reversal
    /// sample; the max |P_act| is 10, so -1e-9 is inside the braking
    /// tolerance.
    #[test]
    fn builder_applies_the_sweep_wide_rules() {
        let mut builder = WeightBreakdownBuilder::new(vec![source("A"), source("B")], ShareBasis::ActuatorForce, [0.0, -9.81], 4);
        builder.push_row(&[4.0, -2.0], 2.0, 5.0, 10.0);
        builder.push_row(&[0.03, 0.5], 0.01, 7.0, -3.0);
        builder.push_row(&[-2.0, 6.0], -1.0, 3.0, -1e-9);
        builder.push_nan();
        let b = builder.finish();
        let nan = f64::NAN;
        // Hand values of magnitude <= 12: exact up to round-off.
        const HAND: f64 = 1e-11;
        assert_eq!(b.basis, ShareBasis::ActuatorForce);
        assert_series("gravity_power[0]", &b.gravity_power[0], &[4.0, 0.03, -2.0, nan], HAND);
        assert_series("force_share[0]", &b.force_share[0], &[-2.0, nan, -2.0, nan], HAND);
        assert_series("force_share[1]", &b.force_share[1], &[1.0, nan, 6.0, nan], HAND);
        assert_series("power_share[0]", &b.power_share[0], &[-4.0, -0.03, 2.0, nan], HAND);
        assert_series("power_share[1]", &b.power_share[1], &[2.0, -0.5, -6.0, nan], HAND);
        assert_series("total_force", &b.total_force, &[5.0, 7.0, 3.0, nan], HAND);
        assert_series("total_power", &b.total_power, &[10.0, -3.0, -1e-9, nan], HAND);
        assert_series("other_force", &b.other_force, &[6.0, nan, -1.0, nan], HAND);
        assert_series("other_power", &b.other_power, &[12.0, -2.47, 4.0 - 1e-9, nan], HAND);
        assert_eq!(b.braking, vec![false, true, false, false]);
        // Source A: max |P_g| 4, neutral band 0.04. Source B: max 6, band 0.06.
        let classes = |i: usize| (0..4).map(|k| b.classification(i, k)).collect::<Vec<_>>();
        assert_eq!(classes(0), vec![Helping, Neutral, Hurting, Neutral]);
        assert_eq!(classes(1), vec![Hurting, Helping, Helping, Neutral]);
        assert_eq!(b.classification(2, 0), Neutral, "source index out of range");
        assert_eq!(b.classification(0, 4), Neutral, "sample index out of range");
    }

    #[test]
    fn required_totals_take_the_required_force_as_is_with_a_reversal_safe_power() {
        use ShareBasis::{ActuatorForce, DriverTorque};
        // Sizing: T = 10 N*m at omega = 2 rad/s over dL/dt = 4 m/s gives
        // F = 5 N and P = 20 W.
        let sizing = required_actuator_force(10.0, 2.0, 4.0, 0.0).unwrap();
        assert_eq!(required_totals(ActuatorForce, 10.0, 2.0, 4.0, sizing, 0.0), (5.0, 20.0));
        // Stored 3 N: statics already applied it, and since BL-026 the
        // sweep's required force includes it (10 * 2 / 4 + 3 = 8 N). The
        // total takes it unchanged, and P = F * dL/dt = 32 W.
        let stored = required_actuator_force(10.0, 2.0, 4.0, 3.0).unwrap();
        assert_eq!(stored, 8.0);
        assert_eq!(required_totals(ActuatorForce, 10.0, 2.0, 4.0, stored, 3.0), (8.0, 32.0));
        // Stroke reversal (dL/dt = 0): no required force, but the power is
        // still the driver's T * omega.
        assert_eq!(required_actuator_force(10.0, 2.0, 0.0, 3.0), None);
        let (force, power) = required_totals(ActuatorForce, 10.0, 2.0, 0.0, f64::NAN, 3.0);
        assert!(force.is_nan());
        assert_eq!(power, 20.0);
        // No actuator: the driver torque and its power.
        assert_eq!(required_totals(DriverTorque, 10.0, 2.0, 2.0, f64::NAN, 0.0), (10.0, 20.0));
        // Failed statics (NaN torque and force) stay NaN, never a fake zero.
        let (force, power) = required_totals(ActuatorForce, f64::NAN, 2.0, 4.0, f64::NAN, 3.0);
        assert!(force.is_nan() && power.is_nan());
    }
}
```

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
        assert_matches_sizing_mode(
            "trajectory actuator_forces",
            stored.actuator_forces.as_ref().unwrap(),
            sizing.actuator_forces.as_ref().unwrap(),
        );
    }
```

Add after it (after a blank line, inside `mod tests`, before its closing `}`; the test reuses that module's `build_linear_driver_mech` and `initial_distance` fixtures):

```rust
    // ── Payload weights in a stroke-mode sweep ────────────────────────────

    /// Linear driver, no actuator element: the weight shares are shares of
    /// the driver's axial force (N, `ShareBasis::DriverTorque` with the
    /// driver velocity as the rate) and the bar's self-weight explains all
    /// of it; samples past the bar's reach are NaN rows of the same length.
    #[test]
    fn stroke_mode_weight_breakdown_splits_the_linear_driver_force() {
        use crate::analysis::gravity_breakdown::weight_sources;
        use crate::forces::elements::GravityElement;

        let length_0 = initial_distance(); // ~0.5887 m; the bar reaches at most 0.6 m
        let velocity = 0.01;
        let (mut mech, q0) = build_linear_driver_mech(velocity, length_0);
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        let sources = weight_sources(&crate::io::mechanism_to_json(&mech).unwrap());
        assert_eq!(sources.len(), 1, "the bar's self-weight");

        let range = Some((length_0 - 0.02, length_0 + 0.02));
        let (data, _) = compute_sweep_data_with_weights(&mech, &q0, velocity, length_0, 9.81, range, sources);
        assert!(data.sweep_mode.is_stroke());
        let b = data.weight_breakdown.as_ref().expect("breakdown");
        assert_eq!(b.basis, ShareBasis::DriverTorque);
        let n = data.angles_deg.len();
        for len in [b.force_share[0].len(), b.power_share[0].len(), b.total_force.len(), b.other_force.len(), b.braking.len()] {
            assert_eq!(len, n);
        }
        let torques = data.driver_torques.as_ref().unwrap();
        let (mut solved, mut failed) = (0, 0);
        for (k, &torque) in torques.iter().enumerate() {
            if torque.is_nan() {
                failed += 1;
                assert!(b.force_share[0][k].is_nan() && b.total_force[k].is_nan() && b.other_force[k].is_nan());
                continue;
            }
            solved += 1;
            assert_eq!(b.total_force[k], torque, "total = driver force at {} m", data.angles_deg[k]);
            let scale = torque.abs().max(1e-12);
            assert!((b.force_share[0][k] + b.other_force[k] - b.total_force[k]).abs() <= 1e-12 * scale);
            assert!(b.other_force[k].abs() <= 1e-9 * scale, "self-weight is the only load at {} m", data.angles_deg[k]);
        }
        assert!(solved > 20 && failed > 0, "{solved} solved, {failed} past the bar's reach");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib sweep::
```

Expected: the lib test target does not compile. It reports 29 errors, all of them items that Step 3 adds: E0432 unresolved imports (`crate::gui::sweep::compute_sweep_data_with_weights`, `Classification`, `ShareBasis`), E0412 ``cannot find type `WeightBreakdown` in this scope`` (also `WeightSource`, `ShareBasis`), E0433 ``use of undeclared type `ShareBasis` `` (also `WeightBreakdownBuilder`), E0425 ``cannot find function `required_totals` in this scope`` (also `compute_sweep_data_with_weights`), E0422 for the `WeightSource` struct literal in the test helper, and E0609 ``no field `weight_breakdown` on type `sweep::SweepData` ``. The warnings printed alongside are pre-existing.

- [ ] **Step 3: Implement**

**3a. The breakdown module.** In `linkage-sim-rs/src/gui/sweep/weights.rs`, insert the implementation between the module doc comment (ending with the line ``//! live in `analysis::gravity_breakdown`.``) and `#[cfg(test)]`, with one blank line on each side:

```rust
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::analysis::gravity_breakdown::{self as gb, Classification, WeightSource};
use crate::core::mechanism::Mechanism;

/// What the force shares are shares of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShareBasis {
    /// The mechanism has a `LinearActuator`: shares of the actuator's
    /// required axial force (N, positive = extension push); the rate is the
    /// actuator extension rate dL/dt (m/s).
    ActuatorForce,
    /// No actuator: shares of the driver effort (N*m for a revolute
    /// driver, N for a linear driver); the rate is the driver rate.
    DriverTorque,
}

/// Per-weight gravity breakdown over one sweep.
///
/// Every per-sample series has `SweepData::angles_deg.len()` entries (NaN
/// where the sample has no pose or no velocity, `false` in `braking`); the
/// outer index of the per-source series follows `sources`. By construction
/// `sum(force_share) + other_force == total_force` and
/// `sum(power_share) + other_power == total_power` wherever the shares are
/// finite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBreakdown {
    /// The weights, in `gravity_breakdown::weight_sources` order.
    pub sources: Vec<WeightSource>,
    /// Actuator-force shares or driver-torque shares.
    pub basis: ShareBasis,
    /// `P_g,i = m_i g . v_i` (W), `[source][sample]`; positive = helping.
    pub gravity_power: Vec<Vec<f64>>,
    /// `-P_g,i / rate`: actuator force share (N), or driver torque share
    /// for `ShareBasis::DriverTorque`. NaN near stroke reversal.
    pub force_share: Vec<Vec<f64>>,
    /// `-P_g,i` (W): actuator (driver) power spent on each weight;
    /// negative = the weight gives power back.
    pub power_share: Vec<Vec<f64>>,
    /// `total_force - sum(force_share)`: non-gravity loads (force zones,
    /// springs, external loads, end stops).
    pub other_force: Vec<f64>,
    /// `total_power - sum(power_share)`.
    pub other_power: Vec<f64>,
    /// Required actuator force (N): the sweep's `actuator_forces`, which
    /// is the required force in sizing and stored-force mode alike
    /// (BL-026). Driver torque for `DriverTorque`.
    pub total_force: Vec<f64>,
    /// Required actuator (driver) power (W); negative = braking.
    pub total_power: Vec<f64>,
    /// True where the load drives the actuator (`gravity_breakdown::is_braking`).
    pub braking: Vec<bool>,
}

impl WeightBreakdown {
    /// Helping / hurting / neutral for `source` at `sample`, against that
    /// weight's sweep-wide neutral band (`gravity_breakdown::classify`).
    /// Neutral when either index is out of range.
    pub fn classification(&self, source: usize, sample: usize) -> Classification {
        let Some(series) = self.gravity_power.get(source) else {
            return Classification::Neutral;
        };
        let Some(&p_g) = series.get(sample) else {
            return Classification::Neutral;
        };
        gb::classify(p_g, gb::max_abs_finite(series))
    }
}

/// Required totals `(total_force, total_power)` at one sample.
///
/// `driver_torque` is the pass-1 statics driver effort (NaN when statics
/// failed) and `omega` the driver rate. For `ActuatorForce`, `rate` is the
/// actuator dL/dt, `actuator_force` the sweep's required actuator force
/// (`solver::reactions::required_actuator_force`, NaN where dL/dt is near
/// zero) and `stored_force` the stored force pass-1 statics applied (0 in
/// sizing mode). The total force is `actuator_force` unchanged: since
/// BL-026 it already includes the stored force, so adding it again would
/// double count it. The total power is the same balance multiplied through
/// by dL/dt, `F_required * rate = driver_torque * omega + stored_force *
/// rate`, which stays finite through stroke reversal where the force is
/// NaN.
pub(super) fn required_totals(
    basis: ShareBasis,
    driver_torque: f64,
    omega: f64,
    rate: f64,
    actuator_force: f64,
    stored_force: f64,
) -> (f64, f64) {
    match basis {
        ShareBasis::ActuatorForce => (actuator_force, driver_torque * omega + stored_force * rate),
        ShareBasis::DriverTorque => (driver_torque, driver_torque * omega),
    }
}

/// Collects one row per sweep sample and turns the rows into a
/// [`WeightBreakdown`] once the sweep-wide maxima are known.
pub(super) struct WeightBreakdownBuilder {
    sources: Vec<WeightSource>,
    basis: ShareBasis,
    g: [f64; 2],
    gravity_power: Vec<Vec<f64>>,
    rate: Vec<f64>,
    total_force: Vec<f64>,
    total_power: Vec<f64>,
}

impl WeightBreakdownBuilder {
    /// `g` is the mechanism's gravity vector (`gravity_breakdown::gravity_vector`).
    pub(super) fn new(sources: Vec<WeightSource>, basis: ShareBasis, g: [f64; 2], capacity: usize) -> Self {
        let gravity_power = sources.iter().map(|_| Vec::with_capacity(capacity)).collect();
        Self {
            sources,
            basis,
            g,
            gravity_power,
            rate: Vec::with_capacity(capacity),
            total_force: Vec::with_capacity(capacity),
            total_power: Vec::with_capacity(capacity),
        }
    }

    /// A sample with pose `q` and velocity `q_dot`: `rate` is the actuator
    /// dL/dt or the driver rate, totals from [`required_totals`].
    pub(super) fn push_sample(
        &mut self,
        mech: &Mechanism,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        rate: f64,
        total_force: f64,
        total_power: f64,
    ) {
        let powers = gb::gravity_powers(mech, &self.sources, q, q_dot, self.g);
        self.push_row(&powers, rate, total_force, total_power);
    }

    /// A sample with no pose or no velocity: NaN in every series.
    pub(super) fn push_nan(&mut self) {
        let nan = vec![f64::NAN; self.sources.len()];
        self.push_row(&nan, f64::NAN, f64::NAN, f64::NAN);
    }

    fn push_row(&mut self, powers: &[f64], rate: f64, total_force: f64, total_power: f64) {
        for (series, &p_g) in self.gravity_power.iter_mut().zip(powers) {
            series.push(p_g);
        }
        self.rate.push(rate);
        self.total_force.push(total_force);
        self.total_power.push(total_power);
    }

    /// Apply the sweep-wide rules: force shares (NaN near stroke reversal),
    /// power shares, the non-gravity remainder and braking.
    pub(super) fn finish(self) -> WeightBreakdown {
        let max_rate = gb::max_abs_finite(&self.rate);
        let force_share: Vec<Vec<f64>> = self
            .gravity_power
            .iter()
            .map(|series| {
                series
                    .iter()
                    .zip(&self.rate)
                    .map(|(&p_g, &rate)| gb::force_share(p_g, rate, max_rate))
                    .collect()
            })
            .collect();
        let power_share: Vec<Vec<f64>> = self
            .gravity_power
            .iter()
            .map(|series| series.iter().map(|&p_g| -p_g).collect())
            .collect();
        let other_force = remainder(&self.total_force, &force_share);
        let other_power = remainder(&self.total_power, &power_share);
        let max_power = gb::max_abs_finite(&self.total_power);
        let braking = self.total_power.iter().map(|&p| gb::is_braking(p, max_power)).collect();
        WeightBreakdown {
            sources: self.sources,
            basis: self.basis,
            gravity_power: self.gravity_power,
            force_share,
            power_share,
            other_force,
            other_power,
            total_force: self.total_force,
            total_power: self.total_power,
            braking,
        }
    }
}

/// `total[k] - sum_i shares[i][k]` for every sample `k`.
fn remainder(total: &[f64], shares: &[Vec<f64>]) -> Vec<f64> {
    total
        .iter()
        .enumerate()
        .map(|(k, &t)| t - shares.iter().map(|s| s[k]).sum::<f64>())
        .collect()
}
```

**3b. `linkage-sim-rs/src/gui/sweep/mod.rs`.**

Re-exports (`mod weights;` is already there from Step 1):

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
pub(crate) use motion_profile::apply_motion_profile;
```

Add after it:

```rust
pub use weights::{ShareBasis, WeightBreakdown};
use weights::{required_totals, WeightBreakdownBuilder};
```

Import:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
use crate::analysis::energy::compute_energy_state_mech;
```

Add after it:

```rust
use crate::analysis::gravity_breakdown::{gravity_vector, WeightSource};
```

The new `SweepData` field, between `pose_snapshots` and `toggle_angles`:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
    pub pose_snapshots: Option<Vec<Vec<[f64; 3]>>>,
    /// Angles (degrees) at which toggle/dead points were detected.
```

Replace with:

```rust
    pub pose_snapshots: Option<Vec<Vec<[f64; 3]>>>,
    /// Per-weight gravity breakdown (payload weights): which weight helps or
    /// hurts the actuator at each sample, force/power shares, and braking.
    /// `None` when the sweep had no weight sources (`compute_sweep_data`,
    /// or no link or point mass) and in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_breakdown: Option<WeightBreakdown>,
    /// Angles (degrees) at which toggle/dead points were detected.
```

`compute_sweep_data` becomes a wrapper; its body stays where it is and now belongs to `compute_sweep_data_with_weights`:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
    sweep_range: Option<(f64, f64)>,
) -> (SweepData, DVector<f64>) {
    // Detect whether this sweep should iterate angle (revolute driver)
```

Replace with:

```rust
    sweep_range: Option<(f64, f64)>,
) -> (SweepData, DVector<f64>) {
    compute_sweep_data_with_weights(mech, q_start, omega, theta_0, gravity_magnitude, sweep_range, Vec::new())
}

/// [`compute_sweep_data`] plus the per-weight gravity breakdown
/// (`SweepData::weight_breakdown`) of `weight_sources`, which must come from
/// `analysis::gravity_breakdown::weight_sources` on the blueprint `mech` was
/// built from. No sources = no breakdown.
pub(crate) fn compute_sweep_data_with_weights(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
    gravity_magnitude: f64,
    sweep_range: Option<(f64, f64)>,
    weight_sources: Vec<WeightSource>,
) -> (SweepData, DVector<f64>) {
    // Detect whether this sweep should iterate angle (revolute driver)
```

Builder setup (`capacity` and `actuator_element` are defined above it):

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
    let has_force_zones = !force_zones.is_empty();

    let mut data = SweepData {
```

Replace with:

```rust
    let has_force_zones = !force_zones.is_empty();

    // Per-weight gravity breakdown: shares of the actuator force when there
    // is an actuator, else of the driver torque. Gravity comes from the
    // mechanism's own element (mounting angle included), not from
    // `gravity_magnitude`.
    let share_basis = if actuator_element.is_some() {
        ShareBasis::ActuatorForce
    } else {
        ShareBasis::DriverTorque
    };
    // The stored force pass-1 statics applies (0 in sizing mode): the
    // breakdown's total power needs it to stay finite through stroke
    // reversal (`required_totals`).
    let stored_force = actuator_element.as_ref().map_or(0.0, |act| act.force);
    let mut breakdown = (!weight_sources.is_empty()).then(|| {
        WeightBreakdownBuilder::new(weight_sources, share_basis, gravity_vector(mech), capacity)
    });

    let mut data = SweepData {
```

The field in that function's own `SweepData` literal (filled after the loop):

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
        pose_snapshots: None,
        toggle_angles: Vec::new(),
        active_range: None, // computed after sweep loop
```

Replace with:

```rust
        pose_snapshots: None,
        weight_breakdown: None, // filled after the sweep loop
        toggle_angles: Vec::new(),
        active_range: None, // computed after sweep loop
```

The per-sample push, at the end of the `if let Ok(q_dot) = solve_velocity(mech, &q, t) {` block, after the block that pushes the actuator force, speed and power:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
                            data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
                        }
                    }
                } else {
                    data.kinetic_energy.push(f64::NAN);
```

Replace with:

```rust
                            data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
                        }
                    }

                    if let Some(builder) = breakdown.as_mut() {
                        // A failed statics solve pushed a 0 driver torque above;
                        // the breakdown reports NaN totals there, not a fake zero.
                        let statics_ok = pending_reactions.is_some();
                        let last = |series: &Option<Vec<f64>>| {
                            series.as_ref().and_then(|s| s.last().copied()).unwrap_or(f64::NAN)
                        };
                        let driver_torque = if statics_ok { last(&data.driver_torques) } else { f64::NAN };
                        let actuator_force = if statics_ok { last(&data.actuator_forces) } else { f64::NAN };
                        let rate = match share_basis {
                            ShareBasis::ActuatorForce => last(&data.actuator_speeds),
                            ShareBasis::DriverTorque => omega,
                        };
                        let (total_force, total_power) =
                            required_totals(share_basis, driver_torque, omega, rate, actuator_force, stored_force);
                        builder.push_sample(mech, &q, &q_dot, rate, total_force, total_power);
                    }
                } else {
                    data.kinetic_energy.push(f64::NAN);
```

The velocity-failure branch (the `else` above) pushes a NaN row:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
                        data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
                    }
                }

                // Push the final reactions from the helper (which handles
```

Replace with:

```rust
                        data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
                    }

                    if let Some(builder) = breakdown.as_mut() {
                        builder.push_nan();
                    }
                }

                // Push the final reactions from the helper (which handles
```

The position-failure call to `push_nan_row` passes the builder, and the breakdown is finished after the loop:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
                    has_force_zones,
                );
            }
        }
    }

    data.joint_reaction_magnitudes = reaction_data;
```

Replace with:

```rust
                    has_force_zones,
                    breakdown.as_mut(),
                );
            }
        }
    }

    data.joint_reaction_magnitudes = reaction_data;
    data.weight_breakdown = breakdown.map(WeightBreakdownBuilder::finish);
```

`push_nan_row` gets the new last parameter and pushes the NaN row:

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
    has_actuator: bool,
    has_force_zones: bool,
) {
    data.angles_deg.push(angle_deg);
```

Replace with:

```rust
    has_actuator: bool,
    has_force_zones: bool,
    breakdown: Option<&mut WeightBreakdownBuilder>,
) {
    data.angles_deg.push(angle_deg);
```

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
    if has_force_zones {
        data.output_forces.as_mut().unwrap().push(f64::NAN);
    }
}

/// Per-sample inverse-kinematics trajectory loop.
```

Replace with:

```rust
    if has_force_zones {
        data.output_forces.as_mut().unwrap().push(f64::NAN);
    }

    if let Some(builder) = breakdown {
        builder.push_nan();
    }
}

/// Per-sample inverse-kinematics trajectory loop.
```

`weight_breakdown: None,` in the other `SweepData` literals of this file: `empty_trajectory_sweep_data` and the two literals of the `apply_motion_profile` tests.

Find in `linkage-sim-rs/src/gui/sweep/mod.rs`:

```rust
        pose_snapshots: None,
        toggle_angles: Vec::new(),
        active_range: None,
        sweep_mode: mode,
```

Replace with:

```rust
        pose_snapshots: None,
        weight_breakdown: None,
        toggle_angles: Vec::new(),
        active_range: None,
        sweep_mode: mode,
```

Find in `linkage-sim-rs/src/gui/sweep/mod.rs` (2 identical occurrences in `mod tests`, replace both):

```rust
            pose_snapshots: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: SweepMode::Angle,
```

Replace with:

```rust
            pose_snapshots: None,
            weight_breakdown: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: SweepMode::Angle,
```

**3c. The other `SweepData` literals** (tests of the exporters).

Find in `linkage-sim-rs/src/gui/export/csv.rs` (2 identical occurrences in `mod tests`, replace both):

```rust
            pose_snapshots: None,
            toggle_angles: Vec::new(),
```

Replace with:

```rust
            pose_snapshots: None,
            weight_breakdown: None,
            toggle_angles: Vec::new(),
```

Find in `linkage-sim-rs/src/gui/export/raster.rs` (1 occurrence, in `mod tests`):

```rust
            pose_snapshots: None,
            toggle_angles: Vec::new(),
```

Replace with:

```rust
            pose_snapshots: None,
            weight_breakdown: None,
            toggle_angles: Vec::new(),
```

**3d. `linkage-sim-rs/src/gui/state/blueprint_ops.rs`.** `AppState::compute_sweep` passes the blueprint's weight sources.

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
use crate::analysis::force_breakdown::evaluate_contributions;
```

Add after it:

```rust
use crate::analysis::gravity_breakdown::weight_sources;
```

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
use crate::gui::sweep::{
    apply_motion_profile, compute_sweep_data, compute_trajectory, detect_fourbar_links,
    empty_trajectory_sweep_data, SweepMode,
};
```

Replace with:

```rust
use crate::gui::sweep::{
    apply_motion_profile, compute_sweep_data_with_weights, compute_trajectory,
    detect_fourbar_links, empty_trajectory_sweep_data, SweepMode,
};
```

Find in `linkage-sim-rs/src/gui/state/blueprint_ops.rs`:

```rust
        let (mut data, q_zero) = compute_sweep_data(mech, &q_start, omega, theta_0, self.gravity_magnitude, sweep_range);
```

Replace with:

```rust
        // Payload weights: the breakdown needs the blueprint's weight list
        // (link self-weights + point masses); the built mechanism only has
        // the composite masses.
        let sources = self.blueprint.as_ref().map(weight_sources).unwrap_or_default();
        let (mut data, q_zero) = compute_sweep_data_with_weights(
            mech,
            &q_start,
            omega,
            theta_0,
            self.gravity_magnitude,
            sweep_range,
            sources,
        );
```

**3e. Docs.**

`docs/ai/02-system.yaml`, `invariants_to_protect`: extend the "Within one sweep, ALL data channels" item and add the new invariant after it.

Find in `docs/ai/02-system.yaml`:

```yaml
    max-min+1. Plot consumers key off angles_deg, not a fixed count.
```

Add after it:

```yaml
    SweepData::weight_breakdown (per-weight gravity breakdown) obeys the
    same rule: every per-source and total series has angles_deg.len()
    entries; push_nan_row (and the velocity-failure branch) push a NaN row
    into its WeightBreakdownBuilder.
  - weight_breakdown_total_is_required_force — gui/sweep/weights.rs
    required_totals: for an actuator, total_force = the sweep's
    actuator_forces as is. Since BL-026 that is already the required force
    in sizing and stored-force mode alike
    (solver::reactions::required_actuator_force), so never add the stored
    force again (stored_force_mode_and_sizing_mode_give_identical_breakdowns
    fails if you do). total_power = driver_torque*omega + stored*dL/dt,
    the same balance times dL/dt, finite through stroke reversal where
    actuator_forces is NaN. Without an actuator the shares are
    driver-effort shares (N*m, or N for a linear driver) with rate =
    driver rate. Force shares are NaN where |dL/dt| < 1 % of its sweep max;
    power shares and classification stay defined. A failed statics solve
    gives NaN totals, not the 0 torque the plot shows.
```

`docs/ai/03-structure.yaml`, `gui_submodules.sweep`: the new file, and the test-support line.

Find in `docs/ai/03-structure.yaml`:

```yaml
      fourbar: sweep/fourbar.rs (4-bar linkage detection for transmission angle)
```

Add after it:

```yaml
      weights: sweep/weights.rs (payload weights - WeightBreakdown {sources, basis, gravity_power, force_share, power_share, other_force, other_power, total_force, total_power, braking} + classification(); ShareBasis {ActuatorForce, DriverTorque}; WeightBreakdownBuilder fed per sample by compute_sweep_data_with_weights; required_totals. AppState::compute_sweep passes gravity_breakdown::weight_sources(blueprint); plain compute_sweep_data = no breakdown)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids; extend it instead of re-implementing fixtures per module)
```

Replace with:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force; extend it instead of re-implementing fixtures per module)
```

`docs/ai/05-update-tracker.md`, new top entry (above the Task 3 entry):

Find in `docs/ai/05-update-tracker.md`:

```markdown
## 2026-09-29 — Payload weights Task 3: gravity breakdown physics module
```

Replace with:

```markdown
## 2026-09-29 — Payload weights Task 4: per-weight breakdown in the sweep
- `gui/sweep/weights.rs` (new): `WeightBreakdown` (per source `gravity_power`,
  `force_share` = `-P_g/rate`, `power_share` = `-P_g`; `other_force`,
  `other_power` = total - sum; `total_force`, `total_power`; `braking`;
  `classification(source, sample)`), `ShareBasis::{ActuatorForce,
  DriverTorque}`, `WeightBreakdownBuilder` and `required_totals`. Re-exported
  as `gui::sweep::{WeightBreakdown, ShareBasis}`.
- `gui/sweep/mod.rs`: `SweepData::weight_breakdown: Option<WeightBreakdown>`;
  `compute_sweep_data_with_weights(.., weight_sources)` feeds one row per
  sample (NaN rows through `push_nan_row` and the velocity-failure branch;
  NaN totals where statics failed instead of the 0 torque the plot shows);
  `compute_sweep_data` keeps its signature and computes no breakdown.
  `AppState::compute_sweep` passes `weight_sources(blueprint)`.
- Totals are the REQUIRED actuator force/power. BL-026 (on main) already
  makes `actuator_forces` the required force in stored-force mode, so
  `required_totals` takes it as is. The draft's stored-force add-back is
  gone. The total power `T*omega + F_stored*dL/dt` stays finite through
  stroke reversal (02-system.yaml `weight_breakdown_total_is_required_force`).
  No actuator: driver-torque shares (N for a linear driver in stroke mode).
  Trajectory mode: `None`.
- `gui/test_support.rs` gains `set_actuator_stored_force` (shared
  stored-force setter; later tasks reuse it instead of copying the loop).
- Tests: `gui::sweep::weights::tests` cover Parallelogram and Chebyshev
  sizing with two weights (sums, totals = plotted actuator force/power,
  remainder ~ 0). `stored_force_mode_and_sizing_mode_give_identical_breakdowns`
  checks totals, shares, remainders, braking and classification on both
  samples. Also covered: known-pose and trace-derived helping/hurting;
  braking follows net gravity power; near-reversal NaN; forced solver
  failures with actuator and driver basis; FourBar driver-torque basis;
  mounting angle; builder and `required_totals` hand values.
  `sweep_gravity_power_matches_energy_change_between_adjacent_samples` now
  checks every source on both actuator samples. Positions are rebuilt from
  the traces plus body angles; the measured central-difference error is
  O(h^2): 5e-5 on the Parallelogram, 1.2e-3 on the Chebyshev.
  `breakdown_sum_invariant_holds_after_a_mass_edit_on_a_weighted_link_without_rebuild`
  guards BL-024. Also `gui::sweep::tests::stroke_mode_weight_breakdown_splits_the_linear_driver_force`.
- Mutation checks done. Restoring the add-back fails the stored/sizing test
  and the `required_totals` test. Dropping `apply_point_masses` from
  `sync_live_mass_props` fails the mass-edit test (F_other = -78 N).

## 2026-09-29 — Payload weights Task 3: gravity breakdown physics module
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib sweep::weights
```

Expected: `test result: ok. 15 passed; 0 failed`.

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib stroke_mode_weight_breakdown
```

Expected: `test result: ok. 1 passed; 0 failed`.

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: `test result: ok. 812 passed; 0 failed` (796 before this task plus 16 new tests: 15 in `gui::sweep::weights::tests` and 1 in `gui::sweep::tests`).

Mutation checks (do them once, do not commit them, restore each edit afterwards):

- In `required_totals` (`weights.rs`), change the first element of the `ShareBasis::ActuatorForce` arm from `actuator_force` to `actuator_force + stored_force` (the stored-force add-back that BL-026 made wrong), then run `cargo test --lib sweep::weights`. Exactly two tests fail: `stored_force_mode_and_sizing_mode_give_identical_breakdowns` and `required_totals_take_the_required_force_as_is_with_a_reversal_safe_power`.
- In `sync_live_mass_props` (`blueprint_ops.rs`), delete the line `crate::io::from_json::apply_point_masses(&mut composite, &bp_body.point_masses);`, then run `cargo test --lib sweep::weights`. Among this task's tests, `breakdown_sum_invariant_holds_after_a_mass_edit_on_a_weighted_link_without_rebuild` fails with `F_other (gravity is the only load) at 0 deg: -77.857...` (about -78 N, the weight the live body lost).
- In `push_nan_row` (`sweep/mod.rs`), replace the `if let Some(builder) = breakdown { builder.push_nan(); }` block with `let _ = breakdown;`, then run `cargo test --lib sweep::`. Three tests fail: `forced_solver_failures_keep_the_breakdown_aligned_actuator_basis`, `forced_solver_failures_keep_the_breakdown_aligned_driver_basis` and `stroke_mode_weight_breakdown_splits_the_linear_driver_force`.

Then the gate, which also runs clippy and the WASM check:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Expected: `GATE PASS`. There are no new clippy warnings: the warning counts for `linkage-sim-rs (lib)` and `(lib test)` are unchanged from before this task (285 and 301 on the branch this plan was verified on). `push_nan_row` already carried a `too_many_arguments` warning and keeps that single warning with its tenth parameter (9/7 becomes 10/7). Two spots are written to avoid new warnings: the `type Weights = &'static [(&'static str, f64, [f64; 2])]` alias in the weights tests (`clippy::type_complexity`) and the `for (k, &torque) in torques.iter().enumerate()` loop in the stroke-mode test (`clippy::needless_range_loop`). The test run rewrites the PNGs in `docs/chebyshev_lambda/`; the `git checkout` puts them back so they stay out of the commit.

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/gui/sweep/weights.rs linkage-sim-rs/src/gui/sweep/mod.rs linkage-sim-rs/src/gui/state/blueprint_ops.rs linkage-sim-rs/src/gui/test_support.rs linkage-sim-rs/src/gui/export/csv.rs linkage-sim-rs/src/gui/export/raster.rs docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 4: per-weight gravity breakdown in the sweep

SweepData gains weight_breakdown. Per weight it holds the gravity power,
the force share -P_g/dL/dt (NaN within 1 % of stroke reversal), the power
share -P_g, the non-gravity remainder, the required totals, braking, and
classification(). compute_sweep_data_with_weights feeds one row per
sample, with NaN rows via push_nan_row. AppState::compute_sweep passes the
blueprint's weight sources. Without an actuator the shares are driver
effort shares.

BL-026 already makes actuator_forces the required force in stored-force
mode, so required_totals takes it as is and never adds the stored force
again. The total power T*omega + F_stored*dL/dt stays finite through
stroke reversal. stored_force_mode_and_sizing_mode_give_identical_
breakdowns compares every series, braking and classification on both
actuator samples; adding the stored force back fails it.

The adjacent-sample energy check covers both actuator samples and every
source (measured O(h^2) error, 1.2e-3 of peak on the Chebyshev). A new
test checks that a Mass edit on a weighted link keeps the sum invariant
without a rebuild (BL-024). Dropping apply_point_masses from
sync_live_mass_props fails it. The stored-force test setter lives in
gui/test_support.rs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 5: Weight hit testing and selection

A weight (point mass) becomes something the canvas can pick. Clicking a weight marker in Select mode selects it, Shift+click toggles it in the multi-selection, and hovering a weight shows a highlight ring and a grab cursor. Drawing and picking compute the marker position through one shared function, so a weight is picked exactly where it is drawn. This task adds selection only: dragging is Task 6, the editor UI is Task 8 and the readout is Task 9.

Design points:

- A weight often sits exactly on a pin, where the joint would otherwise always win the click. Select-mode clicks therefore test weights first, then joints, then pins. `WEIGHT_HIT_RADIUS` (8 px) stays below `HIT_RADIUS` (12 px), so a joint under a weight is still reachable from its outer ring.
- `find_point_mass_at` only considers weights the canvas draws and can address: weights on ground are not drawn (the loader skips them) and blank ids are not addressable. Ties on distance go to the first weight with bodies sorted by id and weights in list order, so the result never depends on `HashMap` order.
- `AppState::body_local_to_world` is the inverse of `world_to_body_local`. Both the marker position and the Move to Link handler need it, so the inline transform in Move to Link is replaced by the method (DRY).
- The canvas click tests build pointer events with a shared `test_support::primary_button_with` helper instead of a local one. The name leaves room for a 2-argument `primary_button` beside it later.

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/types.rs` (`SelectedEntity`)
- Modify: `linkage-sim-rs/src/gui/state/entity_crud.rs` (new method `body_local_to_world` after `world_to_body_local`, inside `impl AppState`)
- Modify: `linkage-sim-rs/src/gui/state/tests.rs` (new tests after `world_to_body_local_identity_pose`)
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (import; new `primary_button_with`)
- Modify: `linkage-sim-rs/src/gui/canvas/colors.rs` (Sizing section, after `HIT_RADIUS`)
- Modify: `linkage-sim-rs/src/gui/canvas/hit_testing.rs` (imports; two new functions after `find_nearest_body_segment`; new `#[cfg(test)] mod tests` at the end)
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs` (import; Move to Link handler; hover call before the click-selection block; new "Weights" section before `// ── Draw Link tool`; the Select branch of `handle_click_selection`)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs` (import; point-mass block in `render_mechanism`)
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (`tests::weight_clicks` helpers and new tests)
- Modify: `docs/FEATURES.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`, `docs/architecture/ARCHITECTURE.md`

**Interfaces:**
- Consumes (Tasks 1-2 and the existing code base):
  - `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>`
  - `AppState::find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson>`
  - `AppState::remove_point_mass_by_id(&mut self, body_id: &str, weight_id: &str) -> bool`
  - `AppState::move_point_mass(&mut self, body_id: &str, weight_id: &str, target_body: &str, local_pos: [f64; 2]) -> bool`
  - `AppState::world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2]`
  - `crate::io::schema::is_blank_point_mass_id(id: &str) -> bool` (`pub(crate)`)
  - `crate::io::PointMassJson { id: String, label: Option<String>, mass: f64, local_pos: [f64; 2] }`
  - `crate::gui::test_support::sorted_link_ids(state: &AppState) -> Vec<String>`
- Produces:
  - `SelectedEntity::Weight { body_id: String, weight_id: String }` (`gui/state/types.rs`)
  - `pub fn find_point_mass_at(state: &AppState, screen_pos: Pos2, radius_px: f32) -> Option<(String, String)>` (`gui/canvas/hit_testing.rs`)
  - `pub fn point_mass_screen_pos(state: &AppState, body_id: &str, local_pos: [f64; 2]) -> Option<Pos2>` (`gui/canvas/hit_testing.rs`; drawing and picking share it)
  - `pub fn AppState::body_local_to_world(&self, body_id: &str, local: [f64; 2]) -> [f64; 2]` (`gui/state/entity_crud.rs`)
  - `pub const WEIGHT_RADIUS: f32 = 5.0; pub const WEIGHT_HIT_RADIUS: f32 = 8.0;` (`gui/canvas/colors.rs`)
  - `pub(crate) fn primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event` (`gui/test_support.rs`, `#[cfg(test)]`)
  - private in `gui/canvas/interaction.rs`: `fn weights_interactive(state: &AppState) -> bool`, `fn draw_weight_hover(ui: &egui::Ui, painter: &egui::Painter, response: &egui::Response, state: &AppState)`

Every edit below is anchored by the quoted text; the tests are added first (Step 1) and the implementation second (Step 3), so an anchor in Step 3 may be followed by the test code from Step 1.

- [ ] **Step 1: Write the failing tests**

(a) Test helper. In `linkage-sim-rs/src/gui/test_support.rs`, add the egui import.

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
//! Helpers shared by the tests of the GUI modules.

use crate::core::state::GROUND_ID;
```

Replace with:

```rust
//! Helpers shared by the tests of the GUI modules.

use eframe::egui;

use crate::core::state::GROUND_ID;
```

Then add the pointer-event helper at the end of the file.

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
            la.force = force;
        }
    }
    state.rebuild();
}
```

Add after it:

```rust

/// A primary-button press (`pressed`) or release at `pos`, with `modifiers`
/// held (e.g. Shift for a multi-selection click).
pub(crate) fn primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers }
}
```

(b) `body_local_to_world` tests. In `linkage-sim-rs/src/gui/state/tests.rs`, directly after the existing `world_to_body_local_identity_pose` test (before `// ── Raw helper tests`):

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
    // ── world_to_body_local tests ─────────────────────────────────────────

    #[test]
    fn world_to_body_local_identity_pose() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let [lx, ly] = state.world_to_body_local("ground", 0.05, 0.03);
        assert!((lx - 0.05).abs() < 1e-10);
        assert!((ly - 0.03).abs() < 1e-10);
    }
```

Add after it:

```rust

    // ── body_local_to_world tests ─────────────────────────────────────────

    #[test]
    fn body_local_to_world_matches_the_mechanism_pose() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.solve_at_angle(0.7);
        let mech = state.mechanism.as_ref().unwrap();
        for body in ["crank", "coupler", "rocker"] {
            let want = mech.state().body_point_global(body, &nalgebra::Vector2::new(0.03, 0.02), &state.q);
            assert_eq!(state.body_local_to_world(body, [0.03, 0.02]), [want.x, want.y], "{body}");
        }
    }

    #[test]
    fn body_local_to_world_inverts_world_to_body_local() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.solve_at_angle(0.7);
        for body in ["crank", "coupler", "rocker"] {
            let [wx, wy] = state.body_local_to_world(body, [0.012, -0.004]);
            let [lx, ly] = state.world_to_body_local(body, wx, wy);
            assert!((lx - 0.012).abs() < 1e-12 && (ly + 0.004).abs() < 1e-12, "{body}: {lx}, {ly}");
        }
    }

    #[test]
    fn body_local_to_world_passes_ground_unknown_bodies_and_no_mechanism_through() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.body_local_to_world("ground", [0.05, 0.03]), [0.05, 0.03]);
        assert_eq!(state.body_local_to_world("no_such_body", [0.05, 0.03]), [0.05, 0.03]);
        state.mechanism = None;
        assert_eq!(state.body_local_to_world("crank", [0.05, 0.03]), [0.05, 0.03]);
    }
```

(c) Hit-testing tests. At the end of `linkage-sim-rs/src/gui/canvas/hit_testing.rs`, after `find_nearest_body_segment`:

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
    best.map(|(_, screen_pos, world_pos, body_id, point_a_name, point_b_name)| SegmentHit {
        body_id, world_pos, screen_pos, point_a_name, point_b_name,
    })
}
```

Add after it:

```rust

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::vec2;

    use crate::gui::samples::SampleMechanism;
    use crate::io::PointMassJson;

    /// Four-bar with weight W1 (2 kg) on the coupler, in the default view
    /// (5000 px/m, so 1 mm is 5 px).
    fn fourbar_with_w1() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.add_point_mass("coupler", 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        state
    }

    fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
        let pm = state.find_point_mass(body, id).expect("weight exists");
        point_mass_screen_pos(state, body, pm.local_pos).expect("finite position")
    }

    fn hit(body: &str, id: &str) -> Option<(String, String)> {
        Some((body.to_string(), id.to_string()))
    }

    #[test]
    fn point_mass_screen_pos_follows_the_body_pose_and_the_view() {
        let state = fourbar_with_w1();
        let [wx, wy] = state.body_local_to_world("coupler", [0.03, 0.02]);
        let [sx, sy] = state.view.world_to_screen(wx, wy);
        assert_eq!(point_mass_screen_pos(&state, "coupler", [0.03, 0.02]), Some(Pos2::new(sx, sy)));
    }

    #[test]
    fn point_mass_screen_pos_is_none_for_a_non_finite_position() {
        let state = fourbar_with_w1();
        assert_eq!(point_mass_screen_pos(&state, "coupler", [f64::NAN, 0.0]), None);
        assert_eq!(point_mass_screen_pos(&state, "coupler", [0.0, f64::INFINITY]), None);
    }

    #[test]
    fn find_point_mass_at_hits_the_weight_under_the_cursor() {
        let state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        assert_eq!(find_point_mass_at(&state, at, 8.0), hit("coupler", "W1"));
        assert_eq!(find_point_mass_at(&state, at + vec2(3.0, 4.0), 8.0), hit("coupler", "W1"), "5 px off");
        assert_eq!(find_point_mass_at(&state, at + vec2(-7.5, 0.0), 8.0), hit("coupler", "W1"), "7.5 px off");
    }

    #[test]
    fn find_point_mass_at_misses_empty_space() {
        let state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        assert_eq!(find_point_mass_at(&state, at + vec2(8.5, 0.0), 8.0), None, "just outside the radius");
        assert_eq!(find_point_mass_at(&state, at + vec2(0.0, 200.0), 8.0), None, "far away");
        assert_eq!(find_point_mass_at(&state, at, 0.0), hit("coupler", "W1"), "a zero radius still hits the centre");
    }

    #[test]
    fn find_point_mass_at_prefers_the_nearest_weight() {
        let mut state = fourbar_with_w1();
        // 2 mm along the coupler from W1: 10 px at 5000 px/m.
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.032, 0.02]).as_deref(), Some("W2"));
        let w1 = weight_screen(&state, "coupler", "W1");
        let w2 = weight_screen(&state, "coupler", "W2");
        assert!((w1.distance(w2) - 10.0).abs() < 0.01, "fixture spacing {}", w1.distance(w2));
        assert_eq!(find_point_mass_at(&state, w2, 12.0), hit("coupler", "W2"));
        assert_eq!(find_point_mass_at(&state, w1, 12.0), hit("coupler", "W1"));
        assert_eq!(find_point_mass_at(&state, w1 + (w2 - w1) * 0.7, 12.0), hit("coupler", "W2"));
    }

    #[test]
    fn find_point_mass_at_breaks_ties_by_sorted_body_then_list_order() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Added rocker first, so insertion order cannot explain the result.
        assert_eq!(state.add_point_mass("rocker", 1.0, [0.01, 0.005]).as_deref(), Some("W1"));
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.01, 0.005]).as_deref(), Some("W2"));
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.01, 0.005]).as_deref(), Some("W3"));
        // Pose both links at the world origin so the three weights overlap exactly.
        let starts: Vec<usize> = ["coupler", "rocker"]
            .iter()
            .map(|b| state.mechanism.as_ref().unwrap().state().get_index(b).unwrap().q_start)
            .collect();
        for q0 in starts {
            for k in 0..3 {
                state.q[q0 + k] = 0.0;
            }
        }
        let at = weight_screen(&state, "rocker", "W1");
        assert_eq!(weight_screen(&state, "coupler", "W2"), at, "fixture: exact overlap");
        assert_eq!(find_point_mass_at(&state, at, 8.0), hit("coupler", "W2"));
    }

    #[test]
    fn find_point_mass_at_skips_ground_blank_ids_and_non_finite_positions() {
        let mut state = fourbar_with_w1();
        let local = state.find_point_mass("coupler", "W1").unwrap().local_pos;
        let at = weight_screen(&state, "coupler", "W1");
        let world = state.body_local_to_world("coupler", local);
        assert!(state.remove_point_mass_by_id("coupler", "W1"));
        // Weights the canvas does not draw or cannot address, all at W1's old spot.
        let weight = |id: &str, local_pos: [f64; 2]| PointMassJson { id: id.to_string(), label: None, mass: 1.0, local_pos };
        let bp = state.blueprint.as_mut().unwrap();
        bp.bodies.get_mut("ground").unwrap().point_masses.push(weight("G1", world));
        let coupler = bp.bodies.get_mut("coupler").unwrap();
        coupler.point_masses.push(weight("", local));
        coupler.point_masses.push(weight("  ", local));
        coupler.point_masses.push(weight("W9", [f64::NAN, local[1]]));
        assert_eq!(find_point_mass_at(&state, at, 50.0), None);
    }

    #[test]
    fn find_point_mass_at_without_a_blueprint_is_none() {
        let mut state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        state.blueprint = None;
        assert_eq!(find_point_mass_at(&state, at, 8.0), None);
    }
}
```

(d) Canvas click tests, inside the `weight_clicks` module of `mod tests` in `linkage-sim-rs/src/gui/canvas/mod.rs`. First replace the imports and the `frame` / `click` helpers with versions that take modifiers, return egui's output (for the cursor icon) and build pointer events with the shared helper.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::{AppState, EditorTool};
        use crate::gui::test_support::sorted_link_ids;
        use super::super::draw_canvas;

        fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw_canvas(ui, state));
            });
        }

        fn click(ctx: &egui::Context, state: &mut AppState, pos: Pos2) {
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            frame(ctx, state, vec![egui::Event::PointerMoved(pos)]);
            frame(ctx, state, vec![button(true)]);
            frame(ctx, state, vec![button(false)]);
        }
```

Replace with:

```rust
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::{AppState, EditorTool, SelectedEntity};
        use crate::gui::test_support::{primary_button_with, sorted_link_ids};
        use super::super::draw_canvas;

        /// One canvas frame with `events`, `modifiers` held; returns egui's
        /// output (cursor icon etc.).
        fn frame_with(
            ctx: &egui::Context,
            state: &mut AppState,
            events: Vec<egui::Event>,
            modifiers: egui::Modifiers,
        ) -> egui::FullOutput {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                events,
                modifiers,
                ..Default::default()
            };
            ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw_canvas(ui, state));
            })
        }

        fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) {
            let _ = frame_with(ctx, state, events, egui::Modifiers::NONE);
        }

        fn click_with(ctx: &egui::Context, state: &mut AppState, pos: Pos2, modifiers: egui::Modifiers) {
            let _ = frame_with(ctx, state, vec![egui::Event::PointerMoved(pos)], modifiers);
            let _ = frame_with(ctx, state, vec![primary_button_with(pos, true, modifiers)], modifiers);
            let _ = frame_with(ctx, state, vec![primary_button_with(pos, false, modifiers)], modifiers);
        }

        fn click(ctx: &egui::Context, state: &mut AppState, pos: Pos2) {
            click_with(ctx, state, pos, egui::Modifiers::NONE);
        }
```

Next, add two fixtures directly after the existing `local_under` helper.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        /// The body-local point under screen position `pos` on `body`.
        fn local_under(state: &AppState, body: &str, pos: Pos2) -> [f64; 2] {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            state.world_to_body_local(body, wx, wy)
        }
```

Add after it:

```rust

        /// Screen position of weight `id` on `body` at the current pose.
        fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
            let pm = state.find_point_mass(body, id).expect("weight exists");
            screen_of(state, world_of(state, body, pm.local_pos))
        }

        fn weight(body: &str, id: &str) -> SelectedEntity {
            SelectedEntity::Weight { body_id: body.to_string(), weight_id: id.to_string() }
        }
```

Finally add the five click tests after `place_mass_click_uses_the_last_point_mass`, the last test in `weight_clicks`.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            assert_eq!(pm.mass, 3.5);
            assert_close("placement", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
        }
```

Add after it:

```rust

        #[test]
        fn clicking_a_weight_selects_it() {
            let (ctx, mut state, body, _) = setup();
            state.selected = Some(SelectedEntity::Body(body.clone()));
            let at = weight_screen(&state, &body, "W1");
            let depth = state.undo_history.undo_count();

            click(&ctx, &mut state, at);

            assert_eq!(state.selected, Some(weight(&body, "W1")));
            assert!(state.multi_selected.is_empty());
            assert_eq!(state.undo_history.undo_count(), depth, "selecting is not an edit");
        }

        #[test]
        fn clicking_empty_canvas_clears_a_weight_selection() {
            let (ctx, mut state, body, _) = setup();
            state.selected = Some(weight(&body, "W1"));

            click(&ctx, &mut state, Pos2::new(120.0, 700.0));

            assert_eq!(state.selected, None);
        }

        #[test]
        fn a_weight_on_a_pin_wins_the_click_over_the_joint() {
            let (ctx, mut state, body, _) = setup();
            // A weight exactly on one of the link's pins, where a joint is drawn too.
            let pin = {
                let mech = state.mechanism.as_ref().unwrap();
                let mut names: Vec<&String> = mech.bodies()[&body].attachment_points.keys().collect();
                names.sort();
                mech.bodies()[&body].attachment_points[names[0]]
            };
            let id = state.add_point_mass(&body, 1.0, [pin.x, pin.y]).expect("weight on the pin");
            let at = weight_screen(&state, &body, &id);

            click(&ctx, &mut state, at);

            assert_eq!(state.selected, Some(weight(&body, &id)));
        }

        #[test]
        fn shift_click_toggles_a_weight_in_the_multi_selection() {
            let (ctx, mut state, body, _) = setup();
            let at = weight_screen(&state, &body, "W1");

            click_with(&ctx, &mut state, at, egui::Modifiers::SHIFT);
            assert_eq!(state.multi_selected, vec![weight(&body, "W1")]);
            assert_eq!(state.selected, Some(weight(&body, "W1")));

            click_with(&ctx, &mut state, at, egui::Modifiers::SHIFT);
            assert!(state.multi_selected.is_empty());
        }

        #[test]
        fn hovering_a_weight_shows_a_grab_cursor_only_in_select_mode() {
            let (ctx, mut state, body, _) = setup();
            let at = weight_screen(&state, &body, "W1");
            let hover = |state: &mut AppState, pos: Pos2| {
                frame_with(&ctx, state, vec![egui::Event::PointerMoved(pos)], egui::Modifiers::NONE)
                    .platform_output
                    .cursor_icon
            };

            assert_eq!(hover(&mut state, at), egui::CursorIcon::Grab);
            assert_eq!(hover(&mut state, Pos2::new(120.0, 700.0)), egui::CursorIcon::Default);

            state.active_tool = EditorTool::PlaceMass;
            assert_eq!(hover(&mut state, at), egui::CursorIcon::Default, "no weight pick in other tools");
        }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib weight
```

Expected: the lib test target does not compile (`error: could not compile linkage-sim-rs (lib test) due to 26 previous errors`), and no test runs. The errors are:
- `error[E0425]: cannot find function find_point_mass_at in this scope` (12 times) and `error[E0425]: cannot find function point_mass_screen_pos in this scope` (4 times), all in `src/gui/canvas/hit_testing.rs` (the new test module).
- `error[E0412]: cannot find type AppState in this scope` (2 times) and `error[E0433]: failed to resolve: use of undeclared type AppState` (2 times), in `hit_testing.rs`. The tests reach `AppState` through `use super::*;`, and `hit_testing.rs` does not import it yet.
- `error[E0599]: no method named body_local_to_world found for struct gui::state::AppState in the current scope` (5 times), all in `src/gui/state/tests.rs`.
- `error[E0599]: no variant named Weight found for enum types::SelectedEntity`, in `src/gui/canvas/mod.rs` (the `weight` fixture).

The new `primary_button_with` helper compiles, so it does not appear in the errors.

- [ ] **Step 3: Implement**

(a) The selection variant. Every existing `match` on `SelectedEntity` ends in a `_` or `other` arm, so no other code needs a new arm to compile.

Find in `linkage-sim-rs/src/gui/state/types.rs`:

```rust
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}
```

Replace with:

```rust
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
    /// A weight (point mass), addressed by its owning body and its id
    /// (`PointMassJson::id`), never by list index.
    Weight { body_id: String, weight_id: String },
}
```

(b) Local-to-world conversion, the inverse of `world_to_body_local`. Add it inside `impl AppState`, directly after `world_to_body_local` and before the closing brace of the block.

Find in `linkage-sim-rs/src/gui/state/entity_crud.rs`:

```rust
        [cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy]
    }
```

Add after it:

```rust

    /// Convert body-local coordinates to world coordinates using the body's
    /// current pose from `self.q`: the inverse of `world_to_body_local`.
    ///
    /// Returns `local` unchanged for the ground body (its frame is the world
    /// frame), for a body the built mechanism does not have, or when the
    /// mechanism is not built.
    pub fn body_local_to_world(&self, body_id: &str, local: [f64; 2]) -> [f64; 2] {
        let Some(mech) = &self.mechanism else { return local };
        let mech_state = mech.state();
        if body_id != GROUND_ID && mech_state.get_index(body_id).is_err() {
            return local;
        }
        let p = mech_state.body_point_global(body_id, &nalgebra::Vector2::new(local[0], local[1]), &self.q);
        [p.x, p.y]
    }
```

(c) Sizes. The pick radius is deliberately smaller than the joint pick radius.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const HIT_RADIUS: f32 = 12.0;
```

Add after it:

```rust
/// Radius of a weight (point mass) marker, in screen pixels.
pub const WEIGHT_RADIUS: f32 = 5.0;
/// Pick radius around a weight marker's centre, in screen pixels. Weights
/// are picked before joints and pins (see `handle_click_selection`), so this
/// stays below `HIT_RADIUS`: a joint under a weight is still reachable from
/// its outer ring.
pub const WEIGHT_HIT_RADIUS: f32 = 8.0;
```

(d) Hit testing. Add the imports:

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
use eframe::egui::Pos2;
```

Replace with:

```rust
use eframe::egui::Pos2;

use crate::core::state::GROUND_ID;
use crate::gui::state::AppState;
use crate::io::schema::is_blank_point_mass_id;
```

Then add the two functions after `find_nearest_body_segment`, above the `#[cfg(test)] mod tests` block from Step 1. Use `is_none_or`, not `map_or(true, ..)`: on the repo toolchain (rustc 1.89) clippy's `unnecessary_map_or` flags the `map_or` form.

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
    best.map(|(_, screen_pos, world_pos, body_id, point_a_name, point_b_name)| SegmentHit {
        body_id, world_pos, screen_pos, point_a_name, point_b_name,
    })
}
```

Add after it:

```rust

/// Screen position of the weight (point mass) at body-local `local_pos` on
/// `body_id`, at the current pose and view. `None` when it is not finite (a
/// non-finite position, which the loader skips). Drawing and hit testing both
/// use this, so a weight is picked exactly where it is drawn.
pub fn point_mass_screen_pos(state: &AppState, body_id: &str, local_pos: [f64; 2]) -> Option<Pos2> {
    let [wx, wy] = state.body_local_to_world(body_id, local_pos);
    let [sx, sy] = state.view.world_to_screen(wx, wy);
    (sx.is_finite() && sy.is_finite()).then(|| Pos2::new(sx, sy))
}

/// The weight under `screen_pos`: `(body_id, weight_id)` of the weight whose
/// marker centre is nearest, within `radius_px` (inclusive).
///
/// Only weights the canvas draws and can address are candidates: weights on
/// ground are not drawn (the loader skips them) and blank ids are not
/// addressable. On a tie the first weight wins, with bodies sorted by id and
/// weights in list order, so the result never depends on HashMap order.
pub fn find_point_mass_at(state: &AppState, screen_pos: Pos2, radius_px: f32) -> Option<(String, String)> {
    let bp = state.blueprint.as_ref()?;
    let mut body_ids: Vec<&String> = bp.bodies.keys().filter(|id| id.as_str() != GROUND_ID).collect();
    body_ids.sort();

    let mut best: Option<(f32, &str, &str)> = None;
    for body_id in body_ids {
        for pm in &bp.bodies[body_id].point_masses {
            if is_blank_point_mass_id(&pm.id) {
                continue;
            }
            let Some(center) = point_mass_screen_pos(state, body_id, pm.local_pos) else { continue };
            let dist = screen_pos.distance(center);
            if dist <= radius_px && best.is_none_or(|(d, _, _)| dist < d) {
                best = Some((dist, body_id, &pm.id));
            }
        }
    }
    best.map(|(_, body_id, weight_id)| (body_id.to_string(), weight_id.to_string()))
}
```

(e) Interaction. Import the two new functions.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
use super::hit_testing::{find_nearest_body_segment, AttachmentHit, BodySegment};
```

Replace with:

```rust
use super::hit_testing::{
    find_nearest_body_segment, find_point_mass_at, point_mass_screen_pos, AttachmentHit, BodySegment,
};
```

In the "Reassign point mass to a different link" (Move to Link) handler, replace the inline local-to-world transform with the new method. This is everything from `// Get the world position of the existing point mass` down to the line before `// Move from old body to new body in one undoable step`.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
                // Get the world position of the existing point mass
                if let Some([lx, ly]) = state.find_point_mass(&old_body_id, &weight_id).map(|pm| pm.local_pos) {
                    let [wx, wy] = {
                        // Convert old body-local to world
                        if old_body_id == "ground" {
                            [lx, ly]
                        } else if let Some(mech) = &state.mechanism {
                            if let Ok(idx) = mech.state().get_index(&old_body_id) {
                                let bx = state.q[idx.q_start];
                                let by = state.q[idx.q_start + 1];
                                let theta = state.q[idx.q_start + 2];
                                let ct = theta.cos();
                                let st = theta.sin();
                                [bx + ct * lx - st * ly, by + st * lx + ct * ly]
                            } else { [lx, ly] }
                        } else { [lx, ly] }
                    };
```

Replace with:

```rust
                // Get the world position of the existing point mass
                if let Some(local) = state.find_point_mass(&old_body_id, &weight_id).map(|pm| pm.local_pos) {
                    let [wx, wy] = state.body_local_to_world(&old_body_id, local);
```

Call the hover feedback directly before the click-selection block, after the Create Joint block.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    // ── Interaction: Create Joint two-click flow ────────────────────────
    if state.creating_joint.is_some() && response.clicked() {
        handle_create_joint(response, state, attachment_hit_targets);
    }
```

Add after it:

```rust

    // ── Weights: hover feedback ─────────────────────────────────────────
    draw_weight_hover(ui, painter, response, state);
```

Add the new "Weights" section after `handle_interaction` ends, directly before `// ── Draw Link tool`. `weights_interactive` is the gate for hover feedback (Task 6 reuses it for dragging): plain Select mode with no other canvas pick armed.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    right_drag_ended
}
```

Add after it:

```rust

// ── Weights (point masses) ───────────────────────────────────────────────────

/// Whether weights answer the pointer (hover feedback, drag): only in plain
/// Select mode, while no other canvas pick (trajectory target, Move to Link,
/// Reposition, Add Joint Point, joint or link creation) waits for a click.
fn weights_interactive(state: &AppState) -> bool {
    state.active_tool == EditorTool::Select
        && state.pending_canvas_pick.is_none()
        && state.reassigning_point_mass.is_none()
        && state.repositioning_point_mass.is_none()
        && state.adding_joint_point.is_none()
        && state.creating_joint.is_none()
        && state.draw_link_start.is_none()
}

/// Hover feedback: a ring around the weight under the pointer and a grab
/// cursor, so the user sees what a click selects.
fn draw_weight_hover(
    ui: &egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &AppState,
) {
    if !weights_interactive(state) || !response.hovered() {
        return;
    }
    let Some(pos) = response.hover_pos() else { return };
    let Some((body_id, weight_id)) = find_point_mass_at(state, pos, WEIGHT_HIT_RADIUS) else { return };
    let Some(pm) = state.find_point_mass(&body_id, &weight_id) else { return };
    let Some(center) = point_mass_screen_pos(state, &body_id, pm.local_pos) else { return };
    painter.circle_stroke(center, WEIGHT_HIT_RADIUS, Stroke::new(2.0, state.nc(JOINT_HOVER_HIGHLIGHT)));
    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
}
```

In `handle_click_selection`, `EditorTool::Select` arm, test weights first and only fall through to the joint loop when nothing was hit. The `hit.is_none()` check around the pin loop and the `if is_shift` / `else` block that follows are unchanged; that block already handles any `SelectedEntity`, so Shift+click toggles weights.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
                if !dxf_handled {
                    let mut hit: Option<SelectedEntity> = None;

                    for (joint_screen, joint_id) in joint_hit_targets {
                        if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                            hit = Some(SelectedEntity::Joint(joint_id.clone()));
                            break;
                        }
                    }
```

Replace with:

```rust
                if !dxf_handled {
                    // Weights first: they are small, user-placed targets that
                    // often sit on a pin, where the joint would otherwise
                    // always win.
                    let mut hit: Option<SelectedEntity> =
                        find_point_mass_at(state, pointer_pos, WEIGHT_HIT_RADIUS)
                            .map(|(body_id, weight_id)| SelectedEntity::Weight { body_id, weight_id });

                    if hit.is_none() {
                        for (joint_screen, joint_id) in joint_hit_targets {
                            if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                                hit = Some(SelectedEntity::Joint(joint_id.clone()));
                                break;
                            }
                        }
                    }
```

(f) Rendering. Draw weights through the shared position function and ring the selected ones (the primary selection or any member of the multi-selection).

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
use super::hit_testing::{AttachmentHit, BodySegment};
```

Replace with:

```rust
use super::hit_testing::{point_mass_screen_pos, AttachmentHit, BodySegment};
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
        let point_mass_color = gc(Color32::from_rgb(255, 200, 50));
        for (body_id, bp_body) in &bp.bodies {
```

Replace with:

```rust
        let point_mass_color = gc(Color32::from_rgb(255, 200, 50));
        let selected_ring = Stroke::new(2.0, gc(BODY_SELECTED_COLOR));
        for (body_id, bp_body) in &bp.bodies {
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
            for pm in &bp_body.point_masses {
                let local = nalgebra::Vector2::new(pm.local_pos[0], pm.local_pos[1]);
                let global = mech_state.body_point_global(body_id, &local, q);
                let sp = view.world_to_screen(global.x, global.y);
                let screen_pos = Pos2::new(sp[0], sp[1]);
                painter.circle_filled(screen_pos, 5.0, point_mass_color);
```

Replace with:

```rust
            for pm in &bp_body.point_masses {
                let Some(screen_pos) = point_mass_screen_pos(state, body_id, pm.local_pos) else {
                    continue;
                };
                painter.circle_filled(screen_pos, WEIGHT_RADIUS, point_mass_color);
                let entity = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: pm.id.clone() };
                if selected.as_ref() == Some(&entity) || state.multi_selected.contains(&entity) {
                    painter.circle_stroke(screen_pos, WEIGHT_RADIUS + 3.0, selected_ring);
                }
```

(g) Docs. `docs/FEATURES.md`, after the Place Mass bullet under "Mechanism Building":

Find in `docs/FEATURES.md`:

```markdown
- **Place Mass tool** -- two-phase workflow: select body, click to place. Move to Link and Reposition buttons. Preview circle at cursor.
```

Add after it:

```markdown
- **Weight selection** -- click a weight (point mass) marker to select it; Shift+click toggles it in the multi-selection. A weight wins over a joint or pin under the same click. Hovering a weight shows a highlight ring and a grab cursor.
```

`docs/ai/02-system.yaml`, `invariants_to_protect`: add the new invariant right after `point_mass_ids_unique_per_mechanism` and before `gravity_breakdown_sources_sum_to_model_gravity`.

Find in `docs/ai/02-system.yaml`:

```yaml
    io::next_point_mass_id. Address weights by (body_id, weight_id), never
    by list index.
```

Add after it:

```yaml
  - weights_picked_before_joints — a Select-mode click tests weights first
    (canvas::hit_testing::find_point_mass_at, WEIGHT_HIT_RADIUS = 8 px,
    below HIT_RADIUS = 12 px so a joint under a weight stays reachable from
    its outer ring), then joints, then pins. A hit selects
    SelectedEntity::Weight { body_id, weight_id }. Drawing and picking share
    point_mass_screen_pos (AppState::body_local_to_world at the current q),
    so a weight is picked exactly where it is drawn. Ground and blank-id
    weights are neither drawn nor picked.
```

`docs/ai/03-structure.yaml`, two lines:

Find in `docs/ai/03-structure.yaml`:

```yaml
      hit_testing: canvas/hit_testing.rs
```

Replace with:

```yaml
      hit_testing: canvas/hit_testing.rs (body-segment projection; weight picking find_point_mass_at + point_mass_screen_pos, which drawing shares)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force; extend it instead of re-implementing fixtures per module)
```

Replace with:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button_with; extend it instead of re-implementing fixtures per module)
```

`docs/architecture/ARCHITECTURE.md`, the point-mass paragraph:

Find in `docs/architecture/ARCHITECTURE.md`:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body.
```

Replace with:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body; clicking a marker selects that weight (`SelectedEntity::Weight`, addressed by body id and weight id).
```

`docs/ai/05-update-tracker.md`, a new top entry above the Task 4 entry (keep Task 4's heading as it is):

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Replace with:

```markdown
Reverse chronological (newest at top).

---

## 2026-09-29 — Payload weights Task 5: weight hit testing and selection
- `gui/canvas/hit_testing.rs`: `point_mass_screen_pos` (weight marker
  position at the current pose/view; `None` when not finite) and
  `find_point_mass_at(state, screen_pos, radius_px)` (nearest weight within
  the radius; ground and blank-id weights skipped; ties go to sorted body id,
  then list order).
- `gui/state/types.rs`: `SelectedEntity::Weight { body_id, weight_id }`.
- `gui/state/entity_crud.rs`: `AppState::body_local_to_world`, the inverse
  of `world_to_body_local`; Move to Link now uses it instead of an inlined
  transform.
- `gui/canvas/interaction.rs`: Select-mode clicks test weights first
  (`WEIGHT_HIT_RADIUS` 8 px), then joints, then pins; Shift+click toggles
  weights in the multi-selection. Hovering a weight draws a ring and sets
  the grab cursor (`weights_interactive`: plain Select mode, no other pick
  armed). `gui/canvas/rendering/mod.rs` draws weights through
  `point_mass_screen_pos` and rings selected weights.
- Tests: `gui::canvas::hit_testing::tests` (hit, miss, nearest, tie order,
  skipped weights, no blueprint), `gui::canvas::tests::weight_clicks`
  (click selects, empty click clears, weight beats joint on a pin,
  Shift+click toggle, grab cursor only in Select mode) and
  `body_local_to_world_*` in `gui/state/tests.rs`.
- `gui/test_support.rs` gains `primary_button_with(pos, pressed, modifiers)`;
  the canvas click tests build their pointer events with it instead of a
  local helper.
- Mutation check done: letting the joint loop run after a weight hit fails
  `a_weight_on_a_pin_wins_the_click_over_the_joint`.
```

Arrow-key nudge and the Delete shortcut ignore a selected weight for now: their `match` arms on `SelectedEntity` fall through `_` / `other`, so nothing else needed changing here.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib hit_testing
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib weight_clicks
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib body_local_to_world
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo clippy --all-targets 2>&1 | grep "generated"
```

Expected:
- `hit_testing`: `test result: ok. 8 passed; 0 failed`.
- `weight_clicks`: `test result: ok. 8 passed; 0 failed` (the 3 existing tests plus the 5 new click tests).
- `body_local_to_world`: `test result: ok. 3 passed; 0 failed`.
- Full lib run: `test result: ok. 828 passed; 0 failed; 0 ignored` (812 before this task plus 16 new: 3 in `gui/state/tests.rs`, 8 in `hit_testing`, 5 in `weight_clicks`).
- Clippy: the warning counts are unchanged from before the task (this task adds none). On the verified run the summary lines read `linkage-sim-rs (lib) generated 285 warnings` and `linkage-sim-rs (lib test) generated 301 warnings`, both before and after the task. If the base you run on differs, compare against a clippy run made before Step 3 rather than against these numbers.

Mutation check (do once, then restore). In `handle_click_selection`, change the `if hit.is_none() {` that guards the joint loop (the first of the two `if hit.is_none() {` lines in the Select arm) to `if true {`, so the joint loop runs after a weight hit. Then run `cargo test --lib weight_clicks`. Expected: 1 failed, 7 passed; `a_weight_on_a_pin_wins_the_click_over_the_joint` fails because the joint under the weight overrides the hit. Restore the line and re-run to see 8 passed.

A second mutation covers the rest of the click and hover path. Pass `-1.0` instead of `WEIGHT_HIT_RADIUS` to `find_point_mass_at` in `handle_click_selection` and comment out the `draw_weight_hover(ui, painter, response, state);` call. Expected: 4 failed, 4 passed (`clicking_a_weight_selects_it`, `a_weight_on_a_pin_wins_the_click_over_the_joint`, `shift_click_toggles_a_weight_in_the_multi_selection` and `hovering_a_weight_shows_a_grab_cursor_only_in_select_mode`). Restore both edits.

The full test run regenerates the PNGs in `docs/chebyshev_lambda/`. Check the tree and restore them if they show up as modified (the commit below lists its paths explicitly, so they would not be committed, but leave the tree clean):

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload status --short
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add docs/FEATURES.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md docs/architecture/ARCHITECTURE.md linkage-sim-rs/src/gui/canvas/colors.rs linkage-sim-rs/src/gui/canvas/hit_testing.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/canvas/mod.rs linkage-sim-rs/src/gui/canvas/rendering/mod.rs linkage-sim-rs/src/gui/state/entity_crud.rs linkage-sim-rs/src/gui/state/tests.rs linkage-sim-rs/src/gui/state/types.rs linkage-sim-rs/src/gui/test_support.rs
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -m "feat(gui): weight hit testing and selection

find_point_mass_at picks the weight nearest the cursor within a pixel
radius (ground and blank-id weights skipped; ties go to sorted body id,
then list order), using point_mass_screen_pos, which drawing now shares.
SelectedEntity::Weight { body_id, weight_id }. Select-mode clicks test
weights before joints and pins; Shift+click toggles weights in the
multi-selection; hovering a weight rings it and shows a grab cursor.
AppState::body_local_to_world replaces the Move to Link inline transform.
The canvas click tests build pointer events with the shared
test_support::primary_button_with.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

### Task 6: Drag and drop weights

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/types.rs` (new `WeightDrag` after `SelectedEntity`)
- Modify: `linkage-sim-rs/src/gui/state/mod.rs` (the `pub use types::{...}` re-export; new `weight_drag` field after `repositioning_point_mass`; its `Default` value)
- Modify: `linkage-sim-rs/src/gui/canvas/colors.rs` (`WEIGHT_COLOR` after `MOUNT_POINT_COLOR`; `LINK_PICK_RADIUS` after `WEIGHT_HIT_RADIUS`)
- Modify: `linkage-sim-rs/src/gui/canvas/hit_testing.rs` (`find_nearest_body_segment` delegates to the new `find_nearest_body_segment_where`; the `tests` module becomes `pub(super)` and gains the shared `segment` fixture and one new test)
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs`:
  - imports; `is_ctrl_dragging_image` moves to the top of `handle_interaction`
  - `handle_weight_drag` is called before the ground-pivot drag
  - `!is_dragging_weight` gates the ground-pivot start, the force-zone start and pan
  - Esc clears `weight_drag`
  - the three `60.0` link-pick literals (Move to Link, Place Mass twice) become `LINK_PICK_RADIUS`
  - the Move to Link / Reposition cursor preview uses `WEIGHT_COLOR`
  - `draw_weight_hover` skips while a drag is in progress
  - new `handle_weight_drag`, `weight_drop_target` and `draw_weight_drag_preview`
  - new `#[cfg(test)] mod tests` at the end
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs` (point-mass block: `WEIGHT_COLOR`, faded dragged weight; Select-mode hint)
- Modify: `linkage-sim-rs/src/gui/mod.rs`:
  - the Delete/Backspace block in `update` becomes a call to `handle_delete_shortcut`
  - Select button tooltip
  - new `handle_delete_shortcut` after the `impl eframe::App` block
  - new `#[cfg(test)] mod tests` at the end
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (`tests::weight_clicks`: imports, drag helpers, `setup` runs two idle frames, ten drag tests)
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (`primary_button` and `key_press`)
- Modify: `docs/FEATURES.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`, `docs/architecture/ARCHITECTURE.md`, `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`

**Interfaces:**
- Consumes:
  - Task 2:
    - `AppState::move_point_mass(&mut self, body_id: &str, weight_id: &str, target_body: &str, local_pos: [f64; 2]) -> bool`. It returns false for a ground target or a body missing from the blueprint, and true without an undo entry for a zero-distance drop.
    - `AppState::remove_point_mass_by_id(&mut self, body_id: &str, weight_id: &str) -> bool`
    - `AppState::find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson>`
    - `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>`
  - Task 5:
    - `find_point_mass_at(state: &AppState, screen_pos: Pos2, radius_px: f32) -> Option<(String, String)>` and `point_mass_screen_pos(state: &AppState, body_id: &str, local_pos: [f64; 2]) -> Option<Pos2>` (`gui/canvas/hit_testing.rs`)
    - `fn weights_interactive(state: &AppState) -> bool` (private, `gui/canvas/interaction.rs`)
    - `SelectedEntity::Weight { body_id: String, weight_id: String }`; `WEIGHT_RADIUS` and `WEIGHT_HIT_RADIUS` (`gui/canvas/colors.rs`)
    - the `gui::canvas::tests::weight_clicks` helpers `frame_with`, `frame`, `click_with`, `click`, `setup`, `world_of`, `screen_of`, `local_under`, `weight_screen`, `weight` and `assert_close`
    - `test_support::primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event`
  - Existing:
    - `AppState::body_local_to_world(&self, body_id: &str, local: [f64; 2]) -> [f64; 2]` and `AppState::world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2]`
    - `GridSettings::snap_point(&self, x: f64, y: f64) -> (f64, f64)` (no-op when `snap_enabled` is false)
    - `ViewTransform::screen_to_world(&self, sx: f32, sy: f32) -> [f64; 2]` and `ViewTransform::world_to_screen(&self, wx: f64, wy: f64) -> [f32; 2]`
    - `AppState::nc(&self, c: egui::Color32) -> egui::Color32` (Nathan-mode colour mapping)
    - `hit_testing::project_onto_segment(point: Pos2, seg_a: Pos2, seg_b: Pos2) -> Option<(Pos2, f32)>` and `draw_dashed_line(painter, start, end, stroke, dash_len, gap_len)`
- Produces:
  - `pub struct WeightDrag { pub body_id: String, pub weight_id: String, pub current_world: [f64; 2] }`. It derives `Debug, Clone, PartialEq` and is re-exported as `crate::gui::state::WeightDrag`.
  - `pub weight_drag: Option<WeightDrag>` on `AppState` (default `None`)
  - `pub fn find_nearest_body_segment_where(point: Pos2, segments: &[BodySegment], max_distance: f32, accept: impl Fn(&str) -> bool) -> Option<SegmentHit>` (`gui/canvas/hit_testing.rs`); `find_nearest_body_segment` keeps its signature and delegates with `|_| true`
  - `pub const WEIGHT_COLOR: Color32 = Color32::from_rgb(255, 200, 50);` and `pub const LINK_PICK_RADIUS: f32 = 60.0;` (`gui/canvas/colors.rs`)
  - private in `gui/canvas/interaction.rs`:
    - `fn handle_weight_drag(ui: &egui::Ui, painter: &egui::Painter, response: &egui::Response, state: &mut AppState, body_segments: &[BodySegment], is_shift: bool, is_ctrl_dragging_image: bool)`
    - `fn weight_drop_target(state: &AppState, own_body: &str, drop_screen: Pos2, body_segments: &[BodySegment]) -> String`
    - `fn draw_weight_drag_preview(painter: &egui::Painter, state: &AppState, drag: &WeightDrag, drop_screen: Pos2, target: &str, body_segments: &[BodySegment])`
  - private in `gui/mod.rs`: `fn handle_delete_shortcut(ctx: &egui::Context, state: &mut AppState)`
  - test-only helpers (shared, so later tasks do not re-implement them):
    - `pub(crate) fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event` and `pub(crate) fn key_press(key: egui::Key) -> egui::Event` in `gui/test_support.rs`
    - `pub(in crate::gui::canvas) fn segment(body: &str, y: f32) -> BodySegment` in `hit_testing::tests` (a horizontal 200 px bar; the module is now `pub(super)`)

- [ ] **Step 1: Write the failing tests**

Nineteen new tests: ten headless canvas drags in `weight_clicks`, three drop-target rules, one filtered segment search and five delete-shortcut cases. Three design points are worth knowing before you type them in.

- egui reports `drag_started` only after the pointer has moved more than 6 px from the press, so the helper `press_and_drag` presses, moves through the midpoint, then moves to the target. The implementation therefore has to pick the weight at `pointer.press_origin()`, not at the position where the drag is reported. The drag tests that must grab the weight depend on that (Step 4 lists them in its mutation check).
- The canvas recomputes `grid.spacing_m` from the zoom at the start of each frame, before it applies a pending fit-to-view. After one idle frame the grid spacing still belongs to the pre-fit zoom, so `setup` now runs two idle frames. The snapped-drop test would otherwise compute its expectation on a different grid than the canvas snaps to.
- DRY: the press/release events come from `test_support::primary_button` and `test_support::key_press` (also used by the `gui/mod.rs` delete tests), and the `hit_testing` and `interaction` tests share one `segment(body, y)` fixture.

`linkage-sim-rs/src/gui/test_support.rs`: add the two input helpers.

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
pub(crate) fn primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers }
}
```

Add after it:

```rust
/// A primary-button press (`pressed`) or release at `pos`, no modifiers.
pub(crate) fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    primary_button_with(pos, pressed, egui::Modifiers::NONE)
}

/// A key press event with no modifiers.
pub(crate) fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
}
```

`linkage-sim-rs/src/gui/canvas/hit_testing.rs`: the tests module must be visible to the interaction tests.

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::vec2;
```

Replace with:

```rust
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use eframe::egui::vec2;
```

Then add the shared fixture and the filtered-search test after `find_point_mass_at_skips_ground_blank_ids_and_non_finite_positions`:

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
        assert_eq!(find_point_mass_at(&state, at, 50.0), None);
    }
```

Add after it:

```rust
    /// A horizontal 200 px bar of `body` at screen height `y` (world
    /// coordinates unused). Shared with the canvas interaction tests.
    pub(in crate::gui::canvas) fn segment(body: &str, y: f32) -> BodySegment {
        BodySegment {
            screen_a: Pos2::new(0.0, y),
            screen_b: Pos2::new(200.0, y),
            world_a: [0.0, 0.0],
            world_b: [0.0, 0.0],
            body_id: body.to_string(),
            point_a_name: "A".to_string(),
            point_b_name: "B".to_string(),
        }
    }

    #[test]
    fn find_nearest_body_segment_where_skips_rejected_bodies() {
        let segments = vec![
            segment("near", 5.0),
            segment("far", 30.0),
        ];
        let p = Pos2::new(50.0, 0.0);
        let nearest = |max: f32, accept: fn(&str) -> bool| {
            find_nearest_body_segment_where(p, &segments, max, accept).map(|h| h.body_id)
        };
        assert_eq!(nearest(60.0, |_| true), Some("near".to_string()));
        assert_eq!(nearest(60.0, |b| b != "near"), Some("far".to_string()));
        assert_eq!(nearest(60.0, |_| false), None);
        assert_eq!(nearest(20.0, |b| b != "near"), None, "the distance limit still applies");
        assert_eq!(
            find_nearest_body_segment(p, &segments, 60.0).map(|h| h.body_id),
            Some("near".to_string()),
            "the unfiltered search is unchanged"
        );
    }
```

`linkage-sim-rs/src/gui/canvas/interaction.rs`: the drop-target rules, at the end of the file.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
        PendingCanvasPickKind::RefPt => {
            if let SweepMode::Trajectory { target, .. } = &mut state.sweep_mode {
                if let ControlTarget::Distance { ref_pt, .. } = target {
                    *ref_pt = [world_pos[0], world_pos[1]];
                }
            }
        }
    }
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hit_testing::tests::segment;
    use crate::gui::samples::SampleMechanism;

    fn fourbar() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state
    }

    #[test]
    fn a_weight_drop_lands_on_the_nearest_link_within_the_pick_radius() {
        let state = fourbar();
        let drop = Pos2::new(100.0, 0.0);
        let segments = vec![segment("crank", 10.0), segment("rocker", 40.0)];
        assert_eq!(weight_drop_target(&state, "coupler", drop, &segments), "crank");
        assert_eq!(weight_drop_target(&state, "rocker", drop, &segments), "crank");
        assert_eq!(weight_drop_target(&state, "crank", drop, &segments), "crank", "its own link is nearest");
    }

    #[test]
    fn a_weight_drop_far_from_every_link_stays_on_its_own_link() {
        let state = fourbar();
        let segments = vec![segment("crank", LINK_PICK_RADIUS + 1.0)];
        assert_eq!(weight_drop_target(&state, "coupler", Pos2::new(100.0, 0.0), &segments), "coupler");
        assert_eq!(weight_drop_target(&state, "coupler", Pos2::new(100.0, 0.0), &[]), "coupler");
    }

    #[test]
    fn a_weight_drop_skips_links_that_cannot_carry_a_weight() {
        // A compound actuator cylinder (expanded from a mount-point actuator at
        // build time, so absent from the blueprint) and ground lie nearer than
        // the crank.
        let state = fourbar();
        let drop = Pos2::new(100.0, 0.0);
        let segments = vec![segment("force_0_cyl", 2.0), segment(GROUND_ID, 4.0), segment("crank", 30.0)];
        assert_eq!(weight_drop_target(&state, "coupler", drop, &segments), "crank");
        let unusable = vec![segment("force_0_cyl", 2.0), segment(GROUND_ID, 4.0)];
        assert_eq!(weight_drop_target(&state, "coupler", drop, &unusable), "coupler");
    }
}
```

`linkage-sim-rs/src/gui/canvas/mod.rs`, in `mod weight_clicks`:

(a) Import the two helpers.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        use crate::gui::test_support::{primary_button_with, sorted_link_ids};
```

Replace with:

```rust
        use crate::gui::test_support::{key_press, primary_button, primary_button_with, sorted_link_ids};
```

(b) Add the drag helpers before `setup`, and replace the `setup` doc comment so it describes the second idle frame.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        /// Four-bar with a 2 kg weight "W1" on the first sorted link, after one
        /// idle frame (which applies the pending fit-to-view, so `state.view`
        /// is final). Returns the context, state, that link and a second link.
        fn setup() -> (egui::Context, AppState, String, String) {
```

Replace with:

```rust
        /// Press at `from` and drag through the midpoint to `to` without
        /// releasing. egui reports the drag once the pointer has moved more
        /// than 6 px from the press.
        fn press_and_drag(ctx: &egui::Context, state: &mut AppState, from: Pos2, to: Pos2) {
            frame(ctx, state, vec![egui::Event::PointerMoved(from)]);
            frame(ctx, state, vec![primary_button(from, true)]);
            frame(ctx, state, vec![egui::Event::PointerMoved(from + (to - from) * 0.5)]);
            frame(ctx, state, vec![egui::Event::PointerMoved(to)]);
        }

        fn release(ctx: &egui::Context, state: &mut AppState, at: Pos2) {
            frame(ctx, state, vec![primary_button(at, false)]);
        }

        fn drag(ctx: &egui::Context, state: &mut AppState, from: Pos2, to: Pos2) {
            press_and_drag(ctx, state, from, to);
            release(ctx, state, to);
        }

        /// Four-bar with a 2 kg weight "W1" on the first sorted link, after two
        /// idle frames: the first applies the pending fit-to-view (so
        /// `state.view` is final), the second adapts the grid spacing to the
        /// fitted zoom (so `state.grid` snaps as a drag will). Returns the
        /// context, state, that link and a second link.
        fn setup() -> (egui::Context, AppState, String, String) {
```

(c) The second idle frame at the end of `setup`.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            frame(&ctx, &mut state, Vec::new());
            (ctx, state, links[0].clone(), links[1].clone())
```

Replace with:

```rust
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state, links[0].clone(), links[1].clone())
```

(d) Geometry helpers after the `weight` helper.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        fn weight(body: &str, id: &str) -> SelectedEntity {
            SelectedEntity::Weight { body_id: body.to_string(), weight_id: id.to_string() }
        }
```

Add after it:

```rust
        /// The world point a drop at screen `pos` lands on: under the
        /// pointer, snapped to the grid when snapping is on.
        fn drop_world(state: &AppState, pos: Pos2) -> [f64; 2] {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (gx, gy) = state.grid.snap_point(wx, wy);
            [gx, gy]
        }

        /// Screen ends (sorted pin names) of a two-pin link's bar.
        fn link_ends(state: &AppState, body: &str) -> (Pos2, Pos2) {
            let mech = state.mechanism.as_ref().unwrap();
            let pins = &mech.bodies()[body].attachment_points;
            let mut names: Vec<&String> = pins.keys().collect();
            names.sort();
            assert_eq!(names.len(), 2, "fixture: a two-pin link");
            let end = |name: &String| screen_of(state, world_of(state, body, [pins[name].x, pins[name].y]));
            (end(names[0]), end(names[1]))
        }

        /// The screen point `t` of the way along a two-pin link's bar.
        fn along_link(state: &AppState, body: &str, t: f32) -> Pos2 {
            let (a, b) = link_ends(state, body);
            a + (b - a) * t
        }
```

(e) The ten drag tests, after `hovering_a_weight_shows_a_grab_cursor_only_in_select_mode`.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            state.active_tool = EditorTool::PlaceMass;
            assert_eq!(hover(&mut state, at), egui::CursorIcon::Default, "no weight pick in other tools");
        }
```

Add after it:

```rust
        #[test]
        fn dragging_a_weight_moves_it_on_release_as_one_undo_step() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0); // empty canvas, far from every link
            assert!(state.grid.snap_enabled, "fixture: snapping is on by default");
            let world = drop_world(&state, to);
            let [wx, wy] = state.view.screen_to_world(to.x, to.y);
            assert_ne!(world, [wx, wy], "fixture: the snap moves the drop point");
            let expected = state.world_to_body_local(&body, world[0], world[1]);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&body, "W1").expect("W1 stays on its link");
            assert_close("drop on the snapped grid point", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1, "one drag = one undo step");
            assert_eq!(state.selected, Some(weight(&body, "W1")));
            assert!(state.weight_drag.is_none());

            state.undo();
            let pm = state.find_point_mass(&body, "W1").expect("undo keeps W1");
            assert_eq!(pm.local_pos, [0.03, 0.02], "one undo restores the start position");
        }

        #[test]
        fn a_drag_with_snapping_off_drops_exactly_under_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let expected = local_under(&state, &body, to);

            drag(&ctx, &mut state, from, to);

            assert_close("unsnapped drop", state.find_point_mass(&body, "W1").unwrap().local_pos, expected);
        }

        #[test]
        fn the_drag_preview_leaves_the_blueprint_alone_until_release() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let depth = state.undo_history.undo_count();

            press_and_drag(&ctx, &mut state, from, to);

            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02], "not moved yet");
            assert_eq!(state.undo_history.undo_count(), depth, "no undo entry mid-drag");
            let preview = state.weight_drag.clone().expect("a weight drag is in progress");
            assert_eq!(preview.body_id, body);
            assert_eq!(preview.weight_id, "W1");
            assert_eq!(preview.current_world, drop_world(&state, to));
            assert_eq!(state.selected, Some(weight(&body, "W1")), "pressing a weight selects it");
            let cursor = frame_with(&ctx, &mut state, vec![egui::Event::PointerMoved(to)], egui::Modifiers::NONE)
                .platform_output
                .cursor_icon;
            assert_eq!(cursor, egui::CursorIcon::Grabbing);

            release(&ctx, &mut state, to);

            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.weight_drag.is_none());
        }

        #[test]
        fn dropping_near_another_link_reattaches_the_weight_at_the_drop_point() {
            let (ctx, mut state, body, other) = setup();
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = along_link(&state, &other, 0.5);
            let expected = local_under(&state, &other, to);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, to);

            assert!(state.find_point_mass(&body, "W1").is_none(), "W1 left its link");
            let pm = state.find_point_mass(&other, "W1").expect("W1 is on the link it was dropped on");
            assert_close("reattached where it was dropped", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1, "a reattach is one undo step");
            assert_eq!(state.selected, Some(weight(&other, "W1")), "the selection follows the weight");
        }

        #[test]
        fn dropping_on_its_own_link_next_to_a_neighbour_keeps_the_weight_there() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            // 90 % along its own link, beside the pin it shares with the rocker,
            // so the rocker's bar is inside the pick radius too.
            let to = along_link(&state, &body, 0.9);
            let (ra, rb) = link_ends(&state, "rocker");
            let near_rocker = super::super::hit_testing::project_onto_segment(to, ra, rb)
                .is_some_and(|(_, d)| d > 0.5 && d <= 60.0);
            assert!(near_rocker, "fixture: the drop point is within the pick radius of the rocker");
            let from = weight_screen(&state, &body, "W1");
            let expected = local_under(&state, &body, to);

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&body, "W1").expect("the nearest link wins: W1 stays");
            assert_close("moved along its own link", pm.local_pos, expected);
        }

        #[test]
        fn dragging_works_under_a_mounting_angle() {
            let (ctx, mut state, body, other) = setup();
            state.mounting_angle = 0.3;
            frame(&ctx, &mut state, Vec::new()); // the canvas copies it into the view
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = along_link(&state, &other, 0.5);
            let expected = local_under(&state, &other, to);

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&other, "W1").expect("reattached in the rotated view");
            assert_close("rotated view drop", pm.local_pos, expected);
        }

        #[test]
        fn escape_cancels_a_weight_drag_without_an_edit() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let depth = state.undo_history.undo_count();

            press_and_drag(&ctx, &mut state, from, to);
            frame(&ctx, &mut state, vec![key_press(egui::Key::Escape)]);
            assert!(state.weight_drag.is_none(), "Esc drops the preview");
            release(&ctx, &mut state, to);

            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }

        #[test]
        fn releasing_a_weight_outside_the_canvas_cancels_the_drag() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            // Inside the window, but in the panel margin around the canvas.
            let outside = Pos2::new(3.0, 400.0);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, outside);

            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }

        #[test]
        fn dragging_empty_canvas_still_pans_the_view() {
            let (ctx, mut state, body, _) = setup();
            let offset = state.view.offset;

            drag(&ctx, &mut state, Pos2::new(120.0, 700.0), Pos2::new(220.0, 650.0));

            assert_ne!(state.view.offset, offset, "the view panned");
            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
        }

        #[test]
        fn a_weight_does_not_drag_outside_select_mode() {
            let (ctx, mut state, body, _) = setup();
            state.active_tool = EditorTool::AddGroundPivot;
            let from = weight_screen(&state, &body, "W1");
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, Pos2::new(120.0, 700.0));

            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }
```

`linkage-sim-rs/src/gui/mod.rs`: the delete-shortcut tests, at the end of the file. They drive `handle_delete_shortcut` directly; the Backspace-while-typing test runs two frames so a real text field owns keyboard focus first.

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
        .unwrap_or_else(|| "background".to_string());
    load_background_image_from_bytes(ctx, &name, &bytes)
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::key_press;

    /// Four-bar with weights W1 and W2 on the coupler.
    fn fourbar_with_weights() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.add_point_mass("coupler", 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.01, 0.0]).as_deref(), Some("W2"));
        state
    }

    fn weight(id: &str) -> SelectedEntity {
        SelectedEntity::Weight { body_id: "coupler".to_string(), weight_id: id.to_string() }
    }

    /// One frame in which `key` is pressed and the delete shortcut runs.
    fn press(state: &mut AppState, key: egui::Key) {
        let ctx = egui::Context::default();
        let input = egui::RawInput { events: vec![key_press(key)], ..Default::default() };
        let _ = ctx.run(input, |ctx| handle_delete_shortcut(ctx, state));
    }

    #[test]
    fn delete_or_backspace_removes_the_selected_weight_as_one_undo_step() {
        for key in [egui::Key::Delete, egui::Key::Backspace] {
            let mut state = fourbar_with_weights();
            state.selected = Some(weight("W1"));
            let depth = state.undo_history.undo_count();

            press(&mut state, key);

            assert!(state.find_point_mass("coupler", "W1").is_none(), "{key:?} removes W1");
            assert!(state.find_point_mass("coupler", "W2").is_some(), "{key:?} keeps W2");
            assert_eq!(state.selected, None);
            assert_eq!(state.undo_history.undo_count(), depth + 1);
        }
    }

    #[test]
    fn delete_removes_every_multi_selected_weight() {
        let mut state = fourbar_with_weights();
        state.multi_selected = vec![weight("W1"), weight("W2")];
        state.selected = Some(weight("W2"));

        press(&mut state, egui::Key::Delete);

        assert!(state.find_point_mass("coupler", "W1").is_none());
        assert!(state.find_point_mass("coupler", "W2").is_none());
        assert!(state.multi_selected.is_empty());
        assert_eq!(state.selected, None);
    }

    #[test]
    fn delete_with_a_stale_weight_selection_changes_nothing() {
        let mut state = fourbar_with_weights();
        state.selected = Some(weight("W7")); // e.g. undone since it was selected
        let depth = state.undo_history.undo_count();

        press(&mut state, egui::Key::Delete);

        assert!(state.find_point_mass("coupler", "W1").is_some());
        assert!(state.find_point_mass("coupler", "W2").is_some());
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn other_keys_do_not_delete() {
        let mut state = fourbar_with_weights();
        state.selected = Some(weight("W1"));

        press(&mut state, egui::Key::A);

        assert!(state.find_point_mass("coupler", "W1").is_some());
        assert_eq!(state.selected, Some(weight("W1")));
    }

    #[test]
    fn backspace_while_typing_in_a_text_field_does_not_delete_the_selection() {
        let mut state = fourbar_with_weights();
        state.selected = Some(weight("W1"));
        let depth = state.undo_history.undo_count();
        let ctx = egui::Context::default();
        let mut text = String::from("2.5");

        // Frame 1: the user clicks into a text field, which takes keyboard focus.
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| ui.text_edit_singleline(&mut text).request_focus());
        });
        // Frame 2: Backspace, handled as in `update`: the shortcut first, then the panels.
        let input = egui::RawInput { events: vec![key_press(egui::Key::Backspace)], ..Default::default() };
        let _ = ctx.run(input, |ctx| {
            handle_delete_shortcut(ctx, &mut state);
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.text_edit_singleline(&mut text);
            });
        });

        assert!(state.find_point_mass("coupler", "W1").is_some(), "the weight survives");
        assert_eq!(state.selected, Some(weight("W1")));
        assert_eq!(state.undo_history.undo_count(), depth);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

From `linkage-sim-rs/` in the execution worktree:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib weight_clicks
```

Expected: the lib test target does not compile. The run ends with `error: could not compile ...` for the `linkage-sim-rs` (lib test) target, "due to 18 previous errors", made up of:

- 7 x `error[E0609]: no field ...` for `weight_drag` on type `gui::state::AppState` (the canvas drag tests)
- 7 x `error[E0425]: cannot find function ...` for `weight_drop_target` in this scope (the interaction tests)
- 2 x `error[E0425]: cannot find function ...` for `handle_delete_shortcut` in this scope (the `gui/mod.rs` tests)
- 1 x `error[E0425]: cannot find value ...` for `LINK_PICK_RADIUS` in this scope
- 1 x `error[E0425]: cannot find function ...` for `find_nearest_body_segment_where` in this scope

Nothing else is reported: the `test_support` helpers and the `pub(super)` `hit_testing::tests::segment` fixture added in Step 1 already compile.

- [ ] **Step 3: Implement**

Rules the implementation must satisfy (they are what the tests and the spec's section 3 pin down): the drag is a preview only (`AppState::weight_drag`) and the blueprint changes exactly once, on `drag_stopped_by(Primary)`, through one `move_point_mass`, so a drag is one undo step. The pointer is snapped with `GridSettings::snap_point` (a no-op when snapping is off). The target link is the nearest link within `LINK_PICK_RADIUS` (60 px) that the blueprint has and that is not ground, own link included, so a drop nearer the weight's own link than a neighbour's stays put, and a drop with no link in range stays on the current link. Ground and compound actuator bodies (drawn, but absent from the blueprint) never take a weight. Esc, or a release outside the canvas rect, commits nothing.

`linkage-sim-rs/src/gui/state/types.rs`: the preview state.

Find in `linkage-sim-rs/src/gui/state/types.rs`:

```rust
    Weight { body_id: String, weight_id: String },
}
```

Add after it:

```rust
/// A weight (point mass) being dragged on the canvas. Only a preview: the
/// blueprint changes once, when the drag ends (`drag_stopped`), through one
/// `AppState::move_point_mass` (one undo step, one rebuild).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightDrag {
    pub body_id: String,
    pub weight_id: String,
    /// Where the weight lands if released now (world, m): under the pointer,
    /// snapped to the grid when snapping is on.
    pub current_world: [f64; 2],
}
```

`linkage-sim-rs/src/gui/state/mod.rs`: re-export, field and default.

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
    PendingJointType, PendingCanvasPickKind, EditorTool, ContextMenuTarget, SelectedEntity,
    ValidationWarnings, SolverStatus, ForceResults, PropertyPanelTab,
```

Replace with:

```rust
    PendingJointType, PendingCanvasPickKind, EditorTool, ContextMenuTarget, SelectedEntity,
    WeightDrag, ValidationWarnings, SolverStatus, ForceResults, PropertyPanelTab,
```

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
    pub repositioning_point_mass: Option<(String, String)>,
```

Add after it:

```rust
    /// Weight being dragged on the canvas (preview only; committed on release).
    pub weight_drag: Option<WeightDrag>,
```

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
            repositioning_point_mass: None,
```

Add after it:

```rust
            weight_drag: None,
```

`linkage-sim-rs/src/gui/canvas/colors.rs`: the weight colour (so previews stop repeating the literal) and the shared pick radius.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const MOUNT_POINT_COLOR: Color32 = Color32::from_rgb(224, 86, 253); // #e056fd magenta
```

Add after it:

```rust
pub const WEIGHT_COLOR: Color32 = Color32::from_rgb(255, 200, 50); // weight (point mass) gold
```

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const WEIGHT_HIT_RADIUS: f32 = 8.0;
```

Add after it:

```rust
/// Pick radius, in screen pixels, for choosing a link by clicking or dropping
/// near its bar: Place Mass, Move to Link and dropping a dragged weight.
pub const LINK_PICK_RADIUS: f32 = 60.0;
```

`linkage-sim-rs/src/gui/canvas/hit_testing.rs`: `find_nearest_body_segment` becomes a wrapper over a filtered search. The loop body is the same computation, written with let-else and one combined condition (which also clears the `collapsible_if` and `unnecessary_map_or` clippy warnings the old loop carried).

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
    max_distance: f32,
) -> Option<SegmentHit> {
    let mut best: Option<(f32, Pos2, [f64; 2], String, String, String)> = None;

    for seg in segments {
        if let Some((proj_screen, dist)) = project_onto_segment(point, seg.screen_a, seg.screen_b) {
            if dist <= max_distance {
                if best.as_ref().map_or(true, |(d, _, _, _, _, _)| dist < *d) {
                    let ab_screen = seg.screen_b - seg.screen_a;
                    let ap_screen = proj_screen - seg.screen_a;
                    let t = if ab_screen.length_sq() > 1e-10 {
                        ap_screen.length() / ab_screen.length()
                    } else {
                        0.0
                    };
                    let world_x = seg.world_a[0] + t as f64 * (seg.world_b[0] - seg.world_a[0]);
                    let world_y = seg.world_a[1] + t as f64 * (seg.world_b[1] - seg.world_a[1]);

                    best = Some((dist, proj_screen, [world_x, world_y], seg.body_id.clone(),
                                 seg.point_a_name.clone(), seg.point_b_name.clone()));
                }
            }
        }
    }
```

Replace with:

```rust
    max_distance: f32,
) -> Option<SegmentHit> {
    find_nearest_body_segment_where(point, segments, max_distance, |_| true)
}

/// Find the nearest body line segment to a screen point among the segments
/// whose body id passes `accept`.
pub fn find_nearest_body_segment_where(
    point: Pos2,
    segments: &[BodySegment],
    max_distance: f32,
    accept: impl Fn(&str) -> bool,
) -> Option<SegmentHit> {
    let mut best: Option<(f32, Pos2, [f64; 2], String, String, String)> = None;

    for seg in segments.iter().filter(|seg| accept(&seg.body_id)) {
        let Some((proj_screen, dist)) = project_onto_segment(point, seg.screen_a, seg.screen_b) else {
            continue;
        };
        if dist <= max_distance && best.as_ref().is_none_or(|(d, ..)| dist < *d) {
            let ab_screen = seg.screen_b - seg.screen_a;
            let ap_screen = proj_screen - seg.screen_a;
            let t = if ab_screen.length_sq() > 1e-10 {
                ap_screen.length() / ab_screen.length()
            } else {
                0.0
            };
            let world_x = seg.world_a[0] + t as f64 * (seg.world_b[0] - seg.world_a[0]);
            let world_y = seg.world_a[1] + t as f64 * (seg.world_b[1] - seg.world_a[1]);

            best = Some((dist, proj_screen, [world_x, world_y], seg.body_id.clone(),
                         seg.point_a_name.clone(), seg.point_b_name.clone()));
        }
    }
```

`linkage-sim-rs/src/gui/canvas/interaction.rs`: imports.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    AddBodyState, AppState, DrawBodyGeometryState, EditorTool, ForceZoneDragState,
    PendingCanvasPickKind, SelectedEntity,
};
```

Replace with:

```rust
    AddBodyState, AppState, DrawBodyGeometryState, EditorTool, ForceZoneDragState,
    PendingCanvasPickKind, SelectedEntity, WeightDrag,
};
```

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
use super::hit_testing::{
    find_nearest_body_segment, find_point_mass_at, point_mass_screen_pos, AttachmentHit, BodySegment,
};
```

Replace with:

```rust
use super::hit_testing::{
    find_nearest_body_segment, find_nearest_body_segment_where, find_point_mass_at,
    point_mass_screen_pos, AttachmentHit, BodySegment,
};
```

`is_ctrl_dragging_image` is now needed before the weight drag, so it moves to the top of `handle_interaction`:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    let is_ctrl = ui.input(|i| i.modifiers.ctrl);
    let mut is_panning = false;
```

Replace with:

```rust
    let is_ctrl = ui.input(|i| i.modifiers.ctrl);
    let is_ctrl_dragging_image = is_ctrl && state.background_image.is_some();
    let mut is_panning = false;
```

Run the weight drag before the ground-pivot drag, and gate the ground-pivot start on it:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    // ── Interaction: ground pivot drag ─────────────────────────────────
    // Start drag when pointer is near a ground attachment point in Select mode.
    if state.active_tool == EditorTool::Select
        && response.drag_started_by(egui::PointerButton::Primary)
        && !is_shift
    {
```

Replace with:

```rust
    // ── Interaction: weight (point mass) drag ──────────────────────────
    // Before the ground-pivot and force-zone drags: a press on a weight
    // drags the weight, as a click on it selects the weight.
    handle_weight_drag(ui, painter, response, state, body_segments, is_shift, is_ctrl_dragging_image);
    let is_dragging_weight = state.weight_drag.is_some();

    // ── Interaction: ground pivot drag ─────────────────────────────────
    // Start drag when pointer is near a ground attachment point in Select mode.
    if state.active_tool == EditorTool::Select
        && response.drag_started_by(egui::PointerButton::Primary)
        && !is_shift
        && !is_dragging_weight
    {
```

Gate the force-zone start on it:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
        && state.dragging_force_zone_app_point.is_none()
        && !is_dragging_ground
    {
```

Replace with:

```rust
        && state.dragging_force_zone_app_point.is_none()
        && !is_dragging_ground
        && !is_dragging_weight
    {
```

Suppress panning while a weight is dragged (the `is_ctrl_dragging_image` definition moved up):

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    // Primary drag on empty space (Select mode) pans the view.
    // Suppress panning when dragging a ground pivot, force-zone app point,
    // or Ctrl+dragging the background image.
    let is_ctrl_dragging_image = is_ctrl && state.background_image.is_some();
    if response.dragged_by(egui::PointerButton::Primary)
        && !is_shift
        && !is_dragging_ground
        && !is_dragging_force_zone_ap
        && !is_ctrl_dragging_image
```

Replace with:

```rust
    // Primary drag on empty space (Select mode) pans the view.
    // Suppress panning when dragging a ground pivot, force-zone app point,
    // weight, or Ctrl+dragging the background image.
    if response.dragged_by(egui::PointerButton::Primary)
        && !is_shift
        && !is_dragging_ground
        && !is_dragging_force_zone_ap
        && !is_dragging_weight
        && !is_ctrl_dragging_image
```

Esc clears the preview:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
        state.repositioning_point_mass = None;
        state.adding_joint_point = None;
        state.active_tool = EditorTool::Select;
```

Replace with:

```rust
        state.repositioning_point_mass = None;
        state.weight_drag = None;
        state.adding_joint_point = None;
        state.active_tool = EditorTool::Select;
```

Move to Link and Place Mass share the pick radius with the drop rule instead of three `60.0` literals:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
            let hit = find_nearest_body_segment(pos, body_segments, 60.0);
```

Replace with:

```rust
            let hit = find_nearest_body_segment(pos, body_segments, LINK_PICK_RADIUS);
```

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
            if let Some(seg_hit) = find_nearest_body_segment(hover, body_segments, 60.0) {
```

Replace with:

```rust
            if let Some(seg_hit) = find_nearest_body_segment(hover, body_segments, LINK_PICK_RADIUS) {
```

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
                // Find nearest body segment within 60px
                if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, 60.0) {
```

Replace with:

```rust
                // Find nearest body segment within the link pick radius
                if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, LINK_PICK_RADIUS) {
```

The Move to Link / Reposition cursor preview drew its fill with `from_rgba_premultiplied(255, 200, 50, 128)`, which is not a valid premultiplied colour (the colour channels exceed the alpha). Use `WEIGHT_COLOR` through `state.nc`, so it follows Nathan mode like the other weight previews, with a 50 % alpha gold fill (the same as the Place Mass preview; a little dimmer than before):

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
            let preview_color = egui::Color32::from_rgba_premultiplied(255, 200, 50, 128);
            painter.circle_filled(pos, 6.0, preview_color);
            painter.circle_stroke(pos, 6.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 50)));
```

Replace with:

```rust
            let color = state.nc(WEIGHT_COLOR);
            painter.circle_filled(pos, 6.0, color.linear_multiply(0.5));
            painter.circle_stroke(pos, 6.0, Stroke::new(1.5, color));
```

Hover feedback stops while a drag is in progress (the drag preview draws its own cursor):

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
/// Hover feedback: a ring around the weight under the pointer and a grab
/// cursor, so the user sees what a click selects.
```

Replace with:

```rust
/// Hover feedback: a ring around the weight under the pointer and a grab
/// cursor, so the user sees what a click selects or a drag moves.
```

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    if !weights_interactive(state) || !response.hovered() {
```

Replace with:

```rust
    if !weights_interactive(state) || state.weight_drag.is_some() || !response.hovered() {
```

The three new functions, directly after `draw_weight_hover`. `handle_weight_drag` takes the weight drag state out of `AppState` for the frame and puts it back only while the drag continues, so every early return (Esc, release outside the canvas, missing pointer) drops the preview without an edit.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
}
```

Add after it:

```rust
/// Drag a weight to a new spot.
///
/// A primary press on a weight followed by a drag starts a preview
/// (`state.weight_drag`: the pointer, snapped to the grid when snapping is
/// on) and selects the weight. The release commits it with one
/// `move_point_mass` (one undo step, one rebuild): dropped nearest to another
/// link within `LINK_PICK_RADIUS`, the weight is reattached to that link at
/// the drop point; otherwise it moves on its own link. A drag that ends
/// without a primary release (Esc), or is released outside the canvas,
/// changes nothing.
fn handle_weight_drag(
    ui: &egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &mut AppState,
    body_segments: &[BodySegment],
    is_shift: bool,
    is_ctrl_dragging_image: bool,
) {
    if state.weight_drag.is_none()
        && weights_interactive(state)
        && response.drag_started_by(egui::PointerButton::Primary)
        && !is_shift
        && !is_ctrl_dragging_image
    {
        // Pick at the press point: egui reports the drag only once the
        // pointer has already moved a few pixels away from it.
        let press = ui.input(|i| i.pointer.press_origin()).or_else(|| response.interact_pointer_pos());
        let grabbed = press
            .and_then(|p| find_point_mass_at(state, p, WEIGHT_HIT_RADIUS))
            .and_then(|(body_id, weight_id)| {
                let local = state.find_point_mass(&body_id, &weight_id)?.local_pos;
                Some((body_id, weight_id, local))
            });
        if let Some((body_id, weight_id, local)) = grabbed {
            state.multi_selected.clear();
            state.selected = Some(SelectedEntity::Weight { body_id: body_id.clone(), weight_id: weight_id.clone() });
            let current_world = state.body_local_to_world(&body_id, local);
            state.weight_drag = Some(WeightDrag { body_id, weight_id, current_world });
        }
    }

    // Taken out for this frame; put back below while the drag goes on.
    let Some(mut drag) = state.weight_drag.take() else { return };

    let released = response.drag_stopped_by(egui::PointerButton::Primary);
    if !released && !response.dragged() {
        return; // egui ended the drag without a release (Esc): drop the preview
    }
    let Some(pointer) = response.interact_pointer_pos() else { return };
    if released && !response.rect.contains(pointer) {
        return; // released outside the canvas: cancel
    }

    let [wx, wy] = state.view.screen_to_world(pointer.x, pointer.y);
    let (gx, gy) = state.grid.snap_point(wx, wy);
    drag.current_world = [gx, gy];
    let [sx, sy] = state.view.world_to_screen(gx, gy);
    let drop_screen = Pos2::new(sx, sy);
    let target = weight_drop_target(state, &drag.body_id, drop_screen, body_segments);

    if released {
        let local = state.world_to_body_local(&target, gx, gy);
        if state.move_point_mass(&drag.body_id, &drag.weight_id, &target, local) {
            state.selected = Some(SelectedEntity::Weight { body_id: target, weight_id: drag.weight_id });
        }
        return;
    }

    draw_weight_drag_preview(painter, state, &drag, drop_screen, &target, body_segments);
    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    state.weight_drag = Some(drag);
}

/// The link a weight dropped at screen point `drop_screen` lands on: the
/// nearest link within `LINK_PICK_RADIUS` that can carry a weight (in the
/// blueprint and not ground, which rules out compound actuator bodies), else
/// `own_body`.
fn weight_drop_target(
    state: &AppState,
    own_body: &str,
    drop_screen: Pos2,
    body_segments: &[BodySegment],
) -> String {
    let carries_weights = |body_id: &str| {
        body_id != GROUND_ID && state.blueprint.as_ref().is_some_and(|bp| bp.bodies.contains_key(body_id))
    };
    find_nearest_body_segment_where(drop_screen, body_segments, LINK_PICK_RADIUS, carries_weights)
        .map_or_else(|| own_body.to_string(), |hit| hit.body_id)
}

/// Drag preview: the target link highlighted when the drop reattaches, a
/// dashed line from the weight to the drop point, and a marker there.
fn draw_weight_drag_preview(
    painter: &egui::Painter,
    state: &AppState,
    drag: &WeightDrag,
    drop_screen: Pos2,
    target: &str,
    body_segments: &[BodySegment],
) {
    let color = state.nc(WEIGHT_COLOR);
    if target != drag.body_id {
        let highlight = Stroke::new(6.0, color.linear_multiply(0.4));
        for seg in body_segments.iter().filter(|s| s.body_id == target) {
            painter.line_segment([seg.screen_a, seg.screen_b], highlight);
        }
    }
    let origin = state
        .find_point_mass(&drag.body_id, &drag.weight_id)
        .and_then(|pm| point_mass_screen_pos(state, &drag.body_id, pm.local_pos));
    if let Some(origin) = origin {
        draw_dashed_line(painter, origin, drop_screen, Stroke::new(1.0, color), 4.0, 3.0);
    }
    painter.circle_filled(drop_screen, WEIGHT_RADIUS, color);
    painter.circle_stroke(drop_screen, WEIGHT_HIT_RADIUS, Stroke::new(1.5, color));
}
```

`linkage-sim-rs/src/gui/canvas/rendering/mod.rs`: draw the weight with `WEIGHT_COLOR`, fade the one being dragged (the preview draws it at the drop point), and show a hint in Select mode.

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
        let point_mass_color = gc(Color32::from_rgb(255, 200, 50));
```

Replace with:

```rust
        let point_mass_color = gc(WEIGHT_COLOR);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
                painter.circle_filled(screen_pos, WEIGHT_RADIUS, point_mass_color);
```

Replace with:

```rust
                // A weight being dragged fades where it is; the drag preview
                // (canvas interaction) draws it at the drop point.
                let dragged = state.weight_drag.as_ref()
                    .is_some_and(|d| d.body_id == *body_id && d.weight_id == pm.id);
                let fill = if dragged { point_mass_color.linear_multiply(0.35) } else { point_mass_color };
                painter.circle_filled(screen_pos, WEIGHT_RADIUS, fill);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
                Some("Click anywhere to reposition the point mass (Esc to cancel)".to_string())
            } else {
                None
            }
```

Replace with:

```rust
                Some("Click anywhere to reposition the point mass (Esc to cancel)".to_string())
            } else {
                state.weight_drag.as_ref().map(|drag| format!(
                    "Release to drop weight '{}'; near another link it moves to that link (Esc or release outside the canvas to cancel)",
                    drag.weight_id
                ))
            }
```

`linkage-sim-rs/src/gui/mod.rs`: the Delete/Backspace block moves into `handle_delete_shortcut`, which also removes weights and stops firing while a text or number field has keyboard focus (Backspace while typing used to delete the selected body or joint).

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
        if ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
            if !self.state.multi_selected.is_empty() {
                // Delete all multi-selected items.
                let items: Vec<_> = self.state.multi_selected.drain(..).collect();
                for entity in items {
                    match entity {
                        SelectedEntity::Body(id) => self.state.remove_body(&id),
                        SelectedEntity::Joint(id) => self.state.remove_joint(&id),
                        _ => {}
                    }
                }
                self.state.selected = None;
            } else {
                match self.state.selected.take() {
                    Some(SelectedEntity::Body(id)) => {
                        self.state.remove_body(&id);
                    }
                    Some(SelectedEntity::Joint(id)) => {
                        self.state.remove_joint(&id);
                    }
                    other => {
                        self.state.selected = other;
                    }
                }
            }
        }
```

Replace with:

```rust
        handle_delete_shortcut(ctx, &mut self.state);
```

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
                    .on_hover_text("Select mode: click a link, joint, or body to select it. Drag empty space to pan the canvas. Shift+click to multi-select. Press Delete/Backspace to remove the selected entity. (Shortcut: Escape returns here from any tool)")
```

Replace with:

```rust
                    .on_hover_text("Select mode: click a link, joint, body, or weight to select it. Drag a weight to move it (drop it on another link to move it there). Drag empty space to pan the canvas. Shift+click to multi-select. Press Delete/Backspace to remove the selected entity. (Shortcut: Escape returns here from any tool)")
```

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
        self.state.tick_save_user_prefs();
    }
}
```

Add after it:

```rust
// ── Delete shortcut ─────────────────────────────────────────────────────────

/// Delete / Backspace removes the selection: every multi-selected item, else
/// the single selected body, joint or weight (each removal is one undo step).
/// Ignored while a widget has keyboard focus, so Backspace while typing in a
/// text or number field edits the text instead of deleting the selection.
fn handle_delete_shortcut(ctx: &egui::Context, state: &mut AppState) {
    if ctx.wants_keyboard_input()
        || !ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
    {
        return;
    }
    if !state.multi_selected.is_empty() {
        // Delete all multi-selected items.
        let items: Vec<_> = state.multi_selected.drain(..).collect();
        for entity in items {
            match entity {
                SelectedEntity::Body(id) => state.remove_body(&id),
                SelectedEntity::Joint(id) => state.remove_joint(&id),
                SelectedEntity::Weight { body_id, weight_id } => {
                    state.remove_point_mass_by_id(&body_id, &weight_id);
                }
                _ => {}
            }
        }
        state.selected = None;
    } else {
        match state.selected.take() {
            Some(SelectedEntity::Body(id)) => {
                state.remove_body(&id);
            }
            Some(SelectedEntity::Joint(id)) => {
                state.remove_joint(&id);
            }
            Some(SelectedEntity::Weight { body_id, weight_id }) => {
                state.remove_point_mass_by_id(&body_id, &weight_id);
            }
            other => {
                state.selected = other;
            }
        }
    }
}
```

Docs (they ship with the code):

- `docs/FEATURES.md`: add after the Weight selection bullet:

Find in `docs/FEATURES.md`:

```markdown
- **Weight selection** -- click a weight (point mass) marker to select it; Shift+click toggles it in the multi-selection. A weight wins over a joint or pin under the same click. Hovering a weight shows a highlight ring and a grab cursor.
```

Add after it:

```markdown
- **Weight drag and drop** -- press on a weight and drag it: a live preview (dashed line and marker) follows the pointer, snapped to the grid when snapping is on, and the weight moves once, on release, as one undo step. Dropped within 60 px of a different link (and nearer to it than to its own), the weight moves to that link at the drop point, keeping its id, name and mass; the target link is highlighted while dragging. Esc, or releasing outside the canvas, cancels. Delete/Backspace removes the selected weight.
- **Delete shortcut ignores typing** -- Delete/Backspace does nothing while a text or number field has keyboard focus, so editing a value never deletes the selected body, joint or weight.
```

- `docs/ai/02-system.yaml`: add two invariants after `weights_picked_before_joints`:

Find in `docs/ai/02-system.yaml`:

```yaml
    point_mass_screen_pos (AppState::body_local_to_world at the current q),
    so a weight is picked exactly where it is drawn. Ground and blank-id
    weights are neither drawn nor picked.
```

Add after it:

```yaml
  - weight_drag_commits_once_on_release — canvas weight drag
    (interaction.rs handle_weight_drag) keeps only a preview in
    AppState::weight_drag (pointer snapped by GridSettings::snap_point) and
    calls move_point_mass exactly once on drag_stopped_by(Primary). The
    target is the nearest link within LINK_PICK_RADIUS (60 px, shared with
    Place Mass and Move to Link) that the blueprint has and is not ground
    (compound actuator bodies are drawn but cannot carry weights), else the
    weight's own link. The own link is a candidate too (nearest link wins,
    as the spec says), so a drop nearer its own link than a neighbour's
    stays put. A drag egui ends without a primary release (Esc) or
    a release outside the canvas rect commits nothing. The drag starts from
    pointer.press_origin(): egui reports drag_started only after the pointer
    has moved > 6 px.
  - delete_shortcut_ignores_focused_widgets — gui/mod.rs
    handle_delete_shortcut returns early while ctx.wants_keyboard_input();
    before this, Backspace in any text/number field deleted the selected
    body, joint (and would have deleted a selected weight).
```

- `docs/ai/03-structure.yaml`: three one-line updates.

Find in `docs/ai/03-structure.yaml`:

```yaml
      interaction: canvas/interaction.rs (drag, pan, zoom, tool handlers)
```

Replace with:

```yaml
      interaction: canvas/interaction.rs (drag, pan, zoom, tool handlers; weight hover/drag - weights_interactive, handle_weight_drag, weight_drop_target)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - mod.rs (LinkageApp + update loop orchestration)
```

Replace with:

```yaml
    - mod.rs (LinkageApp + update loop orchestration; handle_delete_shortcut)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button_with; extend it instead of re-implementing fixtures per module)
```

Replace with:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press; extend it instead of re-implementing fixtures per module)
```

- `docs/architecture/ARCHITECTURE.md`: extend the weights paragraph.

Find in `docs/architecture/ARCHITECTURE.md`:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body; clicking a marker selects that weight (`SelectedEntity::Weight`, addressed by body id and weight id).
```

Replace with:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body; clicking a marker selects that weight (`SelectedEntity::Weight`, addressed by body id and weight id). Dragging a marker previews the move and commits it on release as one `move_point_mass` (one undo step); when the nearest link within 60 px of the drop point is a different link, the weight is reattached to it at the drop point.
```

- `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`, section 3: the drop rule now says nearest link wins (the spec read "within 60 px of a different link reattaches", which a drop nearer the own link than a neighbour's would contradict), and states where the weight lands and which bodies never take one.

Find in `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`:

```markdown
- Dropping within the existing 60 px pick radius of a different link reattaches
  the weight there, preserving its world position; otherwise it moves on its
  current link. Each drag is **one** undo step
  (`AppState::mutate_and_rebuild`). Delete removes the selected weight.
```

Replace with:

```markdown
- On release the weight lands at the drop point (snapped to the grid when
  snapping is on). The nearest link within the existing 60 px pick radius
  takes it, its own link included: a drop nearest a different link reattaches
  the weight there, and a drop nearer its own link, or with no link in range,
  moves it on its current link. Ground and compound actuator bodies never take
  a weight. Each drag is **one** undo step (`AppState::mutate_and_rebuild`).
  Delete removes the selected weight.
```

- `docs/ai/05-update-tracker.md`: new top entry, above the Task 5 entry.

Find in `docs/ai/05-update-tracker.md`:

```markdown
## 2026-09-29 — Payload weights Task 5: weight hit testing and selection
```

Replace with:

```markdown
## 2026-09-29 — Payload weights Task 6: drag and drop weights
- `gui/state/types.rs`: `WeightDrag { body_id, weight_id, current_world }`;
  `AppState::weight_drag: Option<WeightDrag>` (preview only).
- `gui/canvas/interaction.rs`: `handle_weight_drag` (runs before the
  ground-pivot / force-zone drags, which it pre-empts, and suppresses pan):
  press on a weight + drag selects it and previews (pointer snapped to the
  grid when snapping is on; target link highlighted; dashed line; grabbing
  cursor); release = one `move_point_mass` (one undo step), selection
  follows the weight. `weight_drop_target`: nearest link within
  `LINK_PICK_RADIUS` that the blueprint has and is not ground (compound
  actuator bodies are skipped), else its own link. Esc or a release outside
  the canvas cancels. Rendering fades the dragged weight and shows a hint.
- `gui/canvas/hit_testing.rs`: `find_nearest_body_segment_where` (filtered
  search; `find_nearest_body_segment` delegates). `colors.rs`:
  `LINK_PICK_RADIUS` (60 px, now also used by Place Mass and Move to Link),
  `WEIGHT_COLOR`.
- `gui/mod.rs`: the Delete/Backspace block becomes
  `handle_delete_shortcut`, removes a selected (or multi-selected) weight,
  and is ignored while a widget has keyboard focus (Backspace in a text
  field used to delete the selected body or joint).
- Tests: `gui::canvas::tests::weight_clicks` (move on release with snap and
  one undo step + undo, unsnapped drop, preview leaves the blueprint alone,
  reattach at the drop point, nearest-own-link stays, mounting angle, Esc,
  release outside the canvas, empty-canvas drag still pans, no drag outside
  Select mode), `gui::canvas::interaction::tests` (drop target rules),
  `gui::canvas::hit_testing::tests::find_nearest_body_segment_where_*`,
  `gui::tests` (Delete/Backspace removes selected and multi-selected
  weights, stale selection, other keys, Backspace while typing).
- Drop rule: the nearest link within the pick radius wins, its own link
  included, so a drop nearer its own link than a neighbour's stays put. The
  spec's Track 2 section 3 now says so (it read "within 60 px of a different
  link reattaches").
- Cleanups: the Move to Link / Reposition cursor preview uses `WEIGHT_COLOR`
  instead of a literal gold (and follows Nathan mode like the other weight
  previews). `find_nearest_body_segment_where` is clippy-clean (let-else,
  one condition, `is_none_or`), which also clears the three warnings the old
  `find_nearest_body_segment` carried.
- `gui/test_support.rs` gains `primary_button` and `key_press`; the canvas
  drag tests and the `gui::tests` delete tests use them instead of local
  copies. The canvas interaction tests reuse the `hit_testing` tests'
  `segment(body, y)` bar fixture.
- Mutation checks done: dropping the `wants_keyboard_input` guard fails the
  Backspace-while-typing test; accepting every body as a drop target fails
  `a_weight_drop_skips_links_that_cannot_carry_a_weight`.

## 2026-09-29 — Payload weights Task 5: weight hit testing and selection
```

- [ ] **Step 4: Run the tests to verify they pass**

From `linkage-sim-rs/` in the execution worktree:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib weight_clicks
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib gui::canvas::interaction::tests
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib find_nearest_body_segment_where
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib gui::tests::
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo clippy --lib --tests 2>&1 | grep generated
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected:

- `weight_clicks`: `18 passed` (the 8 click tests from Task 5 plus the 10 drag tests).
- `gui::canvas::interaction::tests`: `3 passed`. `find_nearest_body_segment_where`: `1 passed`. `gui::tests::`: `5 passed`.
- Full lib run: `test result: ok. 847 passed; 0 failed` (828 before this task, plus the 19 new tests).
- Clippy: `(lib) generated 282 warnings` and `(lib test) generated 298 warnings`. Before this task they were 285 and 301: the two `collapsible_if` warnings and the `unnecessary_map_or` warning that the old `find_nearest_body_segment` loop carried are gone. `hit_testing.rs:50` (`manual_range_contains` in `project_onto_segment`) is still reported; that function is untouched and the warning predates this task.
- The gate (`cargo test --all`, `cargo clippy --all-targets`, the WASM check) ends with `GATE PASS`.

The gate's test run can rewrite `docs/chebyshev_lambda/*.png` (it did in the original run). If `git status` shows them modified, restore them so they stay out of the commit:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Mutation checks. Apply each edit, run the named test to see it fail, then revert the edit (keep a copy of the file first and copy it back; do not use `git checkout` on these files, the task's work is uncommitted):

- In `handle_delete_shortcut` (`gui/mod.rs`), remove the `ctx.wants_keyboard_input()` guard, so the condition starts `if !ctx.input(...`. `cargo test --lib gui::tests::` then fails `backspace_while_typing_in_a_text_field_does_not_delete_the_selection` (4 passed, 1 failed).
- In `weight_drop_target` (`interaction.rs`), pass `|_| true` to `find_nearest_body_segment_where` instead of `carries_weights`. `cargo test --lib gui::canvas::interaction::tests` then fails `a_weight_drop_skips_links_that_cannot_carry_a_weight`.
- In `handle_weight_drag`, disable the `released && !response.rect.contains(pointer)` early return. `cargo test --lib weight_clicks` then fails `releasing_a_weight_outside_the_canvas_cancels_the_drag`.
- In `handle_weight_drag`, pick at `response.interact_pointer_pos()` instead of `pointer.press_origin()`. `cargo test --lib weight_clicks` then fails six of the drag tests (the ones that must grab the weight: `dragging_a_weight_moves_it_on_release_as_one_undo_step`, `a_drag_with_snapping_off_drops_exactly_under_the_pointer`, `the_drag_preview_leaves_the_blueprint_alone_until_release`, `dropping_near_another_link_reattaches_the_weight_at_the_drop_point`, `dropping_on_its_own_link_next_to_a_neighbour_keeps_the_weight_there`, `dragging_works_under_a_mounting_angle`), because egui reports the drag start 6 px or more away from the weight.

After the last revert, re-run `cargo test --lib` and confirm `847 passed`.

- [ ] **Step 5: Commit**

From the repository root of the execution worktree:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add \
  docs/FEATURES.md \
  docs/ai/02-system.yaml \
  docs/ai/03-structure.yaml \
  docs/ai/05-update-tracker.md \
  docs/architecture/ARCHITECTURE.md \
  docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md \
  linkage-sim-rs/src/gui/canvas/colors.rs \
  linkage-sim-rs/src/gui/canvas/hit_testing.rs \
  linkage-sim-rs/src/gui/canvas/interaction.rs \
  linkage-sim-rs/src/gui/canvas/mod.rs \
  linkage-sim-rs/src/gui/canvas/rendering/mod.rs \
  linkage-sim-rs/src/gui/mod.rs \
  linkage-sim-rs/src/gui/state/mod.rs \
  linkage-sim-rs/src/gui/state/types.rs \
  linkage-sim-rs/src/gui/test_support.rs
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -m "$(cat <<'EOF'
task 6: drag and drop weights

Press on a weight and drag: a preview (AppState::weight_drag, snapped to
the grid when snapping is on) follows the pointer and the release commits
one move_point_mass, so each drag is one undo step. The nearest link
within LINK_PICK_RADIUS (60 px, now shared with Place Mass and Move to
Link) takes the weight, its own link included; a different link means a
reattach at the drop point. Compound actuator bodies and ground are
skipped. The spec's drop rule now says nearest link wins. Esc or a release
outside the canvas cancels. Delete/Backspace removes a selected weight and
no longer fires while a text field has keyboard focus.

The Move to Link / Reposition preview uses WEIGHT_COLOR instead of a
literal gold, find_nearest_body_segment_where is clippy-clean, and the
tests share test_support::{primary_button, key_press} and one segment
fixture.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
)"
```

### Task 7: Braking bands and the Weight Breakdown plot

This task turns the per-weight sweep data of Task 4 into pictures. It adds:

- shaded **braking bands** on the Actuator Force and Actuator Power plots, from `WeightBreakdown::braking`;
- a new **Weight Breakdown** tab: one line per weight source plus "Other loads" and "Total", as force share or power share, each weight's line coloured per sample green (helping), red (hurting) or gray (neutral);
- the single helping/hurting/neutral palette (`canvas::classification_color`) that Task 9 will reuse for the canvas weight arrows;
- the sign wording fix: the actuator force tab and field tooltips say positive = extension, and the rated lines are "Rated (push)" / "Rated (pull)".

Three design points came out of verifying the implementation and are part of this task:

- **One sweep-x conversion.** The conversion from a sweep x value to the plotted x (metres to millimetres in stroke mode, degrees to the display angle unit in angle mode) was written inline at 8 places in `plot_panel`. `draw_braking_bands` would be a ninth. Instead of adding a copy, the task adds `plot_panel::sweep_x_to_display` and replaces all of them, and the new function has its own test. `current_driver_display` (radians plus the display offset) is not a sweep x value and keeps its own branch.
- **BL-026 is already fixed on main**, so the plotted actuator force and power are the *required* values in sizing mode and in stored-force mode. The braking bands come from `WeightBreakdown::braking` (required power) and the Total line from `total_force` / `total_power`, so both follow the plotted curves in both modes. The band test therefore runs in both modes (no sizing-mode-only workaround), and a second test checks that the Total line equals the plotted required actuator force and power. The docs carry no BL-026 caveat.
- **No inline stored-force loop in tests.** The band test uses `test_support::set_actuator_stored_force` (Task 4) instead of looping over `LinearActuator` elements by hand.

The docs for this code (02-system invariant, 03-structure lines, tracker entry, ENGINEERING_OUTPUTS section, FEATURES and README bullets) are part of this commit, per the docs-with-code rule.

**Files:**
- Create: `linkage-sim-rs/src/gui/plot_panel/weights.rs`
- Modify: `linkage-sim-rs/src/gui/plot_panel/mod.rs` (`PlotTab` enum and a tooltip const after it; `mod weights;`; tab buttons for Actuator Force, Actuator Power and the new Weight Breakdown; dispatcher arm; new `sweep_x_to_display` and its use at 4 existing sites; `mod tests`)
- Modify: `linkage-sim-rs/src/gui/plot_panel/actuator.rs` (imports; braking bands at the start of the force and power `plot.show` closures; rated-line names; `sweep_x_to_display` at 3 sites)
- Modify: `linkage-sim-rs/src/gui/plot_panel/dynamics.rs` (import; `sweep_x_to_display` in the driver-torque statics overlay)
- Modify: `linkage-sim-rs/src/gui/canvas/colors.rs` (import, palette after `FORCE_ZONE_OVERLAP_STROKE`, tests at the end)
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (the `colors` re-export)
- Modify: `linkage-sim-rs/src/gui/sweep/weights.rs` (`impl WeightBreakdown`, `mod tests`)
- Modify: `linkage-sim-rs/src/gui/state/mod.rs` (`AppState` field after `actuator_rated_force`, and its `Default`)
- Modify: `linkage-sim-rs/src/gui/property_panel/force_editor.rs` (const after the imports, LinearActuator `F:` field tip, tests at the end)
- Modify: `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`, `docs/architecture/ENGINEERING_OUTPUTS.md`, `docs/FEATURES.md`, `README.md`

**Interfaces:**
- Consumes:
  - Task 3: `crate::analysis::gravity_breakdown::{Classification, WeightSource, classify, max_abs_finite, BRAKE_TOL_REL}`.
    - `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Classification { Helping, Hurting, Neutral }`
    - `pub struct WeightSource { pub id: String, pub name: String, pub body_id: String, pub local_pos: [f64; 2], pub mass: f64, pub is_link_self_weight: bool }`
    - `pub fn classify(p_g: f64, max_abs_p_g: f64) -> Classification`, `pub fn max_abs_finite(values: &[f64]) -> f64`, `pub const BRAKE_TOL_REL: f64 = 1e-6`.
  - Task 4: `crate::gui::sweep::{WeightBreakdown, ShareBasis, SweepData}`.
    - `WeightBreakdown` has the pub fields `sources: Vec<WeightSource>`, `basis: ShareBasis`, `gravity_power`, `force_share`, `power_share: Vec<Vec<f64>>`, `other_force`, `other_power`, `total_force`, `total_power: Vec<f64>` and `braking: Vec<bool>`, plus `pub fn classification(&self, source: usize, sample: usize) -> Classification`. `total_force` / `total_power` are the required actuator force and power (BL-026).
    - `ShareBasis::{ActuatorForce, DriverTorque}`; `SweepData::weight_breakdown: Option<WeightBreakdown>`.
    - Test helpers in `sweep/weights.rs` `mod tests`: `fn swept(sample: SampleMechanism, sizing: bool, weights: &[(&str, f64, [f64; 2])]) -> AppState`, `fn breakdown(state: &AppState) -> (&SweepData, &WeightBreakdown)`, `const CHEBYSHEV_WEIGHTS`.
    - `crate::gui::test_support::set_actuator_stored_force(state: &mut AppState, force: f64)` (test-only; sets every `LinearActuator`'s stored force and calls `rebuild()`).
  - Task 2: `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>`.
  - Existing `plot_panel` helpers (private to `plot_panel`, visible to its child modules): `detect_plot_click(plot_ui: &egui_plot::PlotUi) -> Option<f64>`, `draw_angle_series_with_range(plot_ui, name: &str, color: egui::Color32, width: f32, x_deg_and_y: &[(f64, f64)], sweep: &SweepData, units: &DisplayUnits, nathan_mode: bool)`, `draw_toggle_markers(plot_ui, sweep, units)`, `draw_range_boundary_markers(plot_ui, sweep, units)`, `with_default_x_bounds<'a>(plot: Plot<'a>, plot_id_stem: &str, sweep: &SweepData, units: &DisplayUnits) -> Plot<'a>`, `x_axis_label_for_sweep(sweep: &SweepData, units: &DisplayUnits) -> String`.
- Produces:
  - `pub const WEIGHT_HELPING_COLOR`, `WEIGHT_HURTING_COLOR` and `WEIGHT_NEUTRAL_COLOR: egui::Color32`, and `pub fn classification_color(class: Classification) -> egui::Color32`, in `gui/canvas/colors.rs`, re-exported as `crate::gui::canvas::classification_color`. This is the shared palette; the canvas weight arrows (Task 9) must use it.
  - `impl WeightBreakdown { pub fn classifications(&self, source: usize) -> Vec<Classification> }`: `classification` at every sample of one source, the source's band computed once; empty when `source` is out of range.
  - `AppState { pub weight_breakdown_show_power: bool }`, default `false`, not persisted.
  - `PlotTab::WeightBreakdown` (the enum stays private to `plot_panel`).
  - `fn sweep_x_to_display(x: f64, sweep: &SweepData, units: &DisplayUnits) -> f64` in `plot_panel/mod.rs` (private; the one sweep-x to display-x conversion).
  - In `plot_panel::weights`, all `pub(super)`:
    - `enum RunColor { Weight(Classification), Other, Total }`, `struct Run { color: RunColor, points: Vec<(f64, f64)> }`, `struct BreakdownLine { name: String, runs: Vec<Run> }`
    - `fn braking_bands(xs: &[f64], braking: &[bool]) -> Vec<(f64, f64)>`
    - `fn band_y_extent(series: &[&[f64]]) -> Option<(f64, f64)>`
    - `fn draw_braking_bands(plot_ui: &mut egui_plot::PlotUi, sweep: &SweepData, units: &DisplayUnits, extent_series: &[&[f64]], nathan_mode: bool)`
    - `fn source_line_name(source: &WeightSource) -> String`
    - `fn breakdown_lines(xs: &[f64], b: &WeightBreakdown, show_power: bool) -> Vec<BreakdownLine>`
    - `fn breakdown_y_label(basis: ShareBasis, is_stroke: bool, show_power: bool) -> &'static str`
    - `fn draw_weight_breakdown(ui: &mut egui::Ui, sweep: &SweepData, current_driver_display: f64, units: &DisplayUnits, nathan_mode: bool, show_power: bool) -> Option<f64>`

- [ ] **Step 1: Write the failing tests**

Edit blocks below: "Find in `<path>`:" quotes existing text exactly as it is before this task; "Replace with:" swaps that text for the block; "Add after it:" puts the block after the quoted text with exactly one blank line between them.

(a) Create `linkage-sim-rs/src/gui/plot_panel/weights.rs` with only this test module. The implementation goes above it in Step 3. The two mode-sweeping tests at the end are the ones that pin BL-026: the band test runs with the sample's shipped stored force and in sizing mode, and the Total-line test runs in stored-force mode.

Create `linkage-sim-rs/src/gui/plot_panel/weights.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::AppState;
    use crate::gui::test_support::set_actuator_stored_force;
    use Classification::{Helping, Hurting, Neutral};

    const NAN: f64 = f64::NAN;

    #[test]
    fn braking_bands_run_from_midpoint_to_midpoint() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let braking = [false, true, true, false, false, true, false, false];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.5, 2.5), (4.5, 5.5)]);
    }

    #[test]
    fn braking_bands_stop_at_the_sweep_ends() {
        let xs = [0.0, 10.0, 20.0, 30.0, 40.0];
        let braking = [true, true, false, false, true];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.0, 15.0), (35.0, 40.0)]);
    }

    #[test]
    fn braking_bands_none_when_nothing_brakes_one_when_everything_does() {
        let xs = [0.0, 1.0, 2.0];
        assert!(braking_bands(&xs, &[false; 3]).is_empty());
        assert_eq!(braking_bands(&xs, &[true; 3]), vec![(0.0, 2.0)]);
        assert!(braking_bands(&[], &[]).is_empty());
    }

    #[test]
    fn braking_bands_skip_zero_width_and_non_finite_samples() {
        // A one-sample sweep has no width to shade.
        assert!(braking_bands(&[5.0], &[true]).is_empty());
        // Sample 2 has no x: it never brakes, and the bands next to it end
        // at their own sample instead of a NaN midpoint.
        let xs = [0.0, 1.0, NAN, 3.0, 4.0];
        let braking = [false, true, true, true, false];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.5, 1.0), (3.0, 3.5)]);
    }

    #[test]
    fn braking_bands_are_ordered_on_a_descending_sweep() {
        let xs = [3.0, 2.0, 1.0, 0.0];
        assert_eq!(braking_bands(&xs, &[false, true, true, false]), vec![(0.5, 2.5)]);
    }

    #[test]
    fn braking_bands_read_only_the_common_length() {
        // Mismatched lengths break the sweep invariant; never panic, shade
        // only the samples both series cover.
        let xs = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(braking_bands(&xs, &[false, true]), vec![(0.5, 1.0)]);
        assert_eq!(braking_bands(&xs[..2], &[false, true, true, true]), vec![(0.5, 1.0)]);
    }

    #[test]
    fn band_y_extent_spans_the_finite_values_of_every_series() {
        let a = [1.0, NAN, -3.0];
        let b = [f64::INFINITY, 7.0];
        assert_eq!(band_y_extent(&[&a, &b]), Some((-3.0, 7.0)));
        assert_eq!(band_y_extent(&[&[NAN, f64::NEG_INFINITY]]), None);
        assert_eq!(band_y_extent(&[]), None);
        // A flat series is widened so the band stays visible.
        assert_eq!(band_y_extent(&[&[-200.0, -200.0]]), Some((-210.0, -190.0)));
        assert_eq!(band_y_extent(&[&[0.0]]), Some((-0.05, 0.05)));
    }

    fn source(id: &str, name: &str, body_id: &str, is_link_self_weight: bool) -> WeightSource {
        WeightSource {
            id: id.to_string(),
            name: name.to_string(),
            body_id: body_id.to_string(),
            local_pos: [0.0, 0.0],
            mass: 1.0,
            is_link_self_weight,
        }
    }

    #[test]
    fn source_line_names_are_unique_per_source() {
        assert_eq!(source_line_name(&source("link:coupler", "coupler", "coupler", true)), "coupler (link)");
        assert_eq!(source_line_name(&source("link:b2", "Arm", "b2", true)), "Arm (link b2)");
        assert_eq!(source_line_name(&source("W2", "W2", "b2", false)), "W2");
        assert_eq!(source_line_name(&source("W1", "Robot", "b2", false)), "Robot (W1)");
        // Two weights with the same label still get separate legend entries.
        assert_ne!(
            source_line_name(&source("W3", "Robot", "b2", false)),
            source_line_name(&source("W1", "Robot", "b2", false))
        );
    }

    const HAND_XS: [f64; 6] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0];

    /// Six samples, two sources. Source 0 (the coupler's own weight) comes
    /// down, drifts into its neutral band (0.01 W < 1 % of 4 W) and goes up;
    /// source 1 (W1 "Robot") always goes up. Sample 40 failed (NaN
    /// everywhere); at 20 the actuator reverses (force shares NaN); at 50
    /// it retracts, so the hurting weights' force shares are negative.
    fn hand_breakdown() -> WeightBreakdown {
        WeightBreakdown {
            sources: vec![
                source("link:coupler", "coupler", "coupler", true),
                source("W1", "Robot", "coupler", false),
            ],
            basis: ShareBasis::ActuatorForce,
            gravity_power: vec![vec![4.0, 3.0, 0.01, -2.0, NAN, -4.0], vec![-1.0, -1.0, -1.0, -1.0, NAN, -1.0]],
            force_share: vec![vec![-2.0, -1.5, NAN, 1.0, NAN, -2.0], vec![0.5, 0.5, NAN, 0.5, NAN, -0.5]],
            power_share: vec![vec![-4.0, -3.0, -0.01, 2.0, NAN, 4.0], vec![1.0, 1.0, 1.0, 1.0, NAN, 1.0]],
            other_force: vec![5.0, 5.0, NAN, 5.0, NAN, 5.0],
            other_power: vec![6.0, 6.0, 6.0, 6.0, NAN, 6.0],
            total_force: vec![7.0, 7.0, NAN, 7.0, NAN, 7.0],
            total_power: vec![8.0, 8.0, 8.0, 8.0, NAN, 8.0],
            braking: vec![false; 6],
        }
    }

    fn run(color: RunColor, points: &[(f64, f64)]) -> Run {
        Run { color, points: points.to_vec() }
    }

    #[test]
    fn power_share_lines_change_colour_with_the_classification_and_break_at_failures() {
        let lines = breakdown_lines(&HAND_XS, &hand_breakdown(), true);
        let names: Vec<&str> = lines.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["coupler (link)", "Robot (W1)", "Other loads", "Total"]);
        // Each colour change restarts from the previous sample, so the line
        // stays connected; the failed sample at 40 leaves a gap.
        assert_eq!(
            lines[0].runs,
            vec![
                run(RunColor::Weight(Helping), &[(0.0, -4.0), (10.0, -3.0)]),
                run(RunColor::Weight(Neutral), &[(10.0, -3.0), (20.0, -0.01)]),
                run(RunColor::Weight(Hurting), &[(20.0, -0.01), (30.0, 2.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, 4.0)]),
            ]
        );
        assert_eq!(
            lines[1].runs,
            vec![
                run(RunColor::Weight(Hurting), &[(0.0, 1.0), (10.0, 1.0), (20.0, 1.0), (30.0, 1.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, 1.0)]),
            ]
        );
        assert_eq!(
            lines[2].runs,
            vec![
                run(RunColor::Other, &[(0.0, 6.0), (10.0, 6.0), (20.0, 6.0), (30.0, 6.0)]),
                run(RunColor::Other, &[(50.0, 6.0)]),
            ]
        );
        assert_eq!(
            lines[3].runs,
            vec![
                run(RunColor::Total, &[(0.0, 8.0), (10.0, 8.0), (20.0, 8.0), (30.0, 8.0)]),
                run(RunColor::Total, &[(50.0, 8.0)]),
            ]
        );
    }

    #[test]
    fn force_share_lines_keep_the_gravity_power_colours_and_gap_near_reversal() {
        let lines = breakdown_lines(&HAND_XS, &hand_breakdown(), false);
        // The colour follows the gravity power, not the sign of the force
        // share: at 50 both weights are hurting although their shares are
        // negative (the actuator retracts). The reversal at 20 and the
        // failure at 40 are gaps.
        assert_eq!(
            lines[0].runs,
            vec![
                run(RunColor::Weight(Helping), &[(0.0, -2.0), (10.0, -1.5)]),
                run(RunColor::Weight(Hurting), &[(30.0, 1.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, -2.0)]),
            ]
        );
        assert_eq!(
            lines[1].runs,
            vec![
                run(RunColor::Weight(Hurting), &[(0.0, 0.5), (10.0, 0.5)]),
                run(RunColor::Weight(Hurting), &[(30.0, 0.5)]),
                run(RunColor::Weight(Hurting), &[(50.0, -0.5)]),
            ]
        );
        assert_eq!(
            lines[2].runs,
            vec![
                run(RunColor::Other, &[(0.0, 5.0), (10.0, 5.0)]),
                run(RunColor::Other, &[(30.0, 5.0)]),
                run(RunColor::Other, &[(50.0, 5.0)]),
            ]
        );
        assert_eq!(
            lines[3].runs,
            vec![
                run(RunColor::Total, &[(0.0, 7.0), (10.0, 7.0)]),
                run(RunColor::Total, &[(30.0, 7.0)]),
                run(RunColor::Total, &[(50.0, 7.0)]),
            ]
        );
    }

    #[test]
    fn breakdown_lines_without_sources_or_with_short_series_never_panic() {
        let mut b = hand_breakdown();
        b.sources.clear();
        b.gravity_power.clear();
        b.force_share.clear();
        b.power_share.clear();
        let names: Vec<String> = breakdown_lines(&HAND_XS, &b, true).into_iter().map(|l| l.name).collect();
        assert_eq!(names, ["Other loads", "Total"]);

        // Series shorter than the sweep (a broken invariant): only the
        // common samples are drawn, uncoloured samples count as neutral.
        let mut b = hand_breakdown();
        b.gravity_power[0].truncate(1);
        let lines = breakdown_lines(&HAND_XS, &b, true);
        assert_eq!(
            lines[0].runs[..2],
            [
                run(RunColor::Weight(Helping), &[(0.0, -4.0)]),
                run(RunColor::Weight(Neutral), &[(0.0, -4.0), (10.0, -3.0), (20.0, -0.01), (30.0, 2.0)]),
            ]
        );
        assert!(breakdown_lines(&HAND_XS[..2], &hand_breakdown(), true).iter().all(|l| l.runs.len() == 1));
    }

    #[test]
    fn y_label_names_the_quantity_and_unit_for_each_basis() {
        use ShareBasis::{ActuatorForce, DriverTorque};
        assert_eq!(breakdown_y_label(ActuatorForce, false, false), "Actuator Force Share (N)");
        assert_eq!(breakdown_y_label(ActuatorForce, true, false), "Actuator Force Share (N)");
        assert_eq!(breakdown_y_label(ActuatorForce, false, true), "Actuator Power Share (W)");
        assert_eq!(breakdown_y_label(DriverTorque, false, false), "Driver Torque Share (N\u{00b7}m)");
        assert_eq!(breakdown_y_label(DriverTorque, true, false), "Driver Force Share (N)");
        assert_eq!(breakdown_y_label(DriverTorque, false, true), "Driver Power Share (W)");
        assert_eq!(breakdown_y_label(DriverTorque, true, true), "Driver Power Share (W)");
    }

    /// ParallelogramActuator with a 50 kg weight on the rocker, with the
    /// sample's shipped stored force and in sizing mode: the shaded bands
    /// cover exactly the samples where the plotted actuator power is
    /// negative beyond the braking tolerance, and the cycle both brakes and
    /// motors. Since BL-026 the plotted power is the required power in both
    /// modes, so the bands must follow the curve in stored-force mode too.
    #[test]
    fn braking_bands_cover_exactly_the_negative_actuator_power_samples() {
        for sizing in [false, true] {
            let mode = if sizing { "sizing" } else { "stored force" };
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramActuator);
            if sizing {
                set_actuator_stored_force(&mut state, 0.0);
            }
            state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
            state.compute_sweep();
            let data = state.sweep_data.as_ref().expect("sweep computed");
            let power = data.actuator_power.as_ref().expect("actuator power");
            let braking = &data.weight_breakdown.as_ref().expect("breakdown computed").braking;
            let bands = braking_bands(&data.angles_deg, braking);
            let tol = BRAKE_TOL_REL * max_abs_finite(power);
            let (mut n_braking, mut n_motoring) = (0, 0);
            for (k, &x) in data.angles_deg.iter().enumerate() {
                let shaded = bands.iter().any(|&(lo, hi)| lo <= x && x <= hi);
                let brakes = power[k] < -tol;
                assert_eq!(shaded, brakes, "{mode}, at {x} deg: P = {} W", power[k]);
                if brakes {
                    n_braking += 1;
                } else {
                    n_motoring += 1;
                }
            }
            assert!(n_braking > 0 && n_motoring > 0, "{mode}: {n_braking} braking, {n_motoring} motoring samples");
        }
    }

    /// With the ParallelogramActuator's shipped stored force, the Total line
    /// is the plotted actuator force in the force view and the plotted
    /// actuator power in the power view (both the required values since
    /// BL-026) at every sample where the actuator plot has a value.
    #[test]
    fn total_line_is_the_plotted_required_actuator_force_and_power() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
        state.compute_sweep();
        let data = state.sweep_data.as_ref().expect("sweep computed");
        let b = data.weight_breakdown.as_ref().expect("breakdown computed");
        let views = [
            (false, data.actuator_forces.as_ref().expect("actuator force")),
            (true, data.actuator_power.as_ref().expect("actuator power")),
        ];
        for (show_power, plotted) in views {
            let lines = breakdown_lines(&data.angles_deg, b, show_power);
            let total = lines.iter().find(|l| l.name == TOTAL_LINE_NAME).expect("Total line");
            // One colour, so the runs only break at gaps: no repeated points.
            let points: Vec<(f64, f64)> = total.runs.iter().flat_map(|r| r.points.iter().copied()).collect();
            let tol = 1e-9 * max_abs_finite(plotted);
            let mut compared = 0;
            for (&x, &want) in data.angles_deg.iter().zip(plotted) {
                if !want.is_finite() {
                    continue;
                }
                let (_, got) = points
                    .iter()
                    .find(|p| p.0 == x)
                    .unwrap_or_else(|| panic!("power view {show_power}: no Total point at {x} deg"));
                assert!((got - want).abs() <= tol, "power view {show_power}, at {x} deg: Total {got}, plotted {want}");
                compared += 1;
            }
            assert!(compared > 300, "power view {show_power}: only {compared} samples compared");
        }
    }
}
```

(b) In `linkage-sim-rs/src/gui/plot_panel/mod.rs`, register the module.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
mod trajectory;
```

Replace with:

```rust
mod trajectory;
mod weights;
```

(c) In the same file's `#[cfg(test)] mod tests`, add the `sweep_x_to_display` test after `nan_padding_is_skipped`. It covers both sweep modes, both angle units, and NaN passing through.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
        assert!((lo - 66.5).abs() < 1e-6);
        assert!((hi - 180.0).abs() < 1e-6);
    }
```

Add after it:

```rust
    /// The one sweep-x conversion every plot uses (data points, braking
    /// bands, toggle and range markers, default x bounds): metres to
    /// millimetres in stroke mode whatever the angle unit, degrees to the
    /// display angle unit in angle mode.
    #[test]
    fn sweep_x_to_display_converts_per_sweep_mode_and_angle_unit() {
        let angle = empty_trajectory_sweep_data(SweepMode::Angle);
        let stroke = empty_trajectory_sweep_data(SweepMode::Stroke);
        let rad_units = DisplayUnits { length: LengthUnit::Millimeters, angle: AngleUnit::Radians };
        assert!((sweep_x_to_display(90.0, &angle, &deg_units()) - 90.0).abs() < 1e-12);
        assert!((sweep_x_to_display(-45.0, &angle, &deg_units()) + 45.0).abs() < 1e-12);
        assert!((sweep_x_to_display(90.0, &angle, &rad_units) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(sweep_x_to_display(0.125, &stroke, &deg_units()), 125.0);
        assert_eq!(sweep_x_to_display(0.125, &stroke, &rad_units), 125.0);
        assert!(sweep_x_to_display(f64::NAN, &angle, &deg_units()).is_nan());
        assert!(sweep_x_to_display(f64::NAN, &stroke, &deg_units()).is_nan());
    }
```

Then add the tooltip test and the headless render tests after `outlier_fence_removed_short_series_unchanged`. `plot_panel_frame` runs one idle frame of `draw_plot_panel` with a chosen tab selected. The render tests check the Weight Breakdown tab in both views, the two banded actuator tabs with a rated force set, the driver-torque basis (a mechanism without an actuator), and a sweep with no breakdown at all.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
        assert!(finite_series(&[], &[]).is_empty());
    }
```

Add after it:

```rust
    /// The code's convention is positive = extension (the actuator pushes
    /// its ends apart); the tab tip used to say "positive = tension".
    #[test]
    fn actuator_force_tab_tip_states_the_extension_sign_convention() {
        assert!(ACTUATOR_FORCE_TAB_TIP.contains("Positive = extension"), "{ACTUATOR_FORCE_TAB_TIP}");
        assert!(!ACTUATOR_FORCE_TAB_TIP.contains("Positive = tension"), "{ACTUATOR_FORCE_TAB_TIP}");
        assert!(!ACTUATOR_FORCE_TAB_TIP.contains("compression"), "{ACTUATOR_FORCE_TAB_TIP}");
    }

    /// Run one idle frame of the plot panel with `tab` selected (the panel
    /// keeps its tab in egui memory under `ui.id().with("plot_tab")`).
    fn plot_panel_frame(state: &mut AppState, tab: PlotTab) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let tab_id = ui.id().with("plot_tab");
                ui.memory_mut(|mem| mem.data.insert_temp(tab_id, tab));
                draw_plot_panel(ui, state);
            });
        });
    }

    /// The Weight Breakdown tab (force and power view) and the actuator
    /// tabs with braking bands and a rated force render headlessly, and an
    /// idle frame changes neither the view toggle nor the pose.
    #[test]
    fn weight_breakdown_and_braking_band_tabs_render_idle_frames() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
        state.compute_sweep();
        let breakdown = state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().expect("breakdown");
        assert!(breakdown.braking.iter().any(|&b| b), "the fixture has braking bands to draw");
        state.actuator_rated_force = 500.0;
        for show_power in [false, true] {
            state.weight_breakdown_show_power = show_power;
            let angle = state.driver_angle;
            for tab in [PlotTab::WeightBreakdown, PlotTab::ActuatorForce, PlotTab::ActuatorPower] {
                plot_panel_frame(&mut state, tab);
                assert_eq!(state.weight_breakdown_show_power, show_power, "{tab:?} idle frame");
                assert_eq!(state.driver_angle, angle, "{tab:?} idle frame");
            }
        }
    }

    /// Without an actuator the tab shows driver-torque shares; with no link
    /// mass and no weight the sweep has no breakdown, and the tab and the
    /// band-less actuator tab still render.
    #[test]
    fn weight_breakdown_tab_renders_driver_basis_and_missing_breakdown() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;
        use crate::gui::sweep::ShareBasis;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let breakdown = state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().expect("breakdown");
        assert_eq!(breakdown.basis, ShareBasis::DriverTorque);
        for show_power in [false, true] {
            state.weight_breakdown_show_power = show_power;
            plot_panel_frame(&mut state, PlotTab::WeightBreakdown);
        }

        for body in state.blueprint.as_mut().unwrap().bodies.values_mut() {
            body.mass = 0.0;
        }
        state.rebuild();
        state.compute_sweep();
        assert!(state.sweep_data.as_ref().unwrap().weight_breakdown.is_none());
        plot_panel_frame(&mut state, PlotTab::WeightBreakdown);
        plot_panel_frame(&mut state, PlotTab::ActuatorForce);
    }
```

(d) In `linkage-sim-rs/src/gui/canvas/colors.rs`, add the palette tests at the end of the file.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const MAX_SCALE: f32 = 100_000.0;
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::Classification::{Helping, Hurting, Neutral};

    #[test]
    fn classification_colors_are_green_red_and_gray() {
        assert_eq!(classification_color(Helping), WEIGHT_HELPING_COLOR);
        assert_eq!(classification_color(Hurting), WEIGHT_HURTING_COLOR);
        assert_eq!(classification_color(Neutral), WEIGHT_NEUTRAL_COLOR);
        let green = WEIGHT_HELPING_COLOR;
        assert!(green.g() > green.r() && green.g() > green.b(), "helping is green: {green:?}");
        let red = WEIGHT_HURTING_COLOR;
        assert!(red.r() > red.g() && red.r() > red.b(), "hurting is red: {red:?}");
        let gray = WEIGHT_NEUTRAL_COLOR;
        let (lo, hi) = (gray.r().min(gray.g()).min(gray.b()), gray.r().max(gray.g()).max(gray.b()));
        assert!(hi - lo <= 16, "neutral is gray: {gray:?}");
    }

    /// Nathan Mode draws everything in grayscale: the three classes must
    /// stay apart by brightness (helping brightest, hurting darkest).
    #[test]
    fn classification_colors_stay_distinct_in_grayscale() {
        let lum = |c: Color32| i32::from(to_grayscale(c).r());
        let (help, neutral, hurt) = (lum(WEIGHT_HELPING_COLOR), lum(WEIGHT_NEUTRAL_COLOR), lum(WEIGHT_HURTING_COLOR));
        assert!(help - neutral >= 30 && neutral - hurt >= 30, "grayscale {help} / {neutral} / {hurt}");
    }
}
```

(e) In `linkage-sim-rs/src/gui/sweep/weights.rs` `mod tests`, add this test after `builder_applies_the_sweep_wide_rules`. It uses the existing `swept`, `breakdown` and `CHEBYSHEV_WEIGHTS` helpers from Task 4.

Find in `linkage-sim-rs/src/gui/sweep/weights.rs`:

```rust
        assert_eq!(b.classification(0, 4), Neutral, "sample index out of range");
    }
```

Add after it:

```rust
    /// `classifications` (one band computation per weight, used to colour
    /// whole plot lines) agrees with `classification` at every sample, on a
    /// sweep that has helping, hurting and neutral samples.
    #[test]
    fn classifications_match_classification_at_every_sample() {
        let state = swept(SampleMechanism::ChebyshevLambdaActuator, true, CHEBYSHEV_WEIGHTS);
        let (data, b) = breakdown(&state);
        let mut seen = Vec::new();
        for i in 0..b.sources.len() {
            let all = b.classifications(i);
            assert_eq!(all.len(), data.angles_deg.len(), "{}", b.sources[i].id);
            for (k, &c) in all.iter().enumerate() {
                assert_eq!(c, b.classification(i, k), "{} at {} deg", b.sources[i].id, data.angles_deg[k]);
                if !seen.contains(&c) {
                    seen.push(c);
                }
            }
        }
        assert_eq!(seen.len(), 3, "all three classes occur: {seen:?}");
        assert!(b.classifications(b.sources.len()).is_empty(), "source index out of range");
    }
```

(f) In `linkage-sim-rs/src/gui/property_panel/force_editor.rs`, add the tooltip test at the end of the file.

Find in `linkage-sim-rs/src/gui/property_panel/force_editor.rs`:

```rust
                force: make_element([point[0], y]),
            });
        }
    });
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The code's convention is positive = extension (the actuator pushes
    /// its attachment points apart); the field tip used to say the opposite.
    #[test]
    fn actuator_force_field_tip_states_the_extension_sign_convention() {
        assert!(ACTUATOR_FORCE_FIELD_TIP.contains("Positive = extension"), "{ACTUATOR_FORCE_FIELD_TIP}");
        assert!(!ACTUATOR_FORCE_FIELD_TIP.contains("Positive = tension"), "{ACTUATOR_FORCE_FIELD_TIP}");
        assert!(!ACTUATOR_FORCE_FIELD_TIP.contains("compression"), "{ACTUATOR_FORCE_FIELD_TIP}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `linkage-sim-rs/`:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: the lib tests do not compile. The build ends with `error: could not compile `linkage-sim-rs` (lib test) due to 110 previous errors` (the exact count can differ by a few between toolchains). Every error is a missing new item, reported from the test code:

- `error[E0432]: unresolved import `Classification`` and `unresolved import `ShareBasis``, from the `use Classification::{Helping, Hurting, Neutral};` and `use ShareBasis::{ActuatorForce, DriverTorque};` lines of the `weights.rs` test module (the implementation's own imports do not exist yet).
- `error[E0425]: cannot find function` for `braking_bands`, `band_y_extent`, `breakdown_lines`, `breakdown_y_label`, `source_line_name`, `classification_color` and `sweep_x_to_display`; `cannot find value` for `ACTUATOR_FORCE_TAB_TIP`, `ACTUATOR_FORCE_FIELD_TIP`, `WEIGHT_HELPING_COLOR`, `WEIGHT_HURTING_COLOR`, `WEIGHT_NEUTRAL_COLOR` and `TOTAL_LINE_NAME`.
- `error[E0433]: failed to resolve: use of undeclared type `RunColor``, `error[E0412]: cannot find type` for `WeightSource`, `WeightBreakdown`, `Run` and `RunColor`, and `error[E0422]: cannot find struct, variant or union type` for `WeightSource`, `WeightBreakdown` and `Run`.
- `error[E0599]: no method named `classifications` found for reference `&WeightBreakdown``, and `no variant or associated item named `WeightBreakdown` found for enum `PlotTab``.
- `error[E0609]: no field `weight_breakdown_show_power` on type `AppState``.
- One `error[E0308]: mismatched types` at `band_y_extent(&[&a, &b])` in `band_y_extent_spans_the_finite_values_of_every_series`. It is a knock-on of the missing function: `&[&a, &b]` mixes a 3-element and a 2-element array, which only coerces once the parameter type `&[&[f64]]` exists. It disappears in Step 3.

Compilation failing is the red state here; no test has run yet.

- [ ] **Step 3: Implement**

(a) `linkage-sim-rs/src/gui/canvas/colors.rs`: import the classification type. Brightness of the three colours also differs, so Nathan Mode (grayscale) keeps the classes apart; the grayscale test in Step 1 pins that.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
use eframe::egui::Color32;
```

Replace with:

```rust
use eframe::egui::Color32;

use crate::analysis::gravity_breakdown::Classification;
```

Then add the palette after `FORCE_ZONE_OVERLAP_STROKE`.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const FORCE_ZONE_OVERLAP_STROKE: Color32 = Color32::from_rgb(255, 204, 0);
```

Add after it:

```rust
// Payload weights: helping / hurting / neutral (gravity_breakdown::classify).
// Brightness also differs (grayscale ~186 / ~100 / ~141) so Nathan Mode keeps
// the three classes apart.
pub const WEIGHT_HELPING_COLOR: Color32 = Color32::from_rgb(110, 235, 140);
pub const WEIGHT_HURTING_COLOR: Color32 = Color32::from_rgb(220, 50, 50);
pub const WEIGHT_NEUTRAL_COLOR: Color32 = Color32::from_rgb(140, 140, 150);

/// Colour of a weight that is helping (green), hurting (red) or neutral
/// (gray) at a sample. The single palette for the canvas weight arrows and
/// the Weight Breakdown plot lines, so the two always agree.
pub fn classification_color(class: Classification) -> Color32 {
    match class {
        Classification::Helping => WEIGHT_HELPING_COLOR,
        Classification::Hurting => WEIGHT_HURTING_COLOR,
        Classification::Neutral => WEIGHT_NEUTRAL_COLOR,
    }
}
```

(b) `linkage-sim-rs/src/gui/canvas/mod.rs`: re-export the palette function next to `to_grayscale`.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
pub use colors::to_grayscale;
```

Replace with:

```rust
pub use colors::{classification_color, to_grayscale};
```

(c) `linkage-sim-rs/src/gui/sweep/weights.rs`: add `classifications` to `impl WeightBreakdown`. The plot colours whole lines, so the weight's sweep-wide neutral band is computed once instead of once per sample.

Find in `linkage-sim-rs/src/gui/sweep/weights.rs`:

```rust
        gb::classify(p_g, gb::max_abs_finite(series))
    }
}
```

Replace with:

```rust
        gb::classify(p_g, gb::max_abs_finite(series))
    }

    /// [`Self::classification`] at every sample of `source`, with the
    /// weight's sweep-wide band computed once instead of per sample (the
    /// Weight Breakdown plot colours whole lines with it). Empty when
    /// `source` is out of range.
    pub fn classifications(&self, source: usize) -> Vec<Classification> {
        let Some(series) = self.gravity_power.get(source) else {
            return Vec::new();
        };
        let max_abs_p_g = gb::max_abs_finite(series);
        series.iter().map(|&p_g| gb::classify(p_g, max_abs_p_g)).collect()
    }
}
```

(d) `linkage-sim-rs/src/gui/state/mod.rs`: add the view toggle to `AppState` and to its `Default`. It is a per-session view choice and is not persisted.

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
    /// When 0.0, the margin overlay is disabled.
    pub actuator_rated_force: f64,
```

Replace with:

```rust
    /// When 0.0, the margin overlay is disabled.
    pub actuator_rated_force: f64,
    /// Weight Breakdown plot view: power shares (W) when true, force shares
    /// (N, or driver-torque shares without an actuator) when false.
    pub weight_breakdown_show_power: bool,
```

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
            adding_joint_point: None,
            actuator_rated_force: 0.0,
```

Replace with:

```rust
            adding_joint_point: None,
            actuator_rated_force: 0.0,
            weight_breakdown_show_power: false,
```

(e) `linkage-sim-rs/src/gui/property_panel/force_editor.rs`: the LinearActuator force field tip said the opposite of the code's convention (`forces::elements::evaluation`: positive = extension). Make it a constant so a test can pin the wording.

Find in `linkage-sim-rs/src/gui/property_panel/force_editor.rs`:

```rust
use crate::gui::state::AppState;
```

Add after it:

```rust
/// Hover text of the LinearActuator force field. Sign convention of the
/// element (`forces::elements::evaluation`): positive = extension.
const ACTUATOR_FORCE_FIELD_TIP: &str = "Constant axial force applied by the actuator in Newtons. Positive = extension (pushes the attachment points apart), negative = retraction (pulls them together). Set to 0 to solve for the required actuator force from statics.";
```

Find in `linkage-sim-rs/src/gui/property_panel/force_editor.rs`:

```rust
                    .on_hover_text("Constant axial force applied by the actuator in Newtons. Positive = tension (pulling points together), negative = compression (pushing apart). Set to 0 to solve for the required actuator force from statics.")
```

Replace with:

```rust
                    .on_hover_text(ACTUATOR_FORCE_FIELD_TIP)
```

(f) `linkage-sim-rs/src/gui/plot_panel/mod.rs`. First the tab enum and the Actuator Force tooltip constant (same sign wording fix).

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    ActuatorPower,
    OutputForce,
}
```

Replace with:

```rust
    ActuatorPower,
    WeightBreakdown,
    OutputForce,
}

/// Hover text of the Actuator Force tab. Sign convention of the
/// LinearActuator element (`forces::elements::evaluation`): positive =
/// extension.
const ACTUATOR_FORCE_TAB_TIP: &str = "Force in the linear actuator vs. driver angle. Red line = statics only (no inertia); cyan dashed = with inertia. Positive = extension (the actuator pushes its ends apart), negative = retraction (it pulls them together). Shaded bands mark where the load drives the actuator (braking). If a rated force is entered, a green safe-zone band is shown.";
```

The Actuator Force tab button uses the constant.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
            ).on_hover_text("Force in the linear actuator vs. driver angle. Red line = statics only (no inertia); cyan dashed = with inertia. Positive = tension (extending), negative = compression (retracting). If a rated force is entered, a green safe-zone band is shown.");
```

Replace with:

```rust
            ).on_hover_text(ACTUATOR_FORCE_TAB_TIP);
```

The Actuator Power tab tip mentions braking, and the new Weight Breakdown tab button follows it. The tab is enabled only when the sweep has a breakdown (some link mass or weight).

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
            ).on_hover_text("Mechanical power (Force \u{d7} Speed) delivered by the linear actuator in Watts vs. driver angle. Red = statics only; cyan dashed = with inertia. Peak power determines the motor/pump sizing requirement.");
        });

        // Only show output force tab when force zone data exists.
```

Replace with:

```rust
            ).on_hover_text("Mechanical power (Force \u{d7} Speed) delivered by the linear actuator in Watts vs. driver angle. Red = statics only; cyan dashed = with inertia. Peak power determines the motor/pump sizing requirement. Negative power means the load drives the actuator (braking, shaded bands).");
        });

        // Only show the weight breakdown tab when the sweep has one (some
        // link mass or weight).
        let has_wb = sweep.weight_breakdown.is_some();
        ui.add_enabled_ui(has_wb, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::WeightBreakdown,
                "Weight Breakdown",
            ).on_hover_text("Each weight's share of the load the actuator (or, without an actuator, the driver) must carry vs. driver angle: one line per link self-weight and per placed weight, plus Other loads (springs, force zones, external loads, end stops) and the required Total. A line is green where that weight comes down and helps, red where the actuator lifts it, gray where it moves sideways. Force share = -m g\u{b7}v divided by the actuator speed dL/dt (by the driver rate without an actuator), left blank near stroke reversal; power share = -m g\u{b7}v, negative when the weight gives power back.");
        });

        // Only show output force tab when force zone data exists.
```

The dispatcher gets a Weight Breakdown arm with the force/power toggle above the plot.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
        PlotTab::ActuatorPower => {
            actuator::draw_actuator_power(ui, sweep, current_driver_display, &state.display_units, nm)
        }
```

Replace with:

```rust
        PlotTab::ActuatorPower => {
            actuator::draw_actuator_power(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::WeightBreakdown => {
            ui.horizontal(|ui| {
                ui.label("Show:");
                ui.selectable_value(&mut state.weight_breakdown_show_power, false, "Force share")
                    .on_hover_text("Each weight's share of the required actuator force (N), or of the driver torque without an actuator. Speed-independent.");
                ui.selectable_value(&mut state.weight_breakdown_show_power, true, "Power share")
                    .on_hover_text("Each weight's share of the actuator power (W) at the driver's configured speed. Negative = the weight gives power back (helping).");
            });
            weights::draw_weight_breakdown(
                ui,
                sweep,
                current_driver_display,
                &state.display_units,
                nm,
                state.weight_breakdown_show_power,
            )
        }
```

Now the shared conversion. Add `sweep_x_to_display` between `display_to_radians` and `x_axis_label_for_sweep`.

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
/// Return the x-axis label string for plots that sweep over the driver variable.
```

Replace with:

```rust
/// Convert a sweep x value (`SweepData::angles_deg` entry, toggle angle or
/// range bound) to the plot's display x: metres to millimetres in stroke
/// mode, degrees to the configured display angle unit in angle mode.
///
/// The single place the plots map sweep x to screen x; clicks go back
/// through `display_to_radians` (angle) or `* 1e-3` (stroke).
fn sweep_x_to_display(x: f64, sweep: &SweepData, units: &DisplayUnits) -> f64 {
    if sweep.sweep_mode.is_stroke() {
        x * 1000.0
    } else {
        units.angle(x.to_radians())
    }
}

/// Return the x-axis label string for plots that sweep over the driver variable.
```

Replace the inline copies in `plot_panel/mod.rs` (four sites: `compute_default_x_bounds`, `draw_toggle_markers`, `draw_range_boundary_markers`, `draw_angle_series_with_range`).

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
/// In angle mode the bounds are converted from body-frame radians to the
/// configured display angle unit. In stroke mode they're converted from
/// metres to millimetres. This matches the per-plot conversion that
/// `draw_angle_series_with_range` already applies to data points.
```

Replace with:

```rust
/// The bounds go through `sweep_x_to_display`, the same conversion the
/// plots apply to their data points.
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    if sweep.sweep_mode.is_stroke() {
        Some((raw_min * 1000.0, raw_max * 1000.0))
    } else {
        Some((
            units.angle(raw_min.to_radians()),
            units.angle(raw_max.to_radians()),
        ))
    }
```

Replace with:

```rust
    Some((
        sweep_x_to_display(raw_min, sweep, units),
        sweep_x_to_display(raw_max, sweep, units),
    ))
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    let is_stroke = sweep.sweep_mode.is_stroke();
    for (i, &toggle_val) in sweep.toggle_angles.iter().enumerate() {
        let toggle_display = if is_stroke { toggle_val * 1000.0 } else { units.angle(toggle_val.to_radians()) };
```

Replace with:

```rust
    for (i, &toggle_val) in sweep.toggle_angles.iter().enumerate() {
        let toggle_display = sweep_x_to_display(toggle_val, sweep, units);
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
            let is_stroke = sweep.sweep_mode.is_stroke();
            let min_display = if is_stroke { min_val * 1000.0 } else { units.angle(min_val.to_radians()) };
            let max_display = if is_stroke { max_val * 1000.0 } else { units.angle(max_val.to_radians()) };
```

Replace with:

```rust
            let min_display = sweep_x_to_display(min_val, sweep, units);
            let max_display = sweep_x_to_display(max_val, sweep, units);
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    let is_stroke = sweep.sweep_mode.is_stroke();
    let to_display = |x: f64| -> f64 {
        if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) }
    };
```

Replace with:

```rust
    let to_display = |x: f64| sweep_x_to_display(x, sweep, units);
```

(g) `linkage-sim-rs/src/gui/plot_panel/weights.rs`: insert the implementation above the `#[cfg(test)]` line of the file from Step 1, leaving one blank line between the last `}` of the implementation and `#[cfg(test)]`. The pure pieces (`braking_bands`, `band_y_extent`, `source_line_name`, `split_runs`, `breakdown_lines`, `breakdown_y_label`) hold the logic and are unit-tested; the two draw functions only map them onto egui_plot items. `band_y_extent` sizes the bands from the data's finite y range, never from `plot_ui.plot_bounds()`: egui_plot fits its auto bounds to every item plus a margin, so a band sized from the current bounds would widen the axes on every frame.

Insert above the `#[cfg(test)]` line of `linkage-sim-rs/src/gui/plot_panel/weights.rs`:

```rust
//! Weight Breakdown plot and braking bands (payload weights, spec Track 2
//! section 3, "Plots").
//!
//! - [`draw_weight_breakdown`]: one line per weight (link self-weights
//!   included) plus the non-gravity remainder and the required total, as
//!   force shares or power shares. A weight's line is coloured per sample
//!   by whether that weight helps, hurts or is neutral there, with the same
//!   `canvas::classification_color` the canvas weight arrows use.
//! - [`draw_braking_bands`]: shades the driver ranges where the load drives
//!   the actuator (`WeightBreakdown::braking`) behind the Actuator Force and
//!   Actuator Power curves.
//!
//! The data assembly ([`braking_bands`], [`band_y_extent`],
//! [`breakdown_lines`]) is pure and unit-tested; the draw functions only map
//! it onto egui_plot items.

use eframe::egui;
use egui_plot::{Plot, Polygon, VLine};

use crate::analysis::gravity_breakdown::{Classification, WeightSource};
use crate::gui::canvas::{classification_color, to_grayscale};
use crate::gui::state::DisplayUnits;
use crate::gui::sweep::{ShareBasis, SweepData, WeightBreakdown};

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, sweep_x_to_display, with_default_x_bounds, x_axis_label_for_sweep,
};

/// Braking band fill: (90, 140, 255) at alpha 50, premultiplied.
const BRAKING_BAND_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(18, 27, 50, 50);
/// Legend entry shared by every braking band (one checkbox hides them all).
const BRAKING_LEGEND_NAME: &str = "Braking";
/// Colour of the non-gravity remainder line.
const OTHER_LOADS_COLOR: egui::Color32 = egui::Color32::from_rgb(100, 200, 255);
/// Colour of the required-total line.
const TOTAL_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 235, 235);
const OTHER_LINE_NAME: &str = "Other loads";
const TOTAL_LINE_NAME: &str = "Total";

/// What a plotted run is coloured by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunColor {
    /// A weight's line: helping / hurting / neutral at these samples.
    Weight(Classification),
    /// The non-gravity remainder.
    Other,
    /// The required total.
    Total,
}

/// A contiguous single-colour piece of a plotted line, as
/// `(sweep x, y)` pairs (x in degrees, or metres in stroke mode).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Run {
    pub(super) color: RunColor,
    pub(super) points: Vec<(f64, f64)>,
}

/// One line of the Weight Breakdown plot: its legend name and its runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BreakdownLine {
    pub(super) name: String,
    pub(super) runs: Vec<Run>,
}

/// Sweep-x ranges `(lo, hi)`, `lo < hi`, where the actuator brakes.
///
/// A run of consecutive braking samples `k0..=k1` covers from halfway to
/// its left neighbour to halfway to its right neighbour (its own sample at
/// the sweep ends), so a lone braking sample shows as one sample step wide
/// and bands of neighbouring runs never touch. A sample with a non-finite
/// x never brakes, and a band next to it ends at its own sample. Only the
/// samples both slices cover are read; zero-width bands (a one-sample
/// sweep) are dropped.
pub(super) fn braking_bands(xs: &[f64], braking: &[bool]) -> Vec<(f64, f64)> {
    let n = xs.len().min(braking.len());
    let in_band = |k: usize| braking[k] && xs[k].is_finite();
    // Band edge at sample `k`, halfway towards `neighbour` when it has an x.
    let edge = |k: usize, neighbour: Option<usize>| match neighbour {
        Some(j) if xs[j].is_finite() => 0.5 * (xs[k] + xs[j]),
        _ => xs[k],
    };
    let mut bands = Vec::new();
    let mut k = 0;
    while k < n {
        if !in_band(k) {
            k += 1;
            continue;
        }
        let first = k;
        while k + 1 < n && in_band(k + 1) {
            k += 1;
        }
        let a = edge(first, first.checked_sub(1));
        let b = edge(k, (k + 1 < n).then_some(k + 1));
        let (lo, hi) = (a.min(b), a.max(b));
        if hi > lo {
            bands.push((lo, hi));
        }
        k += 1;
    }
    bands
}

/// Finite y range `(lo, hi)` over every series: the height of the braking
/// bands, so they cover the curves without stretching the auto-fitted
/// axes. A flat range is widened by 5 % of its magnitude (at least 0.05)
/// so the band stays visible; `None` when no value is finite.
pub(super) fn band_y_extent(series: &[&[f64]]) -> Option<(f64, f64)> {
    let (lo, hi) = series
        .iter()
        .flat_map(|s| s.iter())
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if !lo.is_finite() {
        return None;
    }
    if hi > lo {
        return Some((lo, hi));
    }
    let pad = 0.05 * lo.abs().max(1.0);
    Some((lo - pad, hi + pad))
}

/// Shade the ranges where the actuator brakes (`WeightBreakdown::braking`)
/// behind the curves of an actuator plot. The bands span the finite y
/// range of `extent_series` (every curve and reference line the plot
/// draws) and share one legend entry, "Braking". Nothing is drawn when the
/// sweep has no breakdown or nothing brakes. Call before drawing the
/// curves so they stay on top.
pub(super) fn draw_braking_bands(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
    extent_series: &[&[f64]],
    nathan_mode: bool,
) {
    let Some(breakdown) = &sweep.weight_breakdown else {
        return;
    };
    let bands = braking_bands(&sweep.angles_deg, &breakdown.braking);
    if bands.is_empty() {
        return;
    }
    let Some((y_lo, y_hi)) = band_y_extent(extent_series) else {
        return;
    };
    let fill = if nathan_mode { to_grayscale(BRAKING_BAND_FILL) } else { BRAKING_BAND_FILL };
    for (lo, hi) in bands {
        let (x_lo, x_hi) = (sweep_x_to_display(lo, sweep, units), sweep_x_to_display(hi, sweep, units));
        plot_ui.polygon(
            Polygon::new(BRAKING_LEGEND_NAME, vec![[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])
                .fill_color(fill)
                .stroke(egui::Stroke::new(0.0, fill))
                .allow_hover(false),
        );
    }
}

/// Legend name of a weight's line, unique per source because ids are:
/// `"coupler (link)"` or `"Arm (link b2)"` for a link self-weight (body
/// label, plus the body id when the label differs), `"W1"` or
/// `"Robot (W1)"` for a point mass.
pub(super) fn source_line_name(source: &WeightSource) -> String {
    if source.is_link_self_weight {
        if source.name == source.body_id {
            format!("{} (link)", source.name)
        } else {
            format!("{} (link {})", source.name, source.body_id)
        }
    } else if source.name == source.id {
        source.id.clone()
    } else {
        format!("{} ({})", source.name, source.id)
    }
}

/// Split `(xs[k], ys[k])` into runs of consecutive finite samples with one
/// colour. A non-finite sample ends the run (a gap in the plot). Where the
/// colour changes, the new run starts at the previous sample, so the line
/// stays connected and the joining segment takes the new colour.
fn split_runs(xs: &[f64], ys: &[f64], color_at: impl Fn(usize) -> RunColor) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    let mut previous: Option<(f64, f64)> = None;
    for (k, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if !(x.is_finite() && y.is_finite()) {
            previous = None;
            continue;
        }
        let color = color_at(k);
        match (runs.last_mut(), previous) {
            (Some(run), Some(_)) if run.color == color => run.points.push((x, y)),
            (_, Some(p)) => runs.push(Run { color, points: vec![p, (x, y)] }),
            (_, None) => runs.push(Run { color, points: vec![(x, y)] }),
        }
        previous = Some((x, y));
    }
    runs
}

/// The Weight Breakdown lines over sweep x values `xs`: one per source (in
/// `b.sources` order, coloured by its classification at each sample, the
/// same in both views), then "Other loads" and "Total". `show_power` picks
/// the power shares (W) instead of the force shares (N, or driver-torque
/// shares); force shares are NaN near stroke reversal, which leaves a gap.
pub(super) fn breakdown_lines(xs: &[f64], b: &WeightBreakdown, show_power: bool) -> Vec<BreakdownLine> {
    let per_source = if show_power { &b.power_share } else { &b.force_share };
    let mut lines: Vec<BreakdownLine> = b
        .sources
        .iter()
        .zip(per_source)
        .enumerate()
        .map(|(i, (source, ys))| {
            let classes = b.classifications(i);
            let color_at = |k: usize| RunColor::Weight(classes.get(k).copied().unwrap_or(Classification::Neutral));
            BreakdownLine { name: source_line_name(source), runs: split_runs(xs, ys, color_at) }
        })
        .collect();
    let (other, total) = if show_power { (&b.other_power, &b.total_power) } else { (&b.other_force, &b.total_force) };
    lines.push(BreakdownLine { name: OTHER_LINE_NAME.to_string(), runs: split_runs(xs, other, |_| RunColor::Other) });
    lines.push(BreakdownLine { name: TOTAL_LINE_NAME.to_string(), runs: split_runs(xs, total, |_| RunColor::Total) });
    lines
}

/// Y-axis label of the Weight Breakdown plot: actuator or driver shares,
/// force (N), torque (N*m; N for a linear driver in stroke mode) or power
/// (W).
pub(super) fn breakdown_y_label(basis: ShareBasis, is_stroke: bool, show_power: bool) -> &'static str {
    match (basis, show_power) {
        (ShareBasis::ActuatorForce, false) => "Actuator Force Share (N)",
        (ShareBasis::ActuatorForce, true) => "Actuator Power Share (W)",
        (ShareBasis::DriverTorque, false) if is_stroke => "Driver Force Share (N)",
        (ShareBasis::DriverTorque, false) => "Driver Torque Share (N\u{00b7}m)",
        (ShareBasis::DriverTorque, true) => "Driver Power Share (W)",
    }
}

/// Plot colour and width of a run.
fn run_style(color: RunColor) -> (egui::Color32, f32) {
    match color {
        RunColor::Weight(class) => (classification_color(class), 1.5),
        RunColor::Other => (OTHER_LOADS_COLOR, 1.5),
        RunColor::Total => (TOTAL_COLOR, 2.5),
    }
}

/// Plot each weight's force share (or power share when `show_power`) vs
/// the driver, plus the non-gravity remainder and the required total.
///
/// Returns the clicked X coordinate (display units) if the user clicked.
pub(super) fn draw_weight_breakdown(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
    show_power: bool,
) -> Option<f64> {
    let Some(breakdown) = &sweep.weight_breakdown else {
        ui.label("Weight breakdown not available (no link mass or weight in the mechanism).");
        return None;
    };
    let lines = breakdown_lines(&sweep.angles_deg, breakdown, show_power);

    // Separate plot ids per view, so a zoom set on newtons is not reused
    // on watts.
    let plot_id = if show_power { "weight_breakdown_power_plot" } else { "weight_breakdown_force_plot" };
    let plot = Plot::new(plot_id)
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label(breakdown_y_label(breakdown.basis, sweep.sweep_mode.is_stroke(), show_power))
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, plot_id, sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        for line in &lines {
            for run in &line.runs {
                let (color, width) = run_style(run.color);
                draw_angle_series_with_range(plot_ui, &line.name, color, width, &run.points, sweep, units, nathan_mode);
            }
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}
```

(h) `linkage-sim-rs/src/gui/plot_panel/actuator.rs`. Imports first.

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
    draw_toggle_markers, finite_series, series_colors, with_default_x_bounds,
    x_axis_label_for_sweep,
};
```

Replace with:

```rust
    draw_toggle_markers, finite_series, series_colors, sweep_x_to_display, with_default_x_bounds,
    x_axis_label_for_sweep,
};
use super::weights::draw_braking_bands;
```

In `draw_actuator_force`, draw the bands first so the curves stay on top. They span every curve and reference line drawn below (statics, the inertia overlay when present, and the rated lines when a rated force is set).

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
    let plot = with_default_x_bounds(plot, "actuator_force_plot", sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        draw_angle_series_with_range(
```

Replace with:

```rust
    let plot = with_default_x_bounds(plot, "actuator_force_plot", sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Braking bands first so the curves draw on top; they span every
        // curve and rated line drawn below.
        let rated_lines = [actuator_rated_force, -actuator_rated_force];
        let mut extent: Vec<&[f64]> = vec![forces.as_slice()];
        if let Some(id_forces) = &sweep.actuator_forces_id {
            extent.push(id_forces.as_slice());
        }
        if actuator_rated_force > 0.0 {
            extent.push(&rated_lines);
        }
        draw_braking_bands(plot_ui, sweep, units, &extent, nathan_mode);

        draw_angle_series_with_range(
```

The inertia overlay and the safety-factor overlay use `sweep_x_to_display`.

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
            let is_stroke = sweep.sweep_mode.is_stroke();
            let id_points: PlotPoints = id_pairs
                .iter()
                .map(|&(x, f)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, f]
                })
                .collect();
```

Replace with:

```rust
            let id_points: PlotPoints = id_pairs
                .iter()
                .map(|&(x, f)| [sweep_x_to_display(x, sweep, units), f])
                .collect();
```

The rated lines get the extension/retraction wording: `Rated (push)` and `Rated (pull)`.

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
            // Horizontal rated force line (positive / tension).
            let rated_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(80, 200, 80))
            } else {
                egui::Color32::from_rgb(80, 200, 80)
            };
            plot_ui.hline(
                HLine::new("Rated Force", rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );

            // Horizontal rated force line (negative / compression).
            plot_ui.hline(
                HLine::new("Rated (compression)", -rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );
```

Replace with:

```rust
            // Horizontal rated force line (positive = extension, push).
            let rated_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(80, 200, 80))
            } else {
                egui::Color32::from_rgb(80, 200, 80)
            };
            plot_ui.hline(
                HLine::new("Rated (push)", rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );

            // Horizontal rated force line (negative = retraction, pull).
            plot_ui.hline(
                HLine::new("Rated (pull)", -rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );
```

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
            let is_stroke = sweep.sweep_mode.is_stroke();
            let to_display = |x_deg: f64| -> f64 {
                if is_stroke { x_deg * 1000.0 } else { units.angle(x_deg.to_radians()) }
            };
```

Replace with:

```rust
            let to_display = |x_deg: f64| sweep_x_to_display(x_deg, sweep, units);
```

In `draw_actuator_power`, the bands span the statics power and, when present, the inertia power.

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
    let plot = with_default_x_bounds(plot, "actuator_power_plot", sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Statics-based power (solid line).
```

Replace with:

```rust
    let plot = with_default_x_bounds(plot, "actuator_power_plot", sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Braking bands first so the curves draw on top.
        let mut extent: Vec<&[f64]> = vec![power.as_slice()];
        if let Some(id_power) = &sweep.actuator_power_id {
            extent.push(id_power.as_slice());
        }
        draw_braking_bands(plot_ui, sweep, units, &extent, nathan_mode);

        // Statics-based power (solid line).
```

Find in `linkage-sim-rs/src/gui/plot_panel/actuator.rs`:

```rust
            let is_stroke = sweep.sweep_mode.is_stroke();
            let id_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(id_power.iter())
                .filter(|&(_, &p)| p.is_finite())
                .map(|(&x, &p)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, p]
                })
                .collect();
```

Replace with:

```rust
            let id_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(id_power.iter())
                .filter(|&(_, &p)| p.is_finite())
                .map(|(&x, &p)| [sweep_x_to_display(x, sweep, units), p])
                .collect();
```

(i) `linkage-sim-rs/src/gui/plot_panel/dynamics.rs`: the driver-torque statics overlay is the last inline copy. Leave `is_stroke` in `draw_inverse_dynamics`; the statics label below the overlay still uses it.

Find in `linkage-sim-rs/src/gui/plot_panel/dynamics.rs`:

```rust
    series_colors, with_default_x_bounds, x_axis_label_for_sweep,
};
```

Replace with:

```rust
    series_colors, sweep_x_to_display, with_default_x_bounds, x_axis_label_for_sweep,
};
```

Find in `linkage-sim-rs/src/gui/plot_panel/dynamics.rs`:

```rust
                .map(|(&x, &t)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, t]
                })
```

Replace with:

```rust
                .map(|(&x, &t)| [sweep_x_to_display(x, sweep, units), t])
```

(j) Docs, in the same commit. Every new behaviour above is documented where the next reader looks for it. Keep the text as written: `02-system.yaml` and `03-structure.yaml` are YAML, so the new lines contain no `: ` inside a plain scalar (that would add a parse error).

`docs/ai/02-system.yaml`: plot-tab count in `what_it_is` and `current_statuses`.

Find in `docs/ai/02-system.yaml`:

```yaml
  (native + WASM) with 30 sample mechanisms, 14 plot tabs, DXF import,
```

Replace with:

```yaml
  (native + WASM) with 30 sample mechanisms, 15 plot tabs, DXF import,
```

Find in `docs/ai/02-system.yaml`:

```yaml
  gui: Phase 5 complete; 14 plot tabs; 30 sample mechanisms
```

Replace with:

```yaml
  gui: Phase 5 complete; 15 plot tabs (Weight Breakdown added); 30 sample mechanisms
```

The new invariant goes directly after `point_mass_loader_validation`. It records the sum and sign conventions, that the bands and the Total line follow the plotted curves in both stored-force and sizing mode (with the two tests that pin it), and the rule that every plot converts sweep x through `sweep_x_to_display`.

Find in `docs/ai/02-system.yaml`:

```yaml
    io::from_json::apply_point_masses is the only place weights reach the
    physics (the loader and sync_live_mass_props both call it), so a
    no-rebuild mass edit skips exactly what a rebuild skips.
```

Replace with:

```yaml
    io::from_json::apply_point_masses is the only place weights reach the
    physics (the loader and sync_live_mass_props both call it), so a
    no-rebuild mass edit skips exactly what a rebuild skips.
  - weight_breakdown_sum_and_sign_conventions — WeightBreakdownBuilder::finish
    builds other_* = total_* - sum(shares), so wherever the shares are
    finite sum_i force_share[i] + other_force == total_force and
    sum_i power_share[i] + other_power == total_power (other ~ 0 when
    gravity is the only load; tested). Signs, one convention everywhere -
    actuator force positive = extension/push (forces/elements/evaluation.rs;
    tooltips and the "Rated (push)/(pull)" lines say so, never "tension");
    P_g,i = m_i g.v_i > 0 = helping (coming down); power_share = -P_g,i
    (< 0 = gives power back); force_share = -P_g,i / dL/dt also flips with
    the stroke direction, so plots colour weight lines by classification,
    never by the share's sign; braking = required power
    < -BRAKE_TOL_REL * max|P|. Since BL-026 the plotted Actuator Force and
    Actuator Power are the required values in sizing and stored-force mode
    alike, so the braking bands and the Weight Breakdown Total line follow
    the plotted curves in both modes (plot_panel::weights tests
    braking_bands_cover_exactly_the_negative_actuator_power_samples,
    total_line_is_the_plotted_required_actuator_force_and_power).
    canvas::classification_color (green helping, red hurting, gray neutral)
    is the single palette for canvas weight arrows and Weight Breakdown
    lines. Braking bands read WeightBreakdown::braking only (none when the
    sweep has no breakdown). Every plot maps sweep x to display x through
    plot_panel::sweep_x_to_display (mm in stroke mode, display angle unit
    otherwise); never inline the conversion.
```

The egui_plot lesson goes at the end of `lessons_learned`.

Find in `docs/ai/02-system.yaml`:

```yaml
    .clamp_existing_to_range(false) on committed fields (weight mass field).
```

Replace with:

```yaml
    .clamp_existing_to_range(false) on committed fields (weight mass field).
  - egui_plot fits its auto bounds to every item plus a 5 % margin
    (margin_fraction), so a shaded band sized from plot_ui.plot_bounds()
    would widen the axes on every frame. Size bands from the data's finite
    y range instead (plot_panel/weights.rs band_y_extent).
```

`docs/ai/03-structure.yaml`: the palette, the plot module lines, `classifications()`, and the add-a-plot-tab recipe.

Find in `docs/ai/03-structure.yaml`:

```yaml
      colors: canvas/colors.rs
```

Replace with:

```yaml
      colors: canvas/colors.rs (includes classification_color, re-exported as canvas::classification_color — the one helping/hurting/neutral palette for weight arrows and Weight Breakdown lines)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
      mod: plot_panel/mod.rs (PlotTab enum, dispatcher, shared helpers)
```

Replace with:

```yaml
      mod: plot_panel/mod.rs (PlotTab enum incl. WeightBreakdown, dispatcher, shared helpers incl. sweep_x_to_display - the one sweep-x to display-x conversion; Weight Breakdown force/power toggle bound to AppState::weight_breakdown_show_power)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
      actuator: plot_panel/actuator.rs (force, speed, power; force plot draws every finite sample — no outlier filter, BL-010)
```

Replace with:

```yaml
      actuator: plot_panel/actuator.rs (force, speed, power; force plot draws every finite sample — no outlier filter, BL-010; force and power plots shade braking bands via weights::draw_braking_bands)
      weights: plot_panel/weights.rs (payload weights - Weight Breakdown tab draw_weight_breakdown, one line per weight source + Other loads + Total, force or power share, runs coloured per sample by classification; braking bands braking_bands / band_y_extent / draw_braking_bands; pure assembly breakdown_lines, source_line_name, breakdown_y_label)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
      weights: sweep/weights.rs (payload weights - WeightBreakdown {sources, basis, gravity_power, force_share, power_share, other_force, other_power, total_force, total_power, braking} + classification(); ShareBasis {ActuatorForce, DriverTorque}; WeightBreakdownBuilder fed per sample by compute_sweep_data_with_weights; required_totals. AppState::compute_sweep passes gravity_breakdown::weight_sources(blueprint); plain compute_sweep_data = no breakdown)
```

Replace with:

```yaml
      weights: sweep/weights.rs (payload weights - WeightBreakdown {sources, basis, gravity_power, force_share, power_share, other_force, other_power, total_force, total_power, braking} + classification() and classifications() (per-sample class of one source, band computed once; the plot colours lines with it); ShareBasis {ActuatorForce, DriverTorque}; WeightBreakdownBuilder fed per sample by compute_sweep_data_with_weights; required_totals. AppState::compute_sweep passes gravity_breakdown::weight_sources(blueprint); plain compute_sweep_data = no breakdown)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - src/gui/plot_panel/{mechanics,dynamics,actuator,coupler}.rs (add draw_ fn)
```

Replace with:

```yaml
    - src/gui/plot_panel/{mechanics,dynamics,actuator,coupler,weights}.rs (add draw_ fn)
```

`docs/ai/05-update-tracker.md`: a new entry at the top, above the Task 6 entry.

Find in `docs/ai/05-update-tracker.md`:

```markdown
## 2026-09-29 — Payload weights Task 6: drag and drop weights
```

Replace with:

```markdown
## 2026-09-29 — Payload weights Task 7: braking bands and the Weight Breakdown plot
- `gui/plot_panel/weights.rs` (new): `PlotTab::WeightBreakdown` ("Weight
  Breakdown" tab, enabled when `SweepData::weight_breakdown` is `Some`): one
  line per weight source (legend `"coupler (link)"`, `"Robot (W1)"`) plus
  "Other loads" and "Total", force share or power share
  (`AppState::weight_breakdown_show_power`, not persisted). Each weight's
  line is split into runs coloured per sample by
  `WeightBreakdown::classifications` (new: band computed once per weight)
  through `canvas::classification_color` (new, `canvas/colors.rs`: green
  helping / red hurting / gray neutral, distinct in Nathan Mode grayscale),
  the palette the canvas weight arrows will share. Force shares leave gaps
  near stroke reversal and at failed samples.
- Braking bands: `braking_bands(xs, braking)` (runs of braking samples,
  midpoint to midpoint, clipped at the sweep ends, zero-width dropped) and
  `band_y_extent` (data y range, never `plot_bounds()`), drawn by
  `draw_braking_bands` as "Braking" polygons behind the Actuator Force and
  Actuator Power curves. Source: `WeightBreakdown::braking` only, i.e. the
  required power. Since BL-026 the plotted actuator force and power are the
  required values in stored-force mode too, so the bands and the Total line
  follow the plotted curves in both modes; no BL-026 caveat remains.
- `plot_panel::sweep_x_to_display` (new, DRY): the one sweep-x to display-x
  conversion (m -> mm in stroke mode, degrees -> display angle unit). It
  replaces the inline copies at 9 sites: `compute_default_x_bounds`,
  `draw_toggle_markers`, `draw_range_boundary_markers`,
  `draw_angle_series_with_range`, the actuator force / power "With Inertia"
  overlays and the safety-factor overlay, the driver-torque statics
  overlay, and `draw_braking_bands`. `current_driver_display` is not a
  sweep x value (radians plus the display offset) and keeps its own branch.
- Sign wording: the Actuator Force tab tip and the LinearActuator force
  field tip now say positive = extension (push), negative = retraction
  (pull) (were "positive = tension"); the rated-force lines are "Rated
  (push)" / "Rated (pull)" (was "Rated (compression)" for the pull side).
  Both tips are constants with tests.
- Tests: `gui::plot_panel::weights::tests` (band edges, ends, NaN x,
  descending x, length mismatch, y extent, legend names, run splitting in
  both views, y labels; bands = negative plotted actuator power on the
  parallelogram in stored-force AND sizing mode, which uses
  `test_support::set_actuator_stored_force` instead of an inline loop; the
  Total line equals the plotted actuator force and power in stored-force
  mode), `gui::plot_panel::tests` (tooltip, `sweep_x_to_display`, headless
  render of the new tab and the banded tabs incl. driver basis and no
  breakdown), `gui::canvas::colors::tests`,
  `gui::property_panel::force_editor::tests`,
  `gui::sweep::weights::tests::classifications_match_classification_at_every_sample`.
  Mutation check: dropping `stored_force * rate` from `required_totals`'
  total power fails the stored-force case of the band test (91 deg) and
  the Total line test.
- Docs: 02-system `weight_breakdown_sum_and_sign_conventions`, egui_plot
  auto-bounds lesson, 15 plot tabs; 03-structure plot_panel/weights.rs,
  `sweep_x_to_display`, colors palette, `classifications()`;
  `docs/architecture/ENGINEERING_OUTPUTS.md` per-weight breakdown and
  braking section; FEATURES/README actuator sizing bullets. The stale
  "10 plot tabs" lines in README/FEATURES and the hands-on checklist are
  Task 10.

## 2026-09-29 — Payload weights Task 6: drag and drop weights
```

`docs/architecture/ENGINEERING_OUTPUTS.md`: the per-weight breakdown and braking section goes after the Driver Effort Breakdown section. It contains its own fenced block, so the block below uses a longer fence.

Find in `docs/architecture/ENGINEERING_OUTPUTS.md`:

```markdown
Plotted as a stacked chart vs. input angle, this shows the engineer exactly where the motor effort comes from and which sources can be reduced (e.g., adding a counterbalance spring to cancel gravity contribution).
```

Add after it:

````markdown
### Per-Weight Gravity Breakdown (Weight Breakdown tab)

The gravity term is split further, one entry per weight: each link's own mass (base mass at its base CG, named after the link) and each point mass placed on a link. Gravity is linear in mass, so the split is exact. For weight *i* at each sweep sample (`analysis::gravity_breakdown`, `gui::sweep::weights`):

```text
P_g,i = m_i * (g . v_i)          gravity power (W); > 0 = weight coming down = helping
power share  = -P_g,i            actuator power spent on the weight (W); < 0 = gives power back
force share  = -P_g,i / (dL/dt)  actuator force share (N); NaN where |dL/dt| < 1 % of its sweep max
other        = total - sum(shares)   springs, force zones, external loads, end stops
```

Without a linear actuator the shares are driver-torque shares (`-P_g,i / omega`, N·m; N for a linear driver). The total is the **required** actuator force and power, the values the Actuator Force and Actuator Power plots show in sizing and stored-force mode alike (BL-026), so `sum(shares) + other = total` at every sample where the shares are finite.

Sign conventions: actuator force positive = extension (the actuator pushes its ends apart), negative = retraction (it pulls them together). A weight is *helping* where `P_g,i` is above 1 % of its sweep maximum, *hurting* below minus that band, *neutral* in between (moving sideways). Because the force share also changes sign with the stroke direction, the plot colours each weight's line by helping (green) / hurting (red) / neutral (gray) at each sample, the same palette as the canvas weight arrows (`canvas::classification_color`), rather than relying on the sign.

**Braking:** the actuator brakes (the load drives it) where the required actuator power is below `-1e-6 * max|P|` over the sweep. The Actuator Force and Actuator Power plots shade those driver ranges; a band runs halfway to the neighbouring samples on either side. In the quasi-static model the force to hold a load is the same up and down; gravity helping shows up as negative power (braking), not as a smaller force.
````

`docs/FEATURES.md`: the actuator-force bullet keeps main's BL-026 wording and gains the sign convention; the two new bullets follow it.

Find in `docs/FEATURES.md`:

```markdown
- **Actuator force plot** -- required actuator force (N) vs crank angle via power balance (F = T × omega / dL_dt + F_stored); the actuator's stored force is added back, so the curve is the same whether the stored force is 0 (sizing mode) or set
```

Replace with:

```markdown
- **Actuator force plot** -- required actuator force (N) vs crank angle via power balance (F = T × omega / dL_dt + F_stored); the actuator's stored force is added back, so the curve is the same whether the stored force is 0 (sizing mode) or set; positive = extension (push), negative = retraction (pull)
- **Braking bands** -- the Actuator Force and Actuator Power plots shade the crank ranges where the load drives the actuator (negative required power)
- **Weight Breakdown tab** -- one line per link self-weight and per placed weight, plus Other loads and Total, as force share (N) or power share (W); each weight's line is green where it helps (coming down), red where the actuator lifts it, gray where it moves sideways
```

`README.md`: the actuator sizing bullet. (The stale "10 plot tabs" line in `README.md` and `docs/FEATURES.md` is Task 10.)

Find in `README.md`:

```markdown
- **Actuator sizing**: actuator force plot (statics + inverse dynamics curves), stroke display in Health Report, force zone overlap diagnostics
```

Replace with:

```markdown
- **Actuator sizing**: actuator force plot (statics + inverse dynamics curves), braking bands on the actuator force and power plots, per-weight Weight Breakdown plot (which weight helps or hurts, in force and in power), stroke display in Health Report, force zone overlap diagnostics
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the new tests first, from `linkage-sim-rs/`:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib plot_panel
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib canvas::colors
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib classifications_match_classification_at_every_sample
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib actuator_force_field_tip_states_the_extension_sign_convention
```

Expected: each command ends with `test result: ok.`; the counts are `plot_panel` 33 passed (the 4 new tests in `plot_panel/mod.rs` and the 14 in `plot_panel/weights.rs`, plus the existing plot_panel tests), `canvas::colors` 2 passed, and 1 passed each for the last two filters.

Then the whole lib suite:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: `test result: ok. 869 passed; 0 failed`. It was 847 passed at the end of Task 6; the 22 new tests are 14 in `plot_panel/weights.rs`, 4 in `plot_panel/mod.rs`, 2 in `canvas/colors.rs`, 1 in `sweep/weights.rs` and 1 in `property_panel/force_editor.rs`.

Mutation check (do not commit it). BL-026 is what makes the two mode-sweeping tests meaningful, so prove they bite. In `linkage-sim-rs/src/gui/sweep/weights.rs`, change the `ShareBasis::ActuatorForce` arm of `required_totals` from `(actuator_force, driver_torque * omega + stored_force * rate)` to `(actuator_force, driver_torque * omega)` (drop the `stored_force * rate` term). Run:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib braking_bands_cover_exactly_the_negative_actuator_power_samples -- total_line_is_the_plotted_required_actuator_force_and_power
```

(`cargo test` takes one filter before the `--`; libtest takes the second one after it, so both tests run.)

Expected: `test result: FAILED. 0 passed; 2 failed`. `total_line_is_the_plotted_required_actuator_force_and_power` fails in the power view with a message like `power view true, at 0 deg: Total 234.16..., plotted 422.66...` (the force view is unaffected, `total_force` does not use the dropped term). `braking_bands_cover_exactly_the_negative_actuator_power_samples` fails in its stored-force case with `stored force, at 91 deg: P = -111.87... W` (left: false, right: true): the plotted power is negative there but the band, computed from the mutated total power, is not shaded. The sizing-mode case of the band test is unaffected because the stored force is 0 there.

Undo the one-line edit by hand. Do not use `git checkout` on this file: it also holds the uncommitted `classifications` change from Step 3(c). Then re-run `cargo test --lib` and confirm 869 passed again.

Docs sanity, from the repo root:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload && python -c "import yaml,sys; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('03-structure OK')"
cd /c/Users/Cole/source/repos/linkage_simulation-payload && python -c "import yaml,sys; yaml.safe_load(open('docs/ai/02-system.yaml', encoding='utf-8'))"
cd /c/Users/Cole/source/repos/linkage_simulation-payload && git diff -U0 -- docs/ai/02-system.yaml | grep '^+' | grep -v '^+++' | grep ': '
```

Expected: the first command prints `03-structure OK`. The second raises the `ScannerError` `while scanning a simple key` at `docs/ai/02-system.yaml`, line 14, column 5, followed by `could not find expected ':'` at line 15. That error is already on main at the same line and is outside this task; what matters is that the position did not move and that the third command prints only one line, `+  gui: Phase 5 complete; 15 plot tabs (Weight Breakdown added); 30 sample mechanisms`, which is a normal `key: value` line. Any other added line containing `: ` would be a new parse error in a plain scalar; rewrite it without the colon.

Then the gate and the clippy check. `scripts/gate.sh` is the repo's gate (build, tests, clippy, WASM check):

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected: the last line is `GATE PASS`. The clippy warning counts (the `generated N warnings` lines for `linkage-sim-rs` (lib) and `linkage-sim-rs` (lib test) in the `cargo clippy --all-targets` output) are unchanged from before the task; on the verified run they were 282 and 298, and the rustc lib warning count stayed at 34. The exact numbers depend on the toolchain, so compare against a run of the same command before Step 1 rather than against these figures. The test run rewrites the PNGs in `docs/chebyshev_lambda/`; restore them before committing:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/gui/plot_panel/weights.rs linkage-sim-rs/src/gui/plot_panel/mod.rs linkage-sim-rs/src/gui/plot_panel/actuator.rs linkage-sim-rs/src/gui/plot_panel/dynamics.rs linkage-sim-rs/src/gui/canvas/colors.rs linkage-sim-rs/src/gui/canvas/mod.rs linkage-sim-rs/src/gui/sweep/weights.rs linkage-sim-rs/src/gui/state/mod.rs linkage-sim-rs/src/gui/property_panel/force_editor.rs docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md docs/architecture/ENGINEERING_OUTPUTS.md docs/FEATURES.md README.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 7: braking bands and the Weight Breakdown plot

New plot_panel/weights.rs: PlotTab::WeightBreakdown draws one line per
weight source plus Other loads and Total, as force share or power share
(AppState::weight_breakdown_show_power). Each weight's line is split into
runs coloured per sample by WeightBreakdown::classifications (new, band
computed once per weight) through canvas::classification_color (new
shared green/red/gray palette, distinct in Nathan Mode grayscale).
Braking bands (braking_bands midpoint to midpoint, band_y_extent from the
data) shade the Actuator Force and Actuator Power plots from
WeightBreakdown::braking, the required power. Since BL-026 the plotted
actuator force and power are the required values in stored-force mode
too, so the bands and the Total line follow the curves in both modes.
The actuator force tab and field tooltips now say positive = extension;
the rated lines are Rated (push) / Rated (pull).

DRY: plot_panel::sweep_x_to_display is now the one sweep-x to display-x
conversion (mm in stroke mode, display angle unit otherwise), replacing
the inline copies at 9 sites incl. draw_braking_bands. The band test uses
test_support::set_actuator_stored_force instead of an inline loop and now
covers stored-force and sizing mode; a new test checks the Total line is
the plotted required actuator force and power.

Docs: 02-system weight_breakdown_sum_and_sign_conventions invariant,
egui_plot auto-bounds lesson, 15 plot tabs; 03-structure plot module,
tab, palette and sweep_x_to_display; tracker entry; ENGINEERING_OUTPUTS
per-weight breakdown and braking section; FEATURES/README bullets.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 8: Placement and the weight editor

Places weights from the toolbar and edits them in the property panel. The `+ Mass` tool gets a toolbar mass field that starts at the last mass used; its placement click snaps to the grid, names the weight from `io::next_point_mass_id`, selects it and is not also a selection click. The property panel gains `property_panel/weight_editor.rs`: a selected-weight editor (name, mass, owning link, body-local position) and a link editor Weights section that is always shown for a moving link and has an **Add weight** button. Every field commits once per edit (`committed_number`), one undo step each. The task also moves the display-name helpers into `analysis::gravity_breakdown` (the Weight Breakdown legend uses them) and grows `gui/test_support.rs` so the new tests and the older idle-frame helpers share one headless-frame helper.

**Files:**
- Create: `linkage-sim-rs/src/gui/property_panel/weight_editor.rs` (selected-weight editor, link editor Weights section, `committed_number`)
- Create: `linkage-sim-rs/src/gui/canvas/rendering/weights.rs` (`place_mass_hint`)
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (module doc; `typed`, `central_panel_frame`, `visit_shapes`, `drawn_texts`, `drew_text`, `text_rect`)
- Modify: `linkage-sim-rs/src/gui/mod.rs` (`+ Mass` hover text and `draw_place_mass_field` call in the toolbar; new `draw_place_mass_field` under `// ── Delete shortcut ──`; tests module)
- Modify: `linkage-sim-rs/src/gui/state/display_units.rs` (append `DISPLAY_DECIMALS`, `format_decimal`, `format_mass_kg`, tests)
- Modify: `linkage-sim-rs/src/gui/state/mod.rs` (`display_units` re-export)
- Modify: `linkage-sim-rs/src/analysis/gravity_breakdown.rs` (`PointMassJson` import; `display_name` becomes `pub`; new `name_with_id`, `point_mass_title`; tests)
- Modify: `linkage-sim-rs/src/gui/plot_panel/weights.rs` (`source_line_name` uses `name_with_id`)
- Modify: `linkage-sim-rs/src/gui/plot_panel/mod.rs` (test helper `plot_panel_frame` uses `central_panel_frame`)
- Modify: `linkage-sim-rs/src/gui/property_panel/mod.rs` (`weight_editor` module and `weight_mass_drag_value` re-export; selected-weight section after Mechanism Health; the `Point Masses` block becomes the Weights section; test helper `one_idle_frame`)
- Modify: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (`SetPointMassLabel`, `AddPointMass`, `RemovePointMass` clears the selection; tests)
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (re-export `WEIGHT_COLOR`; `weight_clicks` tests)
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs` (Place Mass call and click-selection gate in `handle_interaction`; `handle_weight_drag` uses the new `snapped_world`; `handle_place_mass` rewritten)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs` (`weights` submodule; Place Mass hint in `render_overlays`)
- Modify docs: `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`, `docs/FEATURES.md`, `docs/architecture/ARCHITECTURE.md`. The spec's hands-on checklist wording for the Weights section is not written here; Task 10 owns the checklist.

Edit blocks below: "Find in `<path>`:" quotes existing text exactly as it is before this task; "Replace with:" swaps that text for the block; "Add after it:" puts the block after the quoted text with exactly one blank line between them. "Create `<path>`:" is a new file. "Insert at the very top of `<path>`" prepends the block, plus one blank line, to a file that Step 1 created. Edits are anchored by quoted text, not by line numbers.

**Interfaces:**
- Consumes (Tasks 1, 2, 5, 6):
  - `AppState::add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) -> Option<String>` (sets `last_point_mass_kg`; `None` and no undo entry when the loader would skip the weight)
  - `AppState::find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson>`
  - `AppState::move_point_mass(&mut self, body_id: &str, weight_id: &str, target_body: &str, local_pos: [f64; 2]) -> bool`
  - `AppState::set_point_mass_mass(&mut self, body_id: &str, weight_id: &str, mass: f64) -> bool`
  - `AppState::set_point_mass_label(&mut self, body_id: &str, weight_id: &str, label: Option<String>) -> bool` (trims; blank clears)
  - `AppState::remove_point_mass_by_id(&mut self, body_id: &str, weight_id: &str) -> bool`
  - `AppState::world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2]`; `AppState::nc(&self, c: egui::Color32) -> egui::Color32`
  - `AppState` fields `pub last_point_mass_kg: f64` (default 1.0), `pub place_mass_body: Option<String>`, `pub link_editor_body: Option<String>`, `pub multi_selected: Vec<SelectedEntity>`, `pub status_message: Option<String>`, `pub status_message_time: f64`; `state.undo_history.undo_count() -> usize`
  - `SelectedEntity::Weight { body_id: String, weight_id: String }`; `crate::io::{next_point_mass_id, BodyJson, PointMassJson}` (`pub fn next_point_mass_id(bodies: &HashMap<String, BodyJson>) -> String`; `PointMassJson { id: String, label: Option<String>, mass: f64, local_pos: [f64; 2] }`)
  - `GridSettings::snap_point(&self, x: f64, y: f64) -> (f64, f64)`; `DisplayUnits::{length(&self, meters: f64) -> f64, length_to_si(&self, display: f64) -> f64, length_suffix(&self) -> &'static str}`
  - `canvas::colors::{WEIGHT_COLOR, WEIGHT_RADIUS, WEIGHT_HIT_RADIUS, LINK_PICK_RADIUS}` (`interaction.rs` already has them through `use super::colors::*;`)
  - `PendingPropertyEdit::{SetPointMassMass { body_id, weight_id, mass }, SetPointMassPosition { body_id, weight_id, local_pos }, RemovePointMass { body_id, weight_id }, ReassignPointMass { body_id, weight_id }, RepositionPointMass { body_id, weight_id }}`; `pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>)`
  - `gui::test_support::{sorted_link_ids(state: &AppState) -> Vec<String>, primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event, primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event, key_press(key: egui::Key) -> egui::Event, set_actuator_stored_force(state: &mut AppState, force: f64)}` (cfg(test))
  - `canvas::tests::weight_clicks` helpers `setup() -> (egui::Context, AppState, String, String)`, `click(ctx, state, pos)`, `frame_with(ctx, state, events, modifiers) -> egui::FullOutput`, `drop_world(state, pos) -> [f64; 2]`, `local_under(state, body, pos) -> [f64; 2]`, `screen_of(state, world) -> Pos2`, `weight(body, id) -> SelectedEntity`, `assert_close(what, got, want)`
- Produces:
  - `pub const DISPLAY_DECIMALS: usize = 6;`, `pub fn format_decimal(value: f64) -> String`, `pub fn format_mass_kg(kg: f64) -> String` (`gui::state::display_units`; `format_decimal` and `format_mass_kg` re-exported from `gui::state`)
  - `pub fn display_name(label: Option<&String>, fallback: &str) -> String` (was private), `pub fn name_with_id(name: &str, id: &str) -> String`, `pub fn point_mass_title(pm: &PointMassJson) -> String` (`analysis::gravity_breakdown`)
  - `pub(crate) const WEIGHT_MASS_RANGE_KG: RangeInclusive<f64> = 0.001..=1000.0;` and `pub(crate) fn weight_mass_drag_value(mass: &mut f64) -> egui::DragValue<'_>` (`property_panel::weight_editor`, the latter re-exported as `property_panel::weight_mass_drag_value`)
  - `pub(super) fn draw_selected_weight(ui: &mut egui::Ui, state: &AppState, body_id: &str, weight_id: &str, pending: &mut Option<PendingPropertyEdit>)` and `pub(super) fn draw_link_weights(ui: &mut egui::Ui, state: &AppState, body_id: &str, body: &BodyJson, pending: &mut Option<PendingPropertyEdit>)` (`property_panel::weight_editor`)
  - `PendingPropertyEdit::SetPointMassLabel { body_id: String, weight_id: String, label: Option<String> }` and `PendingPropertyEdit::AddPointMass { body_id: String, local_pos: [f64; 2] }`; `RemovePointMass` now also clears a selection of the removed weight
  - `fn draw_place_mass_field(ui: &mut egui::Ui, state: &mut AppState)` (`gui/mod.rs`, private)
  - `fn handle_place_mass(ui: &mut egui::Ui, painter: &egui::Painter, response: &egui::Response, state: &mut AppState, body_segments: &[BodySegment]) -> bool` (true = this frame's click placed a weight) and `fn snapped_world(state: &AppState, screen: Pos2) -> [f64; 2]` (`canvas/interaction.rs`, private)
  - `pub(super) fn place_mass_hint(state: &AppState, body_id: &str) -> String` (`canvas/rendering/weights.rs`); `pub use colors::WEIGHT_COLOR` from `gui::canvas`
  - `gui::test_support::{typed(text: &str) -> egui::Event, central_panel_frame(ctx: &egui::Context, events: Vec<egui::Event>, draw: impl FnMut(&mut egui::Ui)) -> egui::FullOutput, drawn_texts(output: &egui::FullOutput) -> Vec<String>, drew_text(output: &egui::FullOutput, needle: &str) -> bool, text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect>}` (cfg(test))

- [ ] **Step 1: Write the failing tests**

Order matters below: the shared test helpers come first (they compile on their own), then the tests, and the two `mod` registrations that make the compiler see the new test modules.

1a. `linkage-sim-rs/src/gui/test_support.rs`: the module doc, then the new helpers. `central_panel_frame` is the one headless frame that the new tests, `property_panel::tests::one_idle_frame` (Task 2) and `plot_panel::tests::plot_panel_frame` (Task 7) all use instead of each writing its own `ctx.run`; `drawn_texts`, `drew_text` and `text_rect` inspect what a frame painted.

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
//! Helpers shared by the tests of the GUI modules.
```

Replace with:

```rust
//! Helpers shared by the tests of the GUI modules (fixtures, input events,
//! and inspection of what a headless egui frame painted).
```

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
/// A key press event with no modifiers.
pub(crate) fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
}
```

Add after it:

```rust
/// Text typed into the focused widget.
pub(crate) fn typed(text: &str) -> egui::Event {
    egui::Event::Text(text.to_string())
}

/// One headless frame of `draw` inside a central panel, with `events` as
/// the frame's input. Returns what egui painted.
pub(crate) fn central_panel_frame(
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput { events, ..Default::default() };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
}

/// Call `visit` on every shape egui painted in a frame, nested shapes
/// included, in paint order.
fn visit_shapes(output: &egui::FullOutput, mut visit: impl FnMut(&egui::Shape)) {
    fn walk(shape: &egui::Shape, visit: &mut impl FnMut(&egui::Shape)) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, visit)),
            other => visit(other),
        }
    }
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut visit);
    }
}

/// Every text egui drew in a frame (widgets, painter text, tooltips), in
/// paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    let mut texts = Vec::new();
    visit_shapes(output, |shape| {
        if let egui::Shape::Text(text) = shape {
            texts.push(text.galley.text().to_string());
        }
    });
    texts
}

/// Whether egui drew the text `needle` (exactly) in a frame.
pub(crate) fn drew_text(output: &egui::FullOutput, needle: &str) -> bool {
    drawn_texts(output).iter().any(|t| t == needle)
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    let mut found = None;
    visit_shapes(output, |shape| {
        match shape {
            egui::Shape::Text(text) if found.is_none() && text.galley.text() == needle => {
                found = Some(text.galley.rect.translate(text.pos.to_vec2()));
            }
            _ => {}
        }
    });
    found
}
```

1b. `linkage-sim-rs/src/gui/state/display_units.rs`: tests of the number formatting the weight fields show and read back. Add them after the `impl DisplayUnits` block (Step 3 adds the functions at the same place, above these tests):

Find in `linkage-sim-rs/src/gui/state/display_units.rs`:

```rust
    /// X/Y axis label for length plots.
    pub fn length_axis_label(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => "m",
            LengthUnit::Millimeters => "mm",
        }
    }
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_decimal_keeps_up_to_six_decimals_without_trailing_zeros() {
        assert_eq!(format_decimal(2.0), "2");
        assert_eq!(format_decimal(0.5), "0.5");
        assert_eq!(format_decimal(50.25), "50.25");
        assert_eq!(format_decimal(1.23456789), "1.234568");
        assert_eq!(format_decimal(30.0004), "30.0004");
        assert_eq!(format_decimal(-12.5), "-12.5");
        assert_eq!(format_decimal(1500.0), "1500");
    }

    #[test]
    fn format_decimal_never_prints_negative_zero() {
        assert_eq!(format_decimal(0.0), "0");
        assert_eq!(format_decimal(-0.0), "0");
        assert_eq!(format_decimal(-0.0000001), "0");
    }

    #[test]
    fn format_decimal_passes_non_finite_values_through() {
        assert_eq!(format_decimal(f64::NAN), "NaN");
        assert_eq!(format_decimal(f64::INFINITY), "inf");
    }

    /// What the text reads back as is what `format_decimal` rounds to.
    #[test]
    fn format_decimal_round_trips_through_parse() {
        for v in [2.0, 0.5, 0.031234567, 30.0000004, 999.9999996] {
            let shown: f64 = format_decimal(v).parse().unwrap();
            assert!((shown - v).abs() <= 5e-7, "{v} shows as {shown}");
            assert_eq!(format_decimal(shown), format_decimal(v), "{v}: the shown text is stable");
        }
    }

    #[test]
    fn format_mass_kg_appends_the_unit() {
        assert_eq!(format_mass_kg(2.0), "2 kg");
        assert_eq!(format_mass_kg(0.125), "0.125 kg");
    }
}
```

1c. `linkage-sim-rs/src/analysis/gravity_breakdown.rs`: tests of the shared name helpers (`display_name` is private until Step 3 makes it `pub`, but a child test module can already reach it):

Find in `linkage-sim-rs/src/analysis/gravity_breakdown.rs`:

```rust
        assert!(!is_braking(f64::NAN, 100.0));
    }
```

Add after it:

```rust
    #[test]
    fn display_name_prefers_a_visible_label() {
        assert_eq!(display_name(Some(&"Robot".to_string()), "W1"), "Robot");
        assert_eq!(display_name(Some(&"  ".to_string()), "W1"), "W1");
        assert_eq!(display_name(None, "W1"), "W1");
    }

    #[test]
    fn name_with_id_adds_the_id_only_when_the_name_differs() {
        assert_eq!(name_with_id("W1", "W1"), "W1");
        assert_eq!(name_with_id("Robot torso", "W2"), "Robot torso (W2)");
        assert_eq!(name_with_id("Arm", "b2"), "Arm (b2)");
    }

    #[test]
    fn point_mass_title_is_the_id_or_the_label_with_the_id() {
        let mut pm = PointMassJson { id: "W1".to_string(), label: None, mass: 1.0, local_pos: [0.0, 0.0] };
        assert_eq!(point_mass_title(&pm), "W1");
        pm.label = Some("Robot torso".to_string());
        assert_eq!(point_mass_title(&pm), "Robot torso (W1)");
        pm.label = Some(" ".to_string());
        assert_eq!(point_mass_title(&pm), "W1", "a blank label shows the id");
    }
```

1d. `linkage-sim-rs/src/gui/property_panel/weight_editor.rs`: create the file with its test module only (Step 3 adds the implementation above it). The `committed_number` tests drive a lone field over a stored value; the panel tests drive `draw_property_panel` with a selected weight.

Create `linkage-sim-rs/src/gui/property_panel/weight_editor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::SelectedEntity;
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, drew_text, key_press, primary_button, text_rect, typed,
    };
    use super::super::draw_property_panel;

    // ── committed_number ────────────────────────────────────────────────

    /// A lone weight mass field over a stored value that only changes when
    /// the test says so, as the blueprint only changes on a commit.
    struct Field {
        ctx: egui::Context,
        stored: f64,
    }

    impl Field {
        fn new(stored: f64) -> Self {
            Self { ctx: egui::Context::default(), stored }
        }

        /// One frame; returns what the field committed in it.
        fn frame(&mut self, events: Vec<egui::Event>) -> Option<f64> {
            self.frame_output(events).0
        }

        fn frame_output(&mut self, events: Vec<egui::Event>) -> (Option<f64>, egui::FullOutput) {
            let stored = self.stored;
            let mut committed = None;
            let output = central_panel_frame(&self.ctx, events, |ui| {
                committed = committed_number(ui, "mass", "hover", stored, mass_field);
            });
            (committed, output)
        }

        /// Screen rect of the field, found by giving it keyboard focus with
        /// Tab (it is the only widget) and leaving again with Esc.
        fn rect(&mut self) -> egui::Rect {
            assert_eq!(self.frame(vec![key_press(egui::Key::Tab)]), None);
            let id = self.ctx.memory(|m| m.focused()).expect("Tab focuses the field");
            assert_eq!(self.frame(vec![key_press(egui::Key::Escape)]), None);
            self.ctx.read_response(id).expect("the field was drawn").rect
        }
    }

    #[test]
    fn a_typed_value_commits_once_when_enter_is_pressed() {
        let mut field = Field::new(2.0);
        assert_eq!(field.frame(Vec::new()), None, "idle");
        assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None, "focus");
        assert_eq!(field.frame(vec![typed("5.5")]), None, "typing is not a commit");
        assert_eq!(field.frame(vec![key_press(egui::Key::Enter)]), Some(5.5));
        assert_eq!(field.frame(Vec::new()), None, "one commit per edit");
    }

    #[test]
    fn a_typed_value_commits_when_focus_moves_away() {
        let mut field = Field::new(2.0);
        field.frame(vec![key_press(egui::Key::Tab)]);
        field.frame(vec![typed("7")]);
        assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), Some(7.0), "Tab leaves the only field");
    }

    #[test]
    fn typed_values_are_clamped_to_the_weight_mass_range() {
        for (text, want) in [("0", 0.001), ("-3", 0.001), ("5000", 1000.0)] {
            let mut field = Field::new(2.0);
            field.frame(vec![key_press(egui::Key::Tab)]);
            field.frame(vec![typed(text)]);
            assert_eq!(field.frame(vec![key_press(egui::Key::Enter)]), Some(want), "typed {text}");
        }
    }

    /// egui reads the shown (rounded) text back when a field loses focus;
    /// leaving a field without typing must not turn that rounding into an
    /// edit, whatever the stored precision.
    #[test]
    fn leaving_a_field_without_typing_commits_nothing() {
        for stored in [2.0, 30.0000004, 0.031234567891] {
            let mut field = Field::new(stored);
            assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None);
            assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None, "{stored}: left without typing");
            assert_eq!(field.frame(Vec::new()), None);
        }
    }

    #[test]
    fn escape_drops_a_typed_value() {
        let mut field = Field::new(2.0);
        field.frame(vec![key_press(egui::Key::Tab)]);
        field.frame(vec![typed("9")]);
        assert_eq!(field.frame(vec![key_press(egui::Key::Escape)]), None);
        let (committed, output) = field.frame_output(Vec::new());
        assert_eq!(committed, None);
        assert!(drew_text(&output, "2 kg"), "the field shows the stored mass again");
    }

    /// egui stops reporting `dragged()` on the release frame, so a field
    /// over a per-frame copy must keep the dragged value itself.
    #[test]
    fn a_drag_commits_the_dragged_value_once_on_release() {
        let mut field = Field::new(2.0);
        let start = field.rect().center();
        let far = start + egui::vec2(40.0, 0.0);
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(start)]), None);
        assert_eq!(field.frame(vec![primary_button(start, true)]), None);
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(start + egui::vec2(20.0, 0.0))]), None, "mid-drag");
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(far)]), None, "mid-drag");
        let committed = field.frame(vec![primary_button(far, false)]).expect("the release commits");
        // speed 0.01 kg per point over ~40 points (egui rounds the value).
        assert!((committed - 2.4).abs() < 0.1, "dragged to {committed}");
        assert_eq!(field.frame(Vec::new()), None);
    }

    #[test]
    fn idle_frames_commit_nothing_even_for_out_of_range_masses() {
        for stored in [1500.0, 0.0] {
            let mut field = Field::new(stored);
            for _ in 0..3 {
                assert_eq!(field.frame(Vec::new()), None, "{stored}");
            }
        }
    }

    // ── Property panel ──────────────────────────────────────────────────

    /// Four-bar with weight W1 (2 kg at (0.03, 0.02)) on the coupler,
    /// selected; the link editor shows the crank, so W1's fields appear
    /// once, in the selected-weight editor.
    fn selected_weight_state() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.add_point_mass("coupler", 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        state.selected = Some(SelectedEntity::Weight { body_id: "coupler".to_string(), weight_id: "W1".to_string() });
        state.link_editor_body = Some("crank".to_string());
        state
    }

    fn panel_frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> egui::FullOutput {
        central_panel_frame(ctx, events, |ui| draw_property_panel(ui, state))
    }

    fn click(ctx: &egui::Context, state: &mut AppState, at: egui::Pos2) {
        panel_frame(ctx, state, vec![egui::Event::PointerMoved(at)]);
        panel_frame(ctx, state, vec![primary_button(at, true)]);
        panel_frame(ctx, state, vec![primary_button(at, false)]);
    }

    /// Click into W1's name field in the selected-weight editor.
    fn focus_name_field(ctx: &egui::Context, state: &mut AppState) {
        panel_frame(ctx, state, Vec::new());
        let id = name_field_id(SELECTED_SALT, "coupler", "W1");
        let rect = ctx.read_response(id).expect("the name field is drawn").rect;
        click(ctx, state, rect.center());
        assert!(ctx.memory(|m| m.has_focus(id)), "the click focuses the name field");
    }

    #[test]
    fn the_selected_weight_shows_its_name_mass_link_and_position() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        for want in ["Weight W1", "Link: coupler", "2 kg", "X 30 mm", "Y 20 mm", "Move to Link", "Reposition"] {
            assert!(texts.iter().any(|t| t == want), "missing {want:?} in {texts:?}");
        }
    }

    #[test]
    fn typing_a_name_commits_it_once_when_enter_is_pressed() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![typed("Robot torso")]);
        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label, None, "typing is not a commit");
        assert_eq!(state.undo_history.undo_count(), depth);
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Enter)]);

        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label.as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1, "one name edit = one undo step");
        assert!(drew_text(&panel_frame(&ctx, &mut state, Vec::new()), "Weight Robot torso (W1)"));
    }

    #[test]
    fn escape_drops_a_typed_name() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![typed("Oops")]);
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Escape)]);
        panel_frame(&ctx, &mut state, Vec::new());

        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label, None);
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn tab_from_the_name_to_the_mass_and_typing_commits_the_mass_once() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Tab)]);
        panel_frame(&ctx, &mut state, vec![typed("5")]);
        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().mass, 2.0, "typing is not a commit");
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Enter)]);

        let pm = state.find_point_mass("coupler", "W1").unwrap();
        assert_eq!(pm.mass, 5.0);
        assert_eq!(pm.label, None, "leaving the untouched name field changed nothing");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "one mass edit = one undo step");
    }

    #[test]
    fn a_stale_weight_selection_shows_no_weight_editor() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.selected = Some(SelectedEntity::Weight { body_id: "coupler".to_string(), weight_id: "W7".to_string() });
        let depth = state.undo_history.undo_count();

        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));

        assert!(!texts.iter().any(|t| t.starts_with("Weight ")), "{texts:?}");
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn the_link_editor_weights_section_shows_even_without_weights() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.selected = None;

        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        assert!(texts.iter().any(|t| t == "Weights (0)"), "crank has no weight: {texts:?}");
        assert!(texts.iter().any(|t| t == "Add weight"));

        state.link_editor_body = Some("coupler".to_string());
        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        assert!(texts.iter().any(|t| t == "Weights (1)"), "{texts:?}");
        assert!(texts.iter().any(|t| t == "W1"));
        assert!(texts.iter().any(|t| t == "X 30 mm"));
    }

    #[test]
    fn add_weight_adds_the_last_mass_at_the_link_cg_and_selects_it() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.last_point_mass_kg = 3.5;
        let cg = state.blueprint.as_ref().unwrap().bodies["crank"].cg_local;
        let depth = state.undo_history.undo_count();

        let output = panel_frame(&ctx, &mut state, Vec::new());
        let button = text_rect(&output, "Add weight").expect("the Add weight button is drawn");
        click(&ctx, &mut state, button.center());

        let pm = state.find_point_mass("crank", "W2").expect("the new weight gets the next id");
        assert_eq!(pm.mass, 3.5);
        assert_eq!(pm.local_pos, cg);
        assert_eq!(state.selected, Some(SelectedEntity::Weight { body_id: "crank".to_string(), weight_id: "W2".to_string() }));
        assert_eq!(state.undo_history.undo_count(), depth + 1);
    }
}
```

Register the module and the toolbar field's re-export in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
mod health;
mod undo_panel;
```

Replace with:

```rust
mod health;
mod undo_panel;
mod weight_editor;

pub(crate) use weight_editor::weight_mass_drag_value;
```

1e. `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, `mod tests`: the new tests need `GROUND_ID`; they reach `SelectedEntity` through `use super::*;`, which resolves once Step 3 adds it to the file's own imports.

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
```

Replace with:

```rust
mod tests {
    use super::*;
    use crate::core::state::GROUND_ID;
    use crate::gui::samples::SampleMechanism;
```

Add the tests directly above `fn stale_weight_edit_is_a_no_op` (after the test that ends with `assert_eq!(state.reassigning_point_mass, None);`):

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
        assert_eq!(state.reassigning_point_mass, None);
    }
```

Add after it:

```rust
    #[test]
    fn add_point_mass_uses_the_last_mass_and_selects_the_new_weight() {
        let (mut state, body, other, w) = four_bar_with_weight();
        state.last_point_mass_kg = 4.5;
        state.multi_selected = vec![SelectedEntity::Weight { body_id: body.clone(), weight_id: w }];
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::AddPointMass { body_id: other.clone(), local_pos: [0.01, -0.02] }),
        );

        let pm = state.find_point_mass(&other, "W2").expect("the next id");
        assert_eq!(pm.mass, 4.5);
        assert_eq!(pm.local_pos, [0.01, -0.02]);
        assert_eq!(state.selected, Some(SelectedEntity::Weight { body_id: other, weight_id: "W2".to_string() }));
        assert!(state.multi_selected.is_empty());
        assert_eq!(state.undo_history.undo_count(), depth + 1);
    }

    #[test]
    fn add_point_mass_on_ground_changes_nothing() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let selected = Some(SelectedEntity::Weight { body_id: body, weight_id: w });
        state.selected = selected.clone();
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::AddPointMass { body_id: GROUND_ID.to_string(), local_pos: [0.0, 0.0] }),
        );

        assert_eq!(state.selected, selected, "the selection stays");
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn set_point_mass_label_trims_and_clears_as_one_undo_step_each() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();
        let label = |state: &AppState| state.find_point_mass(&body, &w).unwrap().label.clone();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassLabel {
                body_id: body.clone(),
                weight_id: w.clone(),
                label: Some("  Robot torso ".to_string()),
            }),
        );
        assert_eq!(label(&state).as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassLabel {
                body_id: body.clone(),
                weight_id: w.clone(),
                label: Some("   ".to_string()),
            }),
        );
        assert_eq!(label(&state), None, "a blank name shows the id again");
        assert_eq!(state.undo_history.undo_count(), depth + 2);
    }

    #[test]
    fn removing_the_selected_weight_clears_it_from_the_selection() {
        let (mut state, body, other, w) = four_bar_with_weight();
        let removed = SelectedEntity::Weight { body_id: body.clone(), weight_id: w.clone() };
        let kept = SelectedEntity::Body(other);
        state.selected = Some(removed.clone());
        state.multi_selected = vec![removed, kept.clone()];

        apply_pending(&mut state, Some(PendingPropertyEdit::RemovePointMass { body_id: body, weight_id: w }));

        assert_eq!(state.selected, None);
        assert_eq!(state.multi_selected, vec![kept]);
    }
```

1f. `linkage-sim-rs/src/gui/mod.rs`, `mod tests`: import the new helpers,

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
    use crate::gui::test_support::key_press;
```

Replace with:

```rust
    use crate::gui::test_support::{central_panel_frame, drawn_texts, drew_text, key_press, typed};
```

and add the toolbar-field tests directly above `fn backspace_while_typing_in_a_text_field_does_not_delete_the_selection`:

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
        assert!(state.find_point_mass("coupler", "W1").is_some());
        assert_eq!(state.selected, Some(weight("W1")));
    }
```

Add after it:

```rust
    /// One frame of the + Mass toolbar field, as the toolbar lays it out.
    fn place_mass_field_frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> egui::FullOutput {
        central_panel_frame(ctx, events, |ui| {
            ui.horizontal(|ui| draw_place_mass_field(ui, state));
        })
    }

    #[test]
    fn the_place_mass_field_shows_only_while_the_mass_tool_is_active() {
        let ctx = egui::Context::default();
        let mut state = fourbar_with_weights();
        state.last_point_mass_kg = 7.5;

        state.active_tool = EditorTool::Select;
        let texts = drawn_texts(&place_mass_field_frame(&ctx, &mut state, Vec::new()));
        assert!(texts.is_empty(), "nothing outside the + Mass tool: {texts:?}");

        state.active_tool = EditorTool::PlaceMass;
        assert!(drew_text(&place_mass_field_frame(&ctx, &mut state, Vec::new()), "7.5 kg"));
    }

    #[test]
    fn the_place_mass_field_starts_at_the_last_mass_used() {
        let ctx = egui::Context::default();
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.active_tool = EditorTool::PlaceMass;
        assert!(drew_text(&place_mass_field_frame(&ctx, &mut state, Vec::new()), "1 kg"), "1 kg before any weight");

        state.add_point_mass("coupler", 12.5, [0.0, 0.0]).expect("weight added");
        assert!(drew_text(&place_mass_field_frame(&ctx, &mut state, Vec::new()), "12.5 kg"), "then the last mass added");
    }

    #[test]
    fn typing_in_the_place_mass_field_sets_the_next_mass_without_an_undo_step() {
        let ctx = egui::Context::default();
        let mut state = fourbar_with_weights();
        state.active_tool = EditorTool::PlaceMass;
        let depth = state.undo_history.undo_count();

        place_mass_field_frame(&ctx, &mut state, vec![key_press(egui::Key::Tab)]);
        place_mass_field_frame(&ctx, &mut state, vec![typed("2.5")]);
        place_mass_field_frame(&ctx, &mut state, vec![key_press(egui::Key::Enter)]);

        assert_eq!(state.last_point_mass_kg, 2.5);
        assert_eq!(state.undo_history.undo_count(), depth, "the next mass is a setting, not an edit");
    }

    #[test]
    fn idle_frames_leave_an_out_of_range_next_mass_alone() {
        let ctx = egui::Context::default();
        let mut state = fourbar_with_weights();
        state.active_tool = EditorTool::PlaceMass;
        state.last_point_mass_kg = 1500.0;

        place_mass_field_frame(&ctx, &mut state, Vec::new());
        place_mass_field_frame(&ctx, &mut state, Vec::new());

        assert_eq!(state.last_point_mass_kg, 1500.0);
    }
```

1g. `linkage-sim-rs/src/gui/canvas/mod.rs`, module `tests::weight_clicks`: the placement test now expects the snapped grid point, the selection and one undo step, and three tests join it (snapping off, the snap moving the weight off the pointer, the hint text). Replace the whole test `place_mass_click_uses_the_last_point_mass`:

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        #[test]
        fn place_mass_click_uses_the_last_point_mass() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            let expected = local_under(&state, &body, target);

            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W2").expect("the new weight gets the next id");
            assert_eq!(pm.mass, 3.5);
            assert_close("placement", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
        }
```

Replace with:

```rust
        #[test]
        fn place_mass_click_places_the_last_mass_on_the_snapped_grid_point() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            assert!(state.grid.snap_enabled, "fixture: snapping is on by default");
            let world = drop_world(&state, target);
            let [wx, wy] = state.view.screen_to_world(target.x, target.y);
            assert_ne!(world, [wx, wy], "fixture: the snap moves the placement");
            let expected = state.world_to_body_local(&body, world[0], world[1]);
            let depth = state.undo_history.undo_count();

            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W2").expect("the new weight gets the next id");
            assert_eq!(pm.mass, 3.5);
            assert_close("placement on the snapped grid point", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
            assert_eq!(state.selected, Some(weight(&body, "W2")), "the new weight is selected");
            assert_eq!(state.link_editor_body.as_deref(), Some(body.as_str()));
            assert_eq!(state.undo_history.undo_count(), depth + 1, "one placement = one undo step");
        }

        #[test]
        fn place_mass_with_snapping_off_lands_under_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            let expected = local_under(&state, &body, target);

            click(&ctx, &mut state, target);

            assert_close("unsnapped placement", state.find_point_mass(&body, "W2").unwrap().local_pos, expected);
        }

        /// The click that places a weight is not also a selection click: the
        /// snap can move the weight beyond the pick radius of the pointer,
        /// where selecting at the pointer would clear the selection.
        #[test]
        fn placing_a_weight_selects_it_even_when_the_snap_moves_it_off_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            let pick = super::super::colors::WEIGHT_HIT_RADIUS;
            let target = (0..40)
                .flat_map(|i| (0..20).map(move |j| Pos2::new(110.0 + i as f32 * 3.7, 640.0 + j as f32 * 3.1)))
                .find(|&p| screen_of(&state, drop_world(&state, p)).distance(p) > pick + 1.0)
                .expect("fixture: an empty-canvas point the snap moves off the pointer");
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());

            click(&ctx, &mut state, target);

            assert!(state.find_point_mass(&body, "W2").is_some());
            assert_eq!(state.selected, Some(weight(&body, "W2")));
        }

        #[test]
        fn the_place_mass_hint_names_the_next_weight_and_its_mass() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());

            let output = frame_with(&ctx, &mut state, Vec::new(), egui::Modifiers::NONE);

            let want = format!("Click to place weight W2 (3.5 kg) on '{body}' (Esc to cancel)");
            assert!(crate::gui::test_support::drew_text(&output, &want), "hint {want:?}");
        }
```

1h. `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`: create the file with its test module only (Step 3 adds `place_mass_hint` above it):

Create `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;

    #[test]
    fn place_mass_hint_names_the_next_weight_and_the_field_mass() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.last_point_mass_kg = 3.5;
        assert_eq!(
            place_mass_hint(&state, "coupler"),
            "Click to place weight W1 (3.5 kg) on 'coupler' (Esc to cancel)"
        );

        state.add_point_mass("crank", 1.25, [0.0, 0.0]).expect("W1 added");
        assert_eq!(
            place_mass_hint(&state, "coupler"),
            "Click to place weight W2 (1.25 kg) on 'coupler' (Esc to cancel)",
            "W1 is taken, and adding it made 1.25 kg the last mass used"
        );
    }
}
```

Register it in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
pub mod primitives;
mod force_render;
```

Replace with:

```rust
pub mod primitives;
mod force_render;
mod weights;
```

1i. Two idle-frame test helpers from earlier tasks now call `central_panel_frame` (DRY: one headless-frame helper for every GUI test). Behaviour is unchanged.

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
    use crate::gui::test_support::sorted_link_ids;

    /// Run one frame of the property panel with no user input at all.
    fn one_idle_frame(state: &mut AppState) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| draw_property_panel(ui, state));
        });
    }
```

Replace with:

```rust
    use crate::gui::test_support::{central_panel_frame, sorted_link_ids};

    /// Run one frame of the property panel with no user input at all.
    fn one_idle_frame(state: &mut AppState) {
        let _ = central_panel_frame(&egui::Context::default(), Vec::new(), |ui| draw_property_panel(ui, state));
    }
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    use crate::gui::sweep::{empty_trajectory_sweep_data, SweepMode};
```

Replace with:

```rust
    use crate::gui::sweep::{empty_trajectory_sweep_data, SweepMode};
    use crate::gui::test_support::central_panel_frame;
```

Find in `linkage-sim-rs/src/gui/plot_panel/mod.rs`:

```rust
    fn plot_panel_frame(state: &mut AppState, tab: PlotTab) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let tab_id = ui.id().with("plot_tab");
                ui.memory_mut(|mem| mem.data.insert_temp(tab_id, tab));
                draw_plot_panel(ui, state);
            });
        });
    }
```

Replace with:

```rust
    fn plot_panel_frame(state: &mut AppState, tab: PlotTab) {
        let _ = central_panel_frame(&egui::Context::default(), Vec::new(), |ui| {
            let tab_id = ui.id().with("plot_tab");
            ui.memory_mut(|mem| mem.data.insert_temp(tab_id, tab));
            draw_plot_panel(ui, state);
        });
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib --no-run
```

Expected: the build fails and no test runs: ``error: could not compile `linkage-sim-rs` (lib test) due to 88 previous errors``. The errors are the missing Step 3 pieces:

- ``error[E0425]: cannot find function `format_decimal` in this scope`` and the same for `format_mass_kg`, in the new `display_units` tests and in `weight_editor` (the field text).
- ``error[E0425]: cannot find function `name_with_id` in this scope``, the same for `point_mass_title`, and ``error[E0422]: cannot find struct, variant or union type `PointMassJson` in this scope``, in the `gravity_breakdown` name-helper tests.
- ``error[E0425]: cannot find function `committed_number` in this scope``, and the same for `mass_field`, `name_field_id` and `SELECTED_SALT`, in the `weight_editor` tests.
- ``error[E0433]: failed to resolve: use of unresolved module or unlinked crate `egui` `` (41 times) and ``error[E0412]: cannot find type `AppState` in this scope``, in the two test-only files `weight_editor.rs` and `rendering/weights.rs`, whose `use` lines arrive with the implementation in Step 3.
- ``error[E0599]: no variant named `SetPointMassLabel` found for enum `PendingPropertyEdit` ``, the same for `AddPointMass`, and ``error[E0433]: failed to resolve: use of undeclared type `SelectedEntity` ``, in the `pending_edits` tests.
- ``error[E0425]: cannot find function `draw_place_mass_field` in this scope`` (the `gui/mod.rs` tests) and the same for `place_mass_hint` (the `rendering/weights.rs` tests).
- ``error[E0432]: unresolved import `weight_editor::weight_mass_drag_value` ``, the re-export registered in `property_panel/mod.rs`.

- [ ] **Step 3: Implement**

3a. `linkage-sim-rs/src/gui/state/display_units.rs`: the formatting functions, added after the `impl DisplayUnits` block and above the `#[cfg(test)]` module from Step 1. `format_decimal` is what a weight field shows and what `committed_number` compares against, so "unchanged" and "edited" can be told apart after the rounding.

Find in `linkage-sim-rs/src/gui/state/display_units.rs`:

```rust
    /// X/Y axis label for length plots.
    pub fn length_axis_label(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => "m",
            LengthUnit::Millimeters => "mm",
        }
    }
}
```

Add after it:

```rust
/// Decimal places [`format_decimal`] keeps: 1 um in mm, 1 mg in kg. Finer
/// than anything a user types, coarse enough to read.
pub const DISPLAY_DECIMALS: usize = 6;

/// `value` with at most [`DISPLAY_DECIMALS`] decimals and no trailing
/// zeros: "2", "0.5", "1.234568". Never "-0". The text parses back to the
/// value rounded to those decimals, so a field that shows it and reads it
/// back can tell "unchanged" from "edited".
pub fn format_decimal(value: f64) -> String {
    let fixed = format!("{value:.DISPLAY_DECIMALS$}");
    let trimmed = if fixed.contains('.') { fixed.trim_end_matches('0').trim_end_matches('.') } else { fixed.as_str() };
    match trimmed {
        "-0" => "0".to_string(),
        text => text.to_string(),
    }
}

/// A mass in kg for display: "2 kg", "0.125 kg" ([`format_decimal`]).
pub fn format_mass_kg(kg: f64) -> String {
    format!("{} kg", format_decimal(kg))
}
```

3b. `linkage-sim-rs/src/gui/state/mod.rs`: re-export the formatters.

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
pub use display_units::{LengthUnit, AngleUnit, DisplayUnits};
```

Replace with:

```rust
pub use display_units::{format_decimal, format_mass_kg, LengthUnit, AngleUnit, DisplayUnits};
```

3c. `linkage-sim-rs/src/analysis/gravity_breakdown.rs`: import `PointMassJson`, make `display_name` public, and add `name_with_id` and `point_mass_title`. They are shared by the plot legend and the property panel (and by the canvas readout in Task 9), so no caller repeats the "name (id)" rule.

Find in `linkage-sim-rs/src/analysis/gravity_breakdown.rs`:

```rust
use crate::io::{point_mass_skip_reason, MechanismJson};
```

Replace with:

```rust
use crate::io::{point_mass_skip_reason, MechanismJson, PointMassJson};
```

Find in `linkage-sim-rs/src/analysis/gravity_breakdown.rs`:

```rust
/// `label` when it has visible text, else `fallback`.
fn display_name(label: Option<&String>, fallback: &str) -> String {
    match label {
        Some(text) if !text.trim().is_empty() => text.clone(),
        _ => fallback.to_string(),
    }
}
```

Replace with:

```rust
/// `label` when it has visible text, else `fallback`: the display name of
/// a weight (label or id) or a link (label or body id).
pub fn display_name(label: Option<&String>, fallback: &str) -> String {
    match label {
        Some(text) if !text.trim().is_empty() => text.clone(),
        _ => fallback.to_string(),
    }
}

/// `name` with `id` in parentheses when they differ ("Robot (W1)"), else
/// just `id`: a display name that stays unique when two labels are equal.
pub fn name_with_id(name: &str, id: &str) -> String {
    if name == id {
        id.to_string()
    } else {
        format!("{name} ({id})")
    }
}

/// Display title of point mass `pm`: "W1", or "Robot torso (W1)" when it
/// has a label (the property panel's weight editors, the canvas readout).
pub fn point_mass_title(pm: &PointMassJson) -> String {
    name_with_id(&display_name(pm.label.as_ref(), &pm.id), &pm.id)
}
```

3d. `linkage-sim-rs/src/gui/plot_panel/weights.rs`: the legend's point-mass branch uses `name_with_id`.

Find in `linkage-sim-rs/src/gui/plot_panel/weights.rs`:

```rust
use crate::analysis::gravity_breakdown::{Classification, WeightSource};
```

Replace with:

```rust
use crate::analysis::gravity_breakdown::{name_with_id, Classification, WeightSource};
```

Find in `linkage-sim-rs/src/gui/plot_panel/weights.rs`:

```rust
    } else if source.name == source.id {
        source.id.clone()
    } else {
        format!("{} ({})", source.name, source.id)
    }
```

Replace with:

```rust
    } else {
        name_with_id(&source.name, &source.id)
    }
```

3e. `linkage-sim-rs/src/gui/property_panel/weight_editor.rs`: insert the implementation at the very top of the file, above the `#[cfg(test)]` module created in Step 1. `committed_number` is the heart of it: egui's `DragValue` stops reporting `dragged()` on the release frame and drops its typed-text memory when the pointer is released, so a field over a copy rebuilt from the model every frame would lose a mouse drag that commits on `drag_stopped()`. The copy therefore lives in egui memory while the field is dragged or focused, and the commit is suppressed when the committed number is the stored value or the number the field was showing (leaving a field untouched makes egui read the rounded text back).

Insert at the very top of `linkage-sim-rs/src/gui/property_panel/weight_editor.rs`, above `#[cfg(test)]`:

```rust
//! Weight (point mass) editing in the property panel (payload weights,
//! spec Track 2 section 3, "Property panel"):
//!
//! - [`draw_selected_weight`]: the editor of the weight selected on the
//!   canvas: name, mass, owning link and body-local position.
//! - [`draw_link_weights`]: the link editor's Weights section, shown for
//!   every moving link (even without weights) with an Add weight button.
//!
//! Both draw the same fields ([`draw_weight_fields`]). A field edits a copy
//! of the stored value and commits once, when the user finishes: a drag is
//! released, or the field loses focus after typing (Enter, Tab or a click
//! elsewhere); Esc drops the edit. Each commit is one `PendingPropertyEdit`,
//! which the `AppState` weight API applies as one undo step.

use std::ops::RangeInclusive;

use eframe::egui;

use crate::analysis::gravity_breakdown::{display_name, name_with_id, point_mass_title};
use crate::gui::canvas::WEIGHT_COLOR;
use crate::gui::state::{format_decimal, AppState, DisplayUnits};
use crate::io::{BodyJson, PointMassJson};

use super::pending_edits::PendingPropertyEdit;

/// Masses (kg) a weight mass field accepts when typed or dragged.
pub(crate) const WEIGHT_MASS_RANGE_KG: RangeInclusive<f64> = 0.001..=1000.0;

/// Id salt of the selected-weight editor's fields.
const SELECTED_SALT: &str = "selected_weight";
/// Id salt of the link editor's weight fields.
const LINK_EDITOR_SALT: &str = "link_editor";

/// A weight field's number widget over `value`: shows `format_decimal`
/// text and applies typed text only when the field loses focus without Esc
/// (`update_while_editing(false)`), so Esc drops what was typed.
fn field_drag_value(value: &mut f64) -> egui::DragValue<'_> {
    egui::DragValue::new(value)
        .update_while_editing(false)
        .custom_formatter(|v, _| format_decimal(v))
}

/// Mass settings of a weight field (kg). Out-of-range masses stored in a
/// file are left alone on idle frames (`clamp_existing_to_range(false)`:
/// egui would otherwise clamp them and report a change).
fn mass_field(field: egui::DragValue<'_>) -> egui::DragValue<'_> {
    field.speed(0.01).range(WEIGHT_MASS_RANGE_KG).clamp_existing_to_range(false).suffix(" kg")
}

/// The mass field (kg) of a weight, also the + Mass tool's toolbar field.
pub(crate) fn weight_mass_drag_value(mass: &mut f64) -> egui::DragValue<'_> {
    mass_field(field_drag_value(mass))
}

/// Show a number field over a copy of the stored `value` (`configure` sets
/// its speed, range and text) and return the number the user committed.
///
/// The copy lives in egui memory while the field is dragged or focused:
/// egui stops reporting `dragged()` on the release frame, so a copy made
/// afresh each frame would lose the drag. A commit is the release of a
/// drag or the loss of focus; it returns `None` when the committed number
/// is the stored value or the number the field was showing. Leaving a field
/// without typing makes egui read the shown (rounded) text back, which must
/// not become an edit, and Esc keeps the stored value. Idle frames never
/// commit, whatever the value.
fn committed_number(
    ui: &mut egui::Ui,
    key: &str,
    hover: &str,
    value: f64,
    configure: impl FnOnce(egui::DragValue<'_>) -> egui::DragValue<'_>,
) -> Option<f64> {
    let copy_id = ui.id().with(("committed_number", key));
    let mut edited = ui.data(|d| d.get_temp::<f64>(copy_id)).unwrap_or(value);
    let response = ui.add(configure(field_drag_value(&mut edited))).on_hover_text(hover);
    if response.dragged() || response.has_focus() {
        ui.data_mut(|d| d.insert_temp(copy_id, edited));
        return None;
    }
    ui.data_mut(|d| d.remove::<f64>(copy_id));
    let finished = response.drag_stopped() || response.lost_focus();
    let shown = format_decimal(value).parse().unwrap_or(value);
    (finished && edited != value && edited != shown).then_some(edited)
}

/// Id of the name field of weight `weight_id` on `body_id` in the editor
/// `salt` (the selected-weight editor and the link editor both show one).
fn name_field_id(salt: &str, body_id: &str, weight_id: &str) -> egui::Id {
    egui::Id::new(("weight_name", salt, body_id, weight_id))
}

/// The weight's name (label) field; blank shows the id. The text being
/// typed lives in egui memory while the field has focus. Returns the text
/// to commit when the field loses focus (Enter, Tab or a click elsewhere);
/// Esc drops it.
fn name_field(ui: &mut egui::Ui, salt: &str, body_id: &str, pm: &PointMassJson) -> Option<String> {
    let id = name_field_id(salt, body_id, &pm.id);
    let typing_id = id.with("typing");
    let mut text = ui
        .data(|d| d.get_temp::<String>(typing_id))
        .unwrap_or_else(|| pm.label.clone().unwrap_or_default());
    let response = ui
        .add(egui::TextEdit::singleline(&mut text).id(id).hint_text(pm.id.as_str()).desired_width(140.0))
        .on_hover_text("Weight name, shown on the canvas and in the Weight Breakdown plot. Leave blank to show the id.");
    if response.has_focus() {
        ui.data_mut(|d| d.insert_temp(typing_id, text));
        return None;
    }
    ui.data_mut(|d| d.remove::<String>(typing_id));
    (response.lost_focus() && !ui.input(|i| i.key_pressed(egui::Key::Escape))).then_some(text)
}

/// Name, mass and body-local position fields of weight `pm` on `body_id`,
/// and its Move to Link, Reposition and Delete buttons. `salt` keeps the
/// field ids of the two editors apart.
fn draw_weight_fields(
    ui: &mut egui::Ui,
    units: &DisplayUnits,
    salt: &str,
    body_id: &str,
    pm: &PointMassJson,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let ids = || (body_id.to_string(), pm.id.clone());
    egui::Grid::new(("weight_fields", salt, body_id, pm.id.as_str())).num_columns(2).show(ui, |ui| {
        ui.label("Name");
        if let Some(label) = name_field(ui, salt, body_id, pm) {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::SetPointMassLabel { body_id, weight_id, label: Some(label) });
        }
        ui.end_row();

        ui.label("Mass");
        if let Some(mass) = committed_number(ui, "mass", "Weight mass in kg", pm.mass, mass_field) {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::SetPointMassMass { body_id, weight_id, mass });
        }
        ui.end_row();

        ui.label("Position");
        ui.horizontal(|ui| {
            let [x, y] = pm.local_pos;
            let speed = units.length(0.001);
            let suffix = units.length_suffix();
            let new_x = committed_number(ui, "x", "Body-local X position", units.length(x), |f| {
                f.speed(speed).prefix("X ").suffix(suffix)
            });
            let new_y = committed_number(ui, "y", "Body-local Y position", units.length(y), |f| {
                f.speed(speed).prefix("Y ").suffix(suffix)
            });
            let local_pos = match (new_x, new_y) {
                (Some(nx), _) => Some([units.length_to_si(nx), y]),
                (None, Some(ny)) => Some([x, units.length_to_si(ny)]),
                (None, None) => None,
            };
            if let Some(local_pos) = local_pos {
                let (body_id, weight_id) = ids();
                *pending = Some(PendingPropertyEdit::SetPointMassPosition { body_id, weight_id, local_pos });
            }
        });
        ui.end_row();
    });
    ui.horizontal(|ui| {
        if ui.small_button("Move to Link").on_hover_text("Click a different link to move this weight there").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::ReassignPointMass { body_id, weight_id });
        }
        if ui.small_button("Reposition").on_hover_text("Click on the canvas to move this weight").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::RepositionPointMass { body_id, weight_id });
        }
        if ui.small_button("Delete").on_hover_text("Delete this weight").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::RemovePointMass { body_id, weight_id });
        }
    });
}

/// "coupler", or "Arm (b2)" for a labelled link.
fn link_name(state: &AppState, body_id: &str) -> String {
    let label = state.blueprint.as_ref().and_then(|bp| bp.bodies.get(body_id)).and_then(|b| b.label.as_ref());
    name_with_id(&display_name(label, body_id), body_id)
}

/// The editor of the weight selected on the canvas
/// (`SelectedEntity::Weight`): name, mass, owning link and body-local
/// position. Draws nothing for a stale selection (the weight was deleted,
/// undone or moved to another link since it was selected).
pub(super) fn draw_selected_weight(
    ui: &mut egui::Ui,
    state: &AppState,
    body_id: &str,
    weight_id: &str,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let Some(pm) = state.find_point_mass(body_id, weight_id) else { return };
    egui::CollapsingHeader::new(
        egui::RichText::new(format!("Weight {}", point_mass_title(pm))).color(state.nc(WEIGHT_COLOR)),
    )
    .id_salt("selected_weight")
    .default_open(true)
    .show(ui, |ui| {
        ui.label(format!("Link: {}", link_name(state, body_id)));
        draw_weight_fields(ui, &state.display_units, SELECTED_SALT, body_id, pm, pending);
    });
}

/// The link editor's Weights section for moving link `body_id`, shown even
/// when the link carries no weight: the fields of each weight, then an Add
/// weight button that adds a weight of the last mass used
/// (`AppState::last_point_mass_kg`) at the link's centre of mass.
pub(super) fn draw_link_weights(
    ui: &mut egui::Ui,
    state: &AppState,
    body_id: &str,
    body: &BodyJson,
    pending: &mut Option<PendingPropertyEdit>,
) {
    egui::CollapsingHeader::new(
        egui::RichText::new(format!("Weights ({})", body.point_masses.len())).color(state.nc(WEIGHT_COLOR)),
    )
    .id_salt(format!("point_masses_{body_id}"))
    .default_open(true)
    .show(ui, |ui| {
        for pm in &body.point_masses {
            ui.push_id(pm.id.as_str(), |ui| {
                ui.label(egui::RichText::new(point_mass_title(pm)).strong());
                draw_weight_fields(ui, &state.display_units, LINK_EDITOR_SALT, body_id, pm, pending);
            });
            ui.separator();
        }
        let hover = format!(
            "Add a {} kg weight at this link's centre of mass (the + Mass toolbar field sets the mass), then drag it on the canvas",
            format_decimal(state.last_point_mass_kg)
        );
        if ui.button("Add weight").on_hover_text(hover).clicked() {
            *pending = Some(PendingPropertyEdit::AddPointMass { body_id: body_id.to_string(), local_pos: body.cg_local });
        }
    });
}
```

3f. `linkage-sim-rs/src/gui/property_panel/mod.rs`. Import `SelectedEntity`:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
use crate::gui::state::{AppState, PropertyPanelTab};
```

Replace with:

```rust
use crate::gui::state::{AppState, PropertyPanelTab, SelectedEntity};
```

Draw the selected-weight editor right after the Mechanism Health section, above the Link Editor:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
    // ── Mechanism Health ──────────────────────────────────────────────
    health::draw_health_section(ui, state);
```

Add after it:

```rust
    // ── Selected weight (name, mass, owning link, position) ──────────
    if let Some(SelectedEntity::Weight { body_id, weight_id }) = &state.selected {
        weight_editor::draw_selected_weight(ui, state, body_id, weight_id, &mut pending);
    }
```

Replace the whole `// ── Point Masses ──` block, down to (not including) `// ── Body Geometry ──`. The old block showed a `Point Masses (n)` header only when the list was non-empty, with `#n` rows, per-keystroke commits and an "x" remove button; the new one is always shown for a moving link, so **Add weight** is reachable on a link without weights:

Find in `linkage-sim-rs/src/gui/property_panel/mod.rs`:

```rust
                    // ── Point Masses ─────────────────────────────────────
                    if body_id != GROUND_ID {
                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                if !bp_body.point_masses.is_empty() {
                                    ui.separator();
                                    let pm_color = state.nc(egui::Color32::from_rgb(255, 200, 50));
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new(format!(
                                            "Point Masses ({})", bp_body.point_masses.len()
                                        )).color(pm_color),
                                    )
                                        .id_salt(format!("point_masses_{}", body_id))
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            let units = &state.display_units;
                                            for (i, pm) in bp_body.point_masses.iter().enumerate() {
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("#{}", i + 1));

                                                    let mut mass_val = pm.mass;
                                                    let mr = ui.add(
                                                        egui::DragValue::new(&mut mass_val)
                                                            .speed(0.01)
                                                            .range(0.001..=1000.0)
                                                            // Never clamp a loaded mass on an idle frame:
                                                            // egui would report it as changed and the
                                                            // panel would commit a silent edit.
                                                            .clamp_existing_to_range(false)
                                                            .prefix("m: ")
                                                            .suffix(" kg"),
                                                    ).on_hover_text("Point mass magnitude in kg");
                                                    if mr.drag_stopped() || (mr.changed() && !mr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            mass: mass_val,
                                                        });
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(20.0);
                                                    let mut x_display = units.length(pm.local_pos[0]);
                                                    let xr = ui.add(
                                                        egui::DragValue::new(&mut x_display)
                                                            .speed(units.length(0.001))
                                                            .prefix("X ")
                                                            .suffix(units.length_suffix()),
                                                    ).on_hover_text("Body-local X position");
                                                    if xr.drag_stopped() || (xr.changed() && !xr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassPosition {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            local_pos: [units.length_to_si(x_display), pm.local_pos[1]],
                                                        });
                                                    }

                                                    let mut y_display = units.length(pm.local_pos[1]);
                                                    let yr = ui.add(
                                                        egui::DragValue::new(&mut y_display)
                                                            .speed(units.length(0.001))
                                                            .prefix("Y ")
                                                            .suffix(units.length_suffix()),
                                                    ).on_hover_text("Body-local Y position");
                                                    if yr.drag_stopped() || (yr.changed() && !yr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::SetPointMassPosition {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                            local_pos: [pm.local_pos[0], units.length_to_si(y_display)],
                                                        });
                                                    }

                                                    if ui.small_button("x")
                                                        .on_hover_text("Remove this point mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::RemovePointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(20.0);
                                                    if ui.small_button("Move to Link")
                                                        .on_hover_text("Click a different link to reassign this mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::ReassignPointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
                                                    }
                                                    if ui.small_button("Reposition")
                                                        .on_hover_text("Click on the canvas to move this mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::RepositionPointMass {
                                                            body_id: body_id.clone(),
                                                            weight_id: pm.id.clone(),
                                                        });
                                                    }
                                                });
                                            }
                                        });
                                }
                            }
                        }
                    }
```

Replace with:

```rust
                    // ── Weights (always shown for a moving link) ─────────
                    if body_id != GROUND_ID {
                        if let Some(bp_body) = state.blueprint.as_ref().and_then(|bp| bp.bodies.get(&body_id)) {
                            ui.separator();
                            weight_editor::draw_link_weights(ui, state, &body_id, bp_body, &mut pending);
                        }
                    }
```

3g. `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`. File import:

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
use crate::gui::state::AppState;
```

Replace with:

```rust
use crate::gui::state::{AppState, SelectedEntity};
```

Variants (replacing the `SetPointMassPosition` / `RemovePointMass` lines):

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
    /// Commit a typed / drag-stopped body-local X or Y of a weight.
    SetPointMassPosition { body_id: String, weight_id: String, local_pos: [f64; 2] },
    RemovePointMass { body_id: String, weight_id: String },
```

Replace with:

```rust
    /// Commit a typed / drag-stopped body-local X or Y of a weight.
    SetPointMassPosition { body_id: String, weight_id: String, local_pos: [f64; 2] },
    /// Commit a typed weight name (trimmed; blank clears it).
    SetPointMassLabel { body_id: String, weight_id: String, label: Option<String> },
    /// Add a weight of `AppState::last_point_mass_kg` at `local_pos` on
    /// `body_id` and select it (the link editor's Add weight button).
    AddPointMass { body_id: String, local_pos: [f64; 2] },
    /// Delete a weight; a selection of it is cleared.
    RemovePointMass { body_id: String, weight_id: String },
```

Apply arms (replacing the `RemovePointMass` arm). Adding selects the new weight and clears the multi-selection; removing a weight clears any selection of it, so the selected-weight editor never points at a deleted weight:

Find in `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`:

```rust
            PendingPropertyEdit::RemovePointMass { body_id, weight_id } => {
                state.remove_point_mass_by_id(&body_id, &weight_id);
            }
```

Replace with:

```rust
            PendingPropertyEdit::SetPointMassLabel { body_id, weight_id, label } => {
                state.set_point_mass_label(&body_id, &weight_id, label);
            }
            PendingPropertyEdit::AddPointMass { body_id, local_pos } => {
                if let Some(weight_id) = state.add_point_mass(&body_id, state.last_point_mass_kg, local_pos) {
                    state.multi_selected.clear();
                    state.selected = Some(SelectedEntity::Weight { body_id, weight_id });
                }
            }
            PendingPropertyEdit::RemovePointMass { body_id, weight_id } => {
                if state.remove_point_mass_by_id(&body_id, &weight_id) {
                    let removed = SelectedEntity::Weight { body_id, weight_id };
                    if state.selected.as_ref() == Some(&removed) {
                        state.selected = None;
                    }
                    state.multi_selected.retain(|e| *e != removed);
                }
            }
```

3h. `linkage-sim-rs/src/gui/canvas/mod.rs`: re-export the weight colour for the property panel.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
pub use colors::{classification_color, to_grayscale};
```

Replace with:

```rust
pub use colors::{classification_color, to_grayscale, WEIGHT_COLOR};
```

3i. `linkage-sim-rs/src/gui/canvas/interaction.rs`. In `handle_interaction`, a click that places a weight is not also a selection click: the snap can move the weight beyond the pick radius of the pointer, where selecting at the pointer would clear the selection just made. Replace the Place Mass call

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    // ── Interaction: Place Mass tool ─────────────────────────────────────
    if state.active_tool == EditorTool::PlaceMass {
        handle_place_mass(ui, painter, response, state, body_segments);
    }
```

Replace with:

```rust
    // ── Interaction: Place Mass tool ─────────────────────────────────────
    // A click that places a weight is not also a selection click (below).
    let placed_weight = state.active_tool == EditorTool::PlaceMass
        && handle_place_mass(ui, painter, response, state, body_segments);
```

and add `!placed_weight` to the click-selection gate:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
        && state.active_tool != EditorTool::DrawBodyGeometry
        && response.clicked()
```

Replace with:

```rust
        && state.active_tool != EditorTool::DrawBodyGeometry
        && !placed_weight
        && response.clicked()
```

In `handle_weight_drag`, the drop point uses the shared helper. Replace

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    let [wx, wy] = state.view.screen_to_world(pointer.x, pointer.y);
    let (gx, gy) = state.grid.snap_point(wx, wy);
    drag.current_world = [gx, gy];
```

Replace with:

```rust
    let [gx, gy] = snapped_world(state, pointer);
    drag.current_world = [gx, gy];
```

and add the helper after `handle_weight_drag`, above `weight_drop_target`:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    state.weight_drag = Some(drag);
}
```

Add after it:

```rust
/// The world point under screen point `screen`, snapped to the grid when
/// snapping is on: where a dragged or placed weight lands.
fn snapped_world(state: &AppState, screen: Pos2) -> [f64; 2] {
    let [wx, wy] = state.view.screen_to_world(screen.x, screen.y);
    let (gx, gy) = state.grid.snap_point(wx, wy);
    [gx, gy]
}
```

Replace the whole `handle_place_mass` section (from `// ── Place Mass tool ──` to just before `// ── Add Body tool ──`). Phase 1 (picking the link) is unchanged apart from the colour; phase 2 previews the snapped landing point and, on click, snaps, adds the weight (`add_point_mass` names it from `next_point_mass_id` and remembers the mass), selects it and returns `true`. A failed placement leaves the link selected and shows a status message:

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
// ── Place Mass tool ─────────────────────────────────────────────────────────

fn handle_place_mass(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &mut AppState,
    body_segments: &[BodySegment],
) {
    let has_body = state.place_mass_body.is_some();

    if has_body {
        // Phase 2: body is selected, draw preview and place on click.
        if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
            let point_mass_color = Color32::from_rgb(255, 200, 50);
            // Draw gold circle preview at cursor
            painter.circle_filled(hover, 5.0, point_mass_color.linear_multiply(0.5));
            painter.circle_stroke(hover, 7.0, Stroke::new(1.0, point_mass_color));
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                let body_id = state.place_mass_body.clone().unwrap();
                let [lx, ly] = state.world_to_body_local(&body_id, wx, wy);
                state.add_point_mass(&body_id, state.last_point_mass_kg, [lx, ly]);
                state.place_mass_body = None;
                state.active_tool = EditorTool::Select;
                state.selected = Some(SelectedEntity::Body(body_id.clone()));
                state.link_editor_body = Some(body_id);
            }
        }
    } else {
        // Phase 1: select a body by clicking near a link segment.
        // Highlight nearest body segment on hover.
        if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
            if let Some(seg_hit) = find_nearest_body_segment(hover, body_segments, LINK_PICK_RADIUS) {
                // Draw highlight on the hovered body segment
                let highlight = Color32::from_rgb(255, 200, 50).linear_multiply(0.4);
                painter.line_segment(
                    [seg_hit.screen_pos, {
                        // Re-find the segment endpoints for drawing
                        let seg = body_segments.iter().find(|s| {
                            s.body_id == seg_hit.body_id
                        });
                        if let Some(s) = seg {
                            s.screen_b
                        } else {
                            seg_hit.screen_pos
                        }
                    }],
                    Stroke::new(4.0, highlight),
                );
            }
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                // Find nearest body segment within the link pick radius
                if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, LINK_PICK_RADIUS) {
                    state.place_mass_body = Some(seg_hit.body_id.clone());
                }
            }
        }
    }
}
```

Replace with:

```rust
// ── Place Mass tool ─────────────────────────────────────────────────────────

/// The + Mass tool.
///
/// Phase 1 (no link picked): a click near a link picks it. Phase 2: a
/// preview shows where the weight lands; a click adds a weight of
/// `state.last_point_mass_kg` (the toolbar field) to the picked link at the
/// click point, snapped to the grid when snapping is on (one undo step),
/// selects the new weight and returns to Select. Returns true when this
/// frame's click placed a weight: the caller then skips click selection,
/// which would pick at the pointer although the snap may have moved the
/// weight away from it.
fn handle_place_mass(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &mut AppState,
    body_segments: &[BodySegment],
) -> bool {
    let color = state.nc(WEIGHT_COLOR);
    let Some(body_id) = state.place_mass_body.clone() else {
        // Phase 1: select a body by clicking near a link segment.
        // Highlight nearest body segment on hover.
        if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
            if let Some(seg_hit) = find_nearest_body_segment(hover, body_segments, LINK_PICK_RADIUS) {
                // Draw highlight on the hovered body segment
                let highlight = color.linear_multiply(0.4);
                painter.line_segment(
                    [seg_hit.screen_pos, {
                        // Re-find the segment endpoints for drawing
                        let seg = body_segments.iter().find(|s| {
                            s.body_id == seg_hit.body_id
                        });
                        if let Some(s) = seg {
                            s.screen_b
                        } else {
                            seg_hit.screen_pos
                        }
                    }],
                    Stroke::new(4.0, highlight),
                );
            }
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                // Find nearest body segment within the link pick radius
                if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, LINK_PICK_RADIUS) {
                    state.place_mass_body = Some(seg_hit.body_id.clone());
                }
            }
        }
        return false;
    };

    // Phase 2: preview the landing point; place on click.
    if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
        let [gx, gy] = snapped_world(state, hover);
        let [sx, sy] = state.view.world_to_screen(gx, gy);
        let landing = Pos2::new(sx, sy);
        painter.circle_filled(landing, WEIGHT_RADIUS, color.linear_multiply(0.5));
        painter.circle_stroke(landing, WEIGHT_HIT_RADIUS, Stroke::new(1.0, color));
    }
    if !response.clicked() {
        return false;
    }
    let Some(pos) = response.interact_pointer_pos() else { return false };
    let [gx, gy] = snapped_world(state, pos);
    let local = state.world_to_body_local(&body_id, gx, gy);
    let mass = state.last_point_mass_kg;
    state.place_mass_body = None;
    state.active_tool = EditorTool::Select;
    state.link_editor_body = Some(body_id.clone());
    state.multi_selected.clear();
    match state.add_point_mass(&body_id, mass, local) {
        Some(weight_id) => {
            state.selected = Some(SelectedEntity::Weight { body_id, weight_id });
        }
        None => {
            state.status_message = Some(format!("Could not place a {mass} kg weight on '{body_id}'"));
            state.status_message_time = 3.0;
            state.selected = Some(SelectedEntity::Body(body_id));
        }
    }
    true
}
```

3j. `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`: insert the hint function at the very top of the file, above the `#[cfg(test)]` module created in Step 1. It uses the same next-id rule as `add_point_mass` (`io::next_point_mass_id`), so the hint names the id the weight will get.

Insert at the very top of `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`, above `#[cfg(test)]`:

```rust
//! Canvas weights (point masses): placement hint.

use crate::gui::state::{format_mass_kg, AppState};
use crate::io::next_point_mass_id;

/// Hint while the + Mass tool waits for the drop point on `body_id`: the id
/// the new weight gets (`io::next_point_mass_id`, the default name) and its
/// mass (the toolbar field, `AppState::last_point_mass_kg`).
pub(super) fn place_mass_hint(state: &AppState, body_id: &str) -> String {
    // Without a blueprint no weight exists yet, so the next id is W1.
    let next_id = state
        .blueprint
        .as_ref()
        .map_or_else(|| "W1".to_string(), |bp| next_point_mass_id(&bp.bodies));
    format!(
        "Click to place weight {next_id} ({}) on '{body_id}' (Esc to cancel)",
        format_mass_kg(state.last_point_mass_kg)
    )
}
```

3k. `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`, Place Mass hint in `render_overlays`:

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
                Some(format!("Click anywhere to place a point mass on '{}' (Esc to cancel)", body_id))
```

Replace with:

```rust
                Some(weights::place_mass_hint(state, body_id))
```

3l. `linkage-sim-rs/src/gui/mod.rs`, toolbar. The `+ Mass` hover text explains the two-click workflow and the mass field:

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
                if ui.add(mass_btn)
                    .on_hover_text("Place a point mass on a body")
                    .clicked()
```

Replace with:

```rust
                if ui.add(mass_btn)
                    .on_hover_text("Place a weight (point mass): click a link, then the spot (snapped to the grid when snapping is on). The field next to this button sets its mass; it starts at the last mass used.")
                    .clicked()
```

The mass field goes right after the button's click handler:

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
                    } else {
                        self.state.place_mass_body = None;
                    }
                }

                ui.separator();

                // ── Playback controls (green/yellow) ────────────────
```

Replace with:

```rust
                    } else {
                        self.state.place_mass_body = None;
                    }
                }
                draw_place_mass_field(ui, &mut self.state);

                ui.separator();

                // ── Playback controls (green/yellow) ────────────────
```

Add the field function directly under the `// ── Delete shortcut ──` comment, above `handle_delete_shortcut`. It edits `AppState::last_point_mass_kg`, which every added weight updates (so the field starts at the last mass used), through the same `weight_mass_drag_value` as the property panel. It is a setting, not an edit: no undo step.

Find in `linkage-sim-rs/src/gui/mod.rs`:

```rust
// ── Delete shortcut ─────────────────────────────────────────────────────────
```

Add after it:

```rust
/// The + Mass tool's mass field (kg), shown while the tool is active: the
/// mass of the next weight placed. It edits `AppState::last_point_mass_kg`,
/// which every added weight updates, so it starts at the last mass used. A
/// setting, not an edit: no undo step.
fn draw_place_mass_field(ui: &mut egui::Ui, state: &mut AppState) {
    if state.active_tool != EditorTool::PlaceMass {
        return;
    }
    ui.add(property_panel::weight_mass_drag_value(&mut state.last_point_mass_kg))
        .on_hover_text("Mass of the next weight you place. Starts at the last mass used.");
}
```

3m. Docs. `docs/ai/02-system.yaml`, `invariants_to_protect`: insert the two new invariants directly above `- delete_shortcut_ignores_focused_widgets` (after the weight-drag invariant that ends with "has moved > 6 px."):

Find in `docs/ai/02-system.yaml`:

```yaml
    a release outside the canvas rect commits nothing. The drag starts from
    pointer.press_origin(): egui reports drag_started only after the pointer
    has moved > 6 px.
```

Replace with:

```yaml
    a release outside the canvas rect commits nothing. The drag starts from
    pointer.press_origin(): egui reports drag_started only after the pointer
    has moved > 6 px.
  - weight_fields_commit_once_per_edit — in property_panel/weight_editor.rs
    every weight field (name, mass, body-local X/Y, in the selected-weight
    editor and the link editor's Weights section) edits a copy and emits
    ONE PendingPropertyEdit when the user finishes. committed_number keeps
    the copy in egui memory while the field is dragged or focused and
    commits on drag_stopped() || lost_focus(), but not when the committed
    number is the stored value or the number the field showed
    (format_decimal round trip), so focusing and leaving a field records
    nothing; update_while_editing(false) makes Esc drop typed text. The name
    field buffers typed text the same way and commits on lost focus
    without Esc. The + Mass toolbar field edits
    AppState::last_point_mass_kg directly (a setting, so no undo step) with the
    same weight_mass_drag_value (0.001..=1000 kg).
  - place_mass_click_snaps_and_selects — the + Mass tool's placement click
    snaps to the grid (canvas interaction.rs snapped_world, shared with the
    weight drag), adds one weight of last_point_mass_kg (one undo step),
    selects SelectedEntity::Weight and returns true so handle_interaction
    skips click selection that frame (the snap can move the weight beyond
    the pick radius of the pointer). The link editor's Add weight button
    adds last_point_mass_kg at the link's base cg_local and selects it.
```

and in `lessons_learned`, above `- egui_plot fits its auto bounds ...` (after the `clamp_existing_to_range(false)` lesson):

Find in `docs/ai/02-system.yaml`:

```yaml
    .clamp_existing_to_range(false) on committed fields (weight mass field).
```

Replace with:

```yaml
    .clamp_existing_to_range(false) on committed fields (weight mass field).
  - egui DragValue stops reporting dragged() on the release frame and
    drops its precise-value memory on any pointer release, so a DragValue
    over a copy rebuilt from the model every frame loses a mouse drag when
    it commits on drag_stopped(), since the release frame sees the old value.
    Keep the copy in egui memory while dragged/focused (weight_editor
    committed_number). Existing per-frame-copy DragValues that commit on
    drag_stopped() (ground pivot X/Y, Dist, Angle, driver RPM) still lose
    drags.
```

`docs/ai/03-structure.yaml`: the `gravity_breakdown` note names the shared name helpers,

Find in `docs/ai/03-structure.yaml`:

```yaml
      gravity_breakdown: per-weight gravity power P_g,i = m_i g.v_i (weight_sources from the blueprint = link self-weights + applied point masses; gravity_powers; gravity_vector; classify / force_share / is_braking with EPS_REL_LDOT, NEUTRAL_REL, BRAKE_TOL_REL). Pure, no GUI deps.
```

Replace with:

```yaml
      gravity_breakdown: per-weight gravity power P_g,i = m_i g.v_i (weight_sources from the blueprint = link self-weights + applied point masses; gravity_powers; gravity_vector; classify / force_share / is_braking with EPS_REL_LDOT, NEUTRAL_REL, BRAKE_TOL_REL; display names display_name, name_with_id, point_mass_title shared by the plot legend, the property panel and the canvas). Pure, no GUI deps.
```

`canvas.files.rendering` lists the new canvas file,

Find in `docs/ai/03-structure.yaml`:

```yaml
        force_render: canvas/rendering/force_render.rs (force element visualization + load-path heat map; actuator label value comes from AppState::actuator_label_force)
```

Replace with:

```yaml
        force_render: canvas/rendering/force_render.rs (force element visualization + load-path heat map; actuator label value comes from AppState::actuator_label_force)
        weights: canvas/rendering/weights.rs (payload weights on the canvas - place_mass_hint)
```

the `property_panel` entry gains the module and a note,

Find in `docs/ai/03-structure.yaml`:

```yaml
    files: [mod, diagnostics, force_editor, health, pending_edits, undo_panel]
```

Replace with:

```yaml
    files: [mod, diagnostics, force_editor, health, pending_edits, undo_panel, weight_editor]
    notes:
      weight_editor: selected-weight editor (draw_selected_weight) and the link editor's Weights section with Add weight (draw_link_weights); shared fields draw_weight_fields; committed_number (commit once per edit); weight_mass_drag_value (also the + Mass toolbar field)
```

and `other_top_level_gui` mentions the toolbar field and the new test helpers:

Find in `docs/ai/03-structure.yaml`:

```yaml
    - mod.rs (LinkageApp + update loop orchestration; handle_delete_shortcut)
```

Replace with:

```yaml
    - mod.rs (LinkageApp + update loop orchestration; handle_delete_shortcut; draw_place_mass_field)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press; extend it instead of re-implementing fixtures per module)
```

Replace with:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press, typed, the headless frame central_panel_frame, painted-output inspection drawn_texts, drew_text, text_rect; extend it instead of re-implementing fixtures per module)
```

`docs/ai/05-update-tracker.md`: add the entry at the top, below the `---` line:

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Add after it:

```markdown
## 2026-09-29 — Payload weights Task 8: placement and the weight editor
- `+ Mass` toolbar: a mass field (`draw_place_mass_field`, shown while the
  tool is active) edits `AppState::last_point_mass_kg`, so it starts at the
  last mass used. The placement click snaps to the grid (`snapped_world`,
  shared with the weight drag), previews the snapped landing point, selects
  the new weight and consumes the click (`handle_place_mass` returns true;
  no click selection that frame). The hint names the next id and mass
  (`canvas/rendering/weights.rs place_mass_hint`, "Click to place weight
  W3 (2.5 kg) on 'coupler'"). A failed placement shows a status message.
- `property_panel/weight_editor.rs` (new): `draw_selected_weight` (Weight
  <name> header, Link: <owning link>, name, mass, body-local X/Y, Move to
  Link / Reposition / Delete) above the Link Editor; `draw_link_weights`
  replaces the Point Masses block: "Weights (n)", always shown for a moving
  link, one editor per weight, Add weight (last mass at the link's base
  CG, selected). Fields commit once per edit via `committed_number` (see
  02-system `weight_fields_commit_once_per_edit`).
- `PendingPropertyEdit::{SetPointMassLabel, AddPointMass}`; `RemovePointMass`
  clears a selection of the removed weight.
- `gui/state/display_units.rs`: `format_decimal` (6 decimals, trimmed),
  `format_mass_kg`. `analysis::gravity_breakdown`: `display_name` is pub,
  new `name_with_id` (also used by the Weight Breakdown legend) and
  `point_mass_title` ("W1" / "Robot torso (W1)").
  `canvas::WEIGHT_COLOR` is re-exported for the property panel.
- Tests: `gui::property_panel::weight_editor::tests` (typed commit on Enter
  and on Tab, clamping, no commit when leaving untouched or on Esc, drag
  commit on release, idle frames; panel shows name/mass/link/position,
  name commit, Esc drops a name, Tab to mass, stale selection, empty
  Weights section, Add weight click), `pending_edits::tests`,
  `gui::tests` (toolbar field visibility, starts at last mass, typing, idle
  frames), `gui::canvas::tests::weight_clicks` (snapped placement selects
  the weight, unsnapped placement, snap off the pointer still selects,
  hint), `display_units::tests`, `gravity_breakdown` name helpers.
- `gui/test_support.rs` (shared test helpers, DRY): adds `typed`,
  `central_panel_frame` (one headless frame of a closure in a central
  panel), `drawn_texts`, `drew_text` and `text_rect`. The new tests use
  them instead of local copies, and the property panel's `one_idle_frame`
  (Task 2) and the plot panel's `plot_panel_frame` (Task 7) now call
  `central_panel_frame` too. The toolbar field tests use
  `test_support::key_press`.
- Mutation checks: dropping `!placed_weight` from the click-selection gate
  fails `placing_a_weight_selects_it_even_when_the_snap_moves_it_off_the_pointer`
  and the snapped-placement test; dropping `edited != shown` from
  `committed_number` fails `leaving_a_field_without_typing_commits_nothing`.
- Docs: FEATURES (Place Mass tool, Weight editor), ARCHITECTURE (property
  panel weight editing), 02-system `weight_fields_commit_once_per_edit`,
  `place_mass_click_snaps_and_selects` and the DragValue release-frame
  lesson, 03-structure (weight_editor, rendering/weights, name helpers,
  test_support). The spec's hands-on checklist wording for the Weights
  section lands with the checklist in Task 10.
```

`docs/FEATURES.md`, section Mechanism Building: replace the `**Place Mass tool**` bullet and add the weight-editor bullet:

Find in `docs/FEATURES.md`:

```markdown
- **Place Mass tool** -- two-phase workflow: select body, click to place. Move to Link and Reposition buttons. Preview circle at cursor.
```

Replace with:

```markdown
- **Place Mass tool** -- two-phase workflow: select body, click to place. While the tool is active a toolbar field sets the weight's mass; it starts at the last mass used. The drop point snaps to the grid when snapping is on (the preview circle shows where it lands), the new weight gets the next free name (W1, W2, ...) and is selected. Move to Link and Reposition buttons.
- **Weight editor** -- a selected weight shows its name, mass, owning link and body-local position in the property panel. The link editor's Weights section is always shown for a moving link and has an Add weight button (last mass used, at the link's centre of mass). Each field commits once, when the drag ends or the field loses focus (Esc cancels typing), as one undo step.
```

`docs/architecture/ARCHITECTURE.md`, point masses paragraph:

Find in `docs/architecture/ARCHITECTURE.md`:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body; clicking a marker selects that weight (`SelectedEntity::Weight`, addressed by body id and weight id). Dragging a marker previews the move and commits it on release as one `move_point_mass` (one undo step); when the nearest link within 60 px of the drop point is a different link, the weight is reattached to it at the drop point.
```

Replace with:

```markdown
Multiple point masses can attach to the same body. The GUI displays them as markers on the body; clicking a marker selects that weight (`SelectedEntity::Weight`, addressed by body id and weight id). Dragging a marker previews the move and commits it on release as one `move_point_mass` (one undo step); when the nearest link within 60 px of the drop point is a different link, the weight is reattached to it at the drop point. The property panel edits the selected weight (name, mass, owning link, body-local position) and lists every weight of the link in the link editor, with an Add weight button; each field edit is one undo step.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib -- weight_editor display_units gravity_breakdown pending_edits weight_clicks gui::tests rendering::weights plot_panel property_panel::tests
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo clippy --all-targets
```

Expected: the first command ends with `test result: ok. 105 passed; 0 failed`. 34 tests are new: 14 in `weight_editor` (7 `committed_number` tests, 7 panel tests), 5 in `display_units`, 3 name-helper tests in `gravity_breakdown`, 4 in `pending_edits`, 4 toolbar-field tests in `gui::tests`, 3 more placement tests in `weight_clicks` (next to the reworked `place_mass_click_places_the_last_mass_on_the_snapped_grid_point`, which replaces `place_mass_click_uses_the_last_point_mass`), 1 in `rendering::weights`. The full `cargo test --lib` run ends with `test result: ok. 903 passed; 0 failed` (869 before this task). Clippy adds no new warnings: against the previous task's tree the counts go from 282 to 279 (lib) and from 298 to 295 (lib test), because three `collapsible_if` warnings disappear with the replaced Point Masses block and the old `handle_place_mass`.

Then run the full gate (integration suites and the WASM check included) and restore the PNGs that the Chebyshev tests rewrite:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Expected: `GATE PASS`. If `git status` shows no change under `docs/chebyshev_lambda/`, the checkout is a no-op.

Optional mutation checks (restore the code after each). In `canvas/interaction.rs`, delete `&& !placed_weight` from the click-selection gate: `placing_a_weight_selects_it_even_when_the_snap_moves_it_off_the_pointer` and `place_mass_click_places_the_last_mass_on_the_snapped_grid_point` fail. In `property_panel/weight_editor.rs`, delete `&& edited != shown` from `committed_number`: `leaving_a_field_without_typing_commits_nothing` fails.

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add \
  linkage-sim-rs/src/gui/test_support.rs \
  linkage-sim-rs/src/gui/mod.rs \
  linkage-sim-rs/src/gui/state/display_units.rs \
  linkage-sim-rs/src/gui/state/mod.rs \
  linkage-sim-rs/src/analysis/gravity_breakdown.rs \
  linkage-sim-rs/src/gui/plot_panel/weights.rs \
  linkage-sim-rs/src/gui/plot_panel/mod.rs \
  linkage-sim-rs/src/gui/property_panel/weight_editor.rs \
  linkage-sim-rs/src/gui/property_panel/mod.rs \
  linkage-sim-rs/src/gui/property_panel/pending_edits.rs \
  linkage-sim-rs/src/gui/canvas/mod.rs \
  linkage-sim-rs/src/gui/canvas/interaction.rs \
  linkage-sim-rs/src/gui/canvas/rendering/mod.rs \
  linkage-sim-rs/src/gui/canvas/rendering/weights.rs \
  docs/ai/02-system.yaml \
  docs/ai/03-structure.yaml \
  docs/ai/05-update-tracker.md \
  docs/FEATURES.md \
  docs/architecture/ARCHITECTURE.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -m "task 8: placement and the weight editor" -m "The + Mass tool gets a toolbar mass field that starts at the last mass
used, snaps the placement to the grid, names the weight from
io::next_point_mass_id and selects it; the placing click is not also a
selection click. The property panel gains property_panel/weight_editor.rs:
a selected-weight editor (name, mass, owning link, body-local position)
and an always-shown link-editor Weights section with Add weight. Every
field commits once per edit (committed_number), one undo step each.

Shared name helpers (display_name, name_with_id, point_mass_title) live in
analysis::gravity_breakdown and the Weight Breakdown legend uses them.
gui/test_support.rs gains typed, central_panel_frame, drawn_texts,
drew_text and text_rect; the new tests and the existing property/plot
panel idle-frame helpers use them instead of local copies.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

### Task 9: Canvas readout

Each weight draws a gold marker and an arrow in the gravity direction, 12-40 px long by mass and coloured green / red / gray by `WeightBreakdown::classification` at the current pose, through the shared `canvas::classification_color` palette (Task 7). The permanent "2.00 kg" labels go away: hovering a weight shows a tooltip, and the selected weight a card next to it, with its name, mass and current force share ("-" near stroke reversal). The actuator label becomes `format_actuator_label`: "1.2 kN push, braking".

BL-026 is on main, so the sweep's actuator force is the REQUIRED force in sizing and stored-force mode alike. Push/pull therefore comes from the plotted required force and motoring/braking from the required power, whatever the element's stored force; the label test runs at every sample with stored force 0, the sample's own 50 N and a force above every required force, and `swept_lift` keeps the sample's stored force.

Every straight canvas arrow shares one head: the new `primitives::draw_arrowhead` is used by `draw_arrow` (the weight arrows), the force and external-force arrows, the Fx/Fy component arrows, the actuator line's midpoint head and the force-zone arrows, each with its old head length, so their geometry does not change. Test helpers are shared instead of copied: `test_support` gains the fixtures, and the canvas `weight_readout` tests reuse `weight_clicks::{frame, setup}` and `hit_testing::tests::weight_screen`.

**Files:**
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (`SampleMechanism` import; `drawn_line_colors`, `swept_lift`, `sample_at`, `pose_at`)
- Modify: `linkage-sim-rs/src/gui/state/mod.rs` (`actuator_label_force` reads `current_sweep_index`; new `current_sweep_index`, `actuator_label_power`)
- Modify: `linkage-sim-rs/src/gui/state/tests.rs` (append three tests)
- Modify: `linkage-sim-rs/src/gui/canvas/colors.rs` (arrow sizes after `WEIGHT_HIT_RADIUS`)
- Modify: `linkage-sim-rs/src/gui/canvas/hit_testing.rs` (`tests::weight_screen` becomes `pub(in crate::gui::canvas)`)
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs` (`weights_interactive` becomes `pub(super)`)
- Modify: `linkage-sim-rs/src/gui/canvas/mod.rs` (`weight_clicks` shares `frame`, `setup` and `weight_screen`; new `weight_readout` test module)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs` (hit_testing import; `draw_weights` replaces the point-mass block of `render_mechanism`; weight tooltip in `render_hover_tooltips`)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs` (`draw_arrowhead`, `draw_arrow`, `ARROW_HEAD_LEN_PX`; the force, external-force and Fx/Fy arrows call `draw_arrowhead`)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs` (label helpers above `force_zone_app_point_world`; the actuator label call; the force-zone arrows call `draw_arrowhead`; new test module)
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/weights.rs` (markers, arrows, readout, tooltip; tests)
- Modify docs: `docs/FEATURES.md`, `docs/architecture/ENGINEERING_OUTPUTS.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/05-update-tracker.md`

**Interfaces:**
- Consumes:
  - Task 3 (`analysis/gravity_breakdown.rs`): `pub fn gravity_vector(mech: &Mechanism) -> [f64; 2]`, `pub fn max_abs_finite(values: &[f64]) -> f64`, `pub const BRAKE_TOL_REL: f64` (1e-6), `pub enum Classification { Helping, Hurting, Neutral }` (`Debug, Clone, Copy, PartialEq, Eq`), `pub struct WeightSource { pub id: String, pub name: String, pub body_id: String, pub local_pos: [f64; 2], pub mass: f64, pub is_link_self_weight: bool }`
  - Task 4 (`gui/sweep`): `SweepData::{angles_deg: Vec<f64>, actuator_forces: Option<Vec<f64>>, actuator_power: Option<Vec<f64>>, weight_breakdown: Option<WeightBreakdown>, sweep_mode: SweepMode}`; `pub struct WeightBreakdown { pub sources: Vec<WeightSource>, pub basis: ShareBasis, pub gravity_power: Vec<Vec<f64>>, pub force_share: Vec<Vec<f64>>, pub power_share: Vec<Vec<f64>>, pub other_force: Vec<f64>, pub other_power: Vec<f64>, pub total_force: Vec<f64>, pub total_power: Vec<f64>, pub braking: Vec<bool> }`; `pub fn classification(&self, source: usize, sample: usize) -> Classification`; `pub enum ShareBasis { ActuatorForce, DriverTorque }`; `pub fn index_at_driver(&self, driver_value: f64) -> Option<usize>`; `test_support::set_actuator_stored_force(state: &mut AppState, force: f64)` (rebuilds; since BL-026 the sweep reports the required force whatever the stored force)
  - Task 1: `test_support::sorted_link_ids(state: &AppState) -> Vec<String>`
  - Task 2: `AppState::find_point_mass(&self, body_id: &str, weight_id: &str) -> Option<&PointMassJson>`, `AppState::move_point_mass(&mut self, body_id: &str, weight_id: &str, target_body: &str, local_pos: [f64; 2]) -> bool`, `AppState::set_point_mass_label(&mut self, body_id: &str, weight_id: &str, label: Option<String>) -> bool`
  - Task 5: `hit_testing::{find_point_mass_at(state: &AppState, screen_pos: Pos2, radius_px: f32) -> Option<(String, String)>, point_mass_screen_pos(state: &AppState, body_id: &str, local_pos: [f64; 2]) -> Option<Pos2>}`, `interaction::weights_interactive(state: &AppState) -> bool` (private until this task), the `hit_testing::tests::weight_screen(state: &AppState, body: &str, id: &str) -> Pos2` helper (private until this task)
  - Task 6: `AppState::weight_drag: Option<WeightDrag>` (`WeightDrag { body_id, weight_id, current_world }`)
  - Tasks 2, 5 and 6: the `weight_clicks` test module of `canvas/mod.rs`, whose helpers `frame_with`, `frame` (returns `()`), `setup`, `click`, `drag`, `press_and_drag` and its own `weight_screen` are private until this task
  - Task 7: `pub fn classification_color(class: Classification) -> Color32`, `WEIGHT_HELPING_COLOR`, `WEIGHT_HURTING_COLOR`, `WEIGHT_NEUTRAL_COLOR: Color32` (`canvas/colors.rs`)
  - Task 8: `pub fn point_mass_title(pm: &PointMassJson) -> String` (`analysis/gravity_breakdown.rs`), `pub fn format_mass_kg(kg: f64) -> String` (`gui/state/display_units.rs`, re-exported by `gui::state`), `test_support::{drawn_texts, drew_text, text_rect}`, `weights::place_mass_hint`
  - Main (BL-010, BL-026): `AppState::actuator_label_force(&self, la: &LinearActuatorElement) -> ActuatorLabelForce`, `pub enum ActuatorLabelForce { Computed(f64), Stored(f64) }`, `AppState::driver_stroke(&self) -> f64`
- Produces:
  - `pub fn format_actuator_label(force_n: f64, power_w: Option<f64>, brake_tol_w: f64) -> String`, `pub(super) fn actuator_label_text(state: &AppState, la: &LinearActuatorElement) -> String`, `pub(super) fn format_magnitude(value: f64, unit: &str) -> String` (`canvas/rendering/force_render.rs`)
  - `pub fn current_sweep_index(&self) -> Option<usize>` and `pub fn actuator_label_power(&self) -> Option<(f64, f64)>` on `AppState`
  - `canvas/rendering/weights.rs`: `pub(super) fn draw_weights(painter: &egui::Painter, state: &AppState)`, `pub(super) fn show_weight_tooltip(ui: &egui::Ui, state: &AppState, hover_pos: Pos2) -> bool`, `pub(super) fn weight_readout_lines(state: &AppState, body_id: &str, weight_id: &str) -> Option<Vec<String>>`, `pub(super) fn weight_at_pose(state: &AppState, body_id: &str, pm: &PointMassJson) -> Option<WeightAtPose>`, `pub(super) struct WeightAtPose { classification: Classification, force_share: f64, basis: ShareBasis, is_stroke: bool }`, `pub(super) fn weight_arrow_length_px(mass: f64, max_mass: f64) -> f32`, `pub(super) fn gravity_screen_dir(view: &ViewTransform, g: [f64; 2]) -> Option<Vec2>`
  - `canvas/rendering/primitives.rs`: `pub(super) fn draw_arrowhead(painter: &egui::Painter, tip: Pos2, dir: Vec2, head_len: f32, stroke: Stroke)` (the one head of every straight canvas arrow), `pub(super) fn draw_arrow(painter: &egui::Painter, tail: Pos2, tip: Pos2, stroke: Stroke)`, `pub(super) const ARROW_HEAD_LEN_PX: f32 = 8.0`
  - `canvas/colors.rs`: `pub const WEIGHT_ARROW_MIN_PX: f32 = 12.0; pub const WEIGHT_ARROW_MAX_PX: f32 = 40.0; pub const WEIGHT_ARROW_WIDTH: f32 = 2.0;`
  - `interaction::weights_interactive` is `pub(super)`; `hit_testing::tests::weight_screen` is `pub(in crate::gui::canvas)`; `weight_clicks::frame` is `pub(super)` and returns `egui::FullOutput`; `weight_clicks::setup` is `pub(super)`
  - `test_support::{drawn_line_colors(output: &egui::FullOutput) -> Vec<egui::Color32>, swept_lift() -> AppState, sample_at(state: &AppState, deg: f64) -> usize, pose_at(state: &mut AppState, deg: f64)}`

- [ ] **Step 1: Write the failing tests**

1a. `linkage-sim-rs/src/gui/test_support.rs`: the helpers. `swept_lift` keeps the sample's stored force (50 N): since BL-026 the sweep reports the required force whatever the stored force, so the fixture no longer zeroes it to dodge a residual.

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
use crate::forces::elements::ForceElement;
use crate::gui::state::AppState;
```

Replace with:

```rust
use crate::forces::elements::ForceElement;
use crate::gui::samples::SampleMechanism;
use crate::gui::state::AppState;
```

Find in `linkage-sim-rs/src/gui/test_support.rs`:

```rust
/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    let mut found = None;
    visit_shapes(output, |shape| {
        match shape {
            egui::Shape::Text(text) if found.is_none() && text.galley.text() == needle => {
                found = Some(text.galley.rect.translate(text.pos.to_vec2()));
            }
            _ => {}
        }
    });
    found
}
```

Add after it:

```rust
/// The stroke colour of every line segment egui drew in a frame.
pub(crate) fn drawn_line_colors(output: &egui::FullOutput) -> Vec<egui::Color32> {
    let mut colors = Vec::new();
    visit_shapes(output, |shape| {
        if let egui::Shape::LineSegment { stroke, .. } = shape {
            colors.push(stroke.color);
        }
    });
    colors
}

/// The robot lift of the payload spec's hands-on checklist: Parallelogram +
/// Actuator with the sample's stored force (50 N; since BL-026 the sweep
/// reports the required force whatever it is), weight W1 (50 kg) at the
/// rocker tip and W2 (20 kg) on the coupler, swept 0..=360 deg.
pub(crate) fn swept_lift() -> AppState {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ParallelogramActuator);
    assert_eq!(state.add_point_mass("rocker", 50.0, [0.0, 0.0]).as_deref(), Some("W1"));
    assert_eq!(state.add_point_mass("coupler", 20.0, [2.0, 0.0]).as_deref(), Some("W2"));
    state.compute_sweep();
    state
}

/// Index of the sweep sample at `deg` (the driver angle, in degrees).
pub(crate) fn sample_at(state: &AppState, deg: f64) -> usize {
    let sweep = state.sweep_data.as_ref().expect("sweep computed");
    sweep
        .angles_deg
        .iter()
        .position(|&a| (a - deg).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no sample at {deg} deg"))
}

/// Solve the mechanism at driver angle `deg`, so the canvas shows that pose
/// and the readouts read the sweep sample there.
pub(crate) fn pose_at(state: &mut AppState, deg: f64) {
    state.solve_at_angle(deg.to_radians());
    assert!(state.solver_status.converged, "the mechanism assembles at {deg} deg");
    assert_eq!(state.current_sweep_index(), Some(sample_at(state, deg)));
}
```

1b. `linkage-sim-rs/src/gui/state/tests.rs`: `current_sweep_index` (the sample nearest the driver, angle sweep), `actuator_label_power` (the required power and the braking tolerance at every sample, `None` without a sweep, `None` for driver-torque shares).

Find in `linkage-sim-rs/src/gui/state/tests.rs`:

```rust
        assert_eq!(seed_q_by_body_id(Some(prev), &wrong_len, &expanded), zeros);
    }
```

Add after it:

```rust
    // ── Canvas readouts at the current pose (payload weights Task 8) ─────

    #[test]
    fn current_sweep_index_is_the_sample_nearest_the_driver() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let angles = state.sweep_data.as_ref().unwrap().angles_deg.clone();
        for k in [0, 45, 90, 200] {
            state.driver_angle = angles[k].to_radians();
            assert_eq!(state.current_sweep_index(), Some(k), "at {} deg", angles[k]);
        }
        state.driver_angle = (angles[45] + 0.3).to_radians();
        assert_eq!(state.current_sweep_index(), Some(45), "nearest sample");

        state.sweep_data = None;
        assert_eq!(state.current_sweep_index(), None);
    }

    #[test]
    fn actuator_label_power_is_the_required_power_and_braking_tolerance() {
        use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};

        let mut state = crate::gui::test_support::swept_lift();
        let sweep = state.sweep_data.clone().unwrap();
        let breakdown = sweep.weight_breakdown.clone().unwrap();
        let tol = BRAKE_TOL_REL * max_abs_finite(&breakdown.total_power);
        let mut checked = 0;
        for (k, &deg) in sweep.angles_deg.iter().enumerate() {
            if deg >= 360.0 {
                continue; // the pose of 0 deg: the label reads sample 0
            }
            state.driver_angle = deg.to_radians();
            let power = breakdown.total_power[k];
            if power.is_finite() {
                assert_eq!(state.actuator_label_power(), Some((power, tol)), "{deg} deg");
                checked += 1;
            } else {
                assert_eq!(state.actuator_label_power(), None, "{deg} deg: no finite power");
            }
        }
        assert!(checked > 300, "fixture: most samples have a power ({checked})");

        state.sweep_data = None;
        assert_eq!(state.actuator_label_power(), None, "no sweep");
    }

    #[test]
    fn actuator_label_power_is_none_without_an_actuator() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_point_mass("coupler", 2.0, [0.03, 0.02]).unwrap();
        state.compute_sweep();
        assert!(state.sweep_data.as_ref().unwrap().weight_breakdown.is_some(), "fixture: driver-torque shares");
        state.driver_angle = 60.0_f64.to_radians();
        assert_eq!(state.actuator_label_power(), None);
    }
```

1c. `linkage-sim-rs/src/gui/canvas/hit_testing.rs`: the canvas tests share `weight_screen` instead of each keeping a copy.

Find in `linkage-sim-rs/src/gui/canvas/hit_testing.rs`:

```rust
    fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
```

Replace with:

```rust
    /// Screen position of weight `id` on `body` at the current pose (the
    /// canvas tests share it).
    pub(in crate::gui::canvas) fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
```

1d. `linkage-sim-rs/src/gui/canvas/mod.rs`: `weight_clicks` hands its `frame` (now returning the frame's output) and `setup` to the new `weight_readout` module and uses the shared `weight_screen`. Its own copy of `weight_screen` goes, and the place-mass hint test calls `frame` instead of `frame_with`; `frame_with` stays private.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        use crate::gui::test_support::{key_press, primary_button, primary_button_with, sorted_link_ids};
        use super::super::draw_canvas;
```

Replace with:

```rust
        use crate::gui::test_support::{key_press, primary_button, primary_button_with, sorted_link_ids};
        use super::super::draw_canvas;
        use super::super::hit_testing::tests::weight_screen;
```

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) {
            let _ = frame_with(ctx, state, events, egui::Modifiers::NONE);
        }
```

Replace with:

```rust
        /// One canvas frame with `events` and no modifiers.
        pub(super) fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> egui::FullOutput {
            frame_with(ctx, state, events, egui::Modifiers::NONE)
        }
```

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
        fn setup() -> (egui::Context, AppState, String, String) {
```

Replace with:

```rust
        pub(super) fn setup() -> (egui::Context, AppState, String, String) {
```

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            state.world_to_body_local(body, wx, wy)
        }

        /// Screen position of weight `id` on `body` at the current pose.
        fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
            let pm = state.find_point_mass(body, id).expect("weight exists");
            screen_of(state, world_of(state, body, pm.local_pos))
        }

        fn weight(body: &str, id: &str) -> SelectedEntity {
```

Replace with:

```rust
            state.world_to_body_local(body, wx, wy)
        }

        fn weight(body: &str, id: &str) -> SelectedEntity {
```

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            let output = frame_with(&ctx, &mut state, Vec::new(), egui::Modifiers::NONE);
```

Replace with:

```rust
            let output = frame(&ctx, &mut state, Vec::new());
```

Then the headless canvas tests of the readout: arrow colours at 45 / 135 / 90 degrees, a gold arrow for a weight the sweep has not seen, no arrows without gravity, no permanent labels, the hover tooltip, the selected-weight card and the actuator label words on the canvas. The new `weight_readout` module goes inside `mod tests`, after `weight_clicks` and before the closing brace of `mod tests`.

Find in `linkage-sim-rs/src/gui/canvas/mod.rs`:

```rust
            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }
    }
```

Add after it:

```rust
    /// Headless canvas frames checking the weight readout: arrows coloured
    /// at the current pose, hover tooltip, selected-weight card.
    mod weight_readout {
        use eframe::egui::{self, Pos2};

        use crate::gui::state::{AppState, SelectedEntity};
        use crate::gui::test_support::{drawn_line_colors, drawn_texts, pose_at, swept_lift};
        use super::super::colors::{
            WEIGHT_COLOR, WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR, WEIGHT_NEUTRAL_COLOR,
        };
        use super::super::hit_testing::tests::weight_screen;
        use super::weight_clicks::{frame, setup};

        /// The swept robot lift after two idle frames (fit to view, grid).
        fn lift() -> (egui::Context, AppState) {
            let mut state = swept_lift();
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state)
        }

        #[test]
        fn weight_arrows_take_the_classification_colour_of_the_current_pose() {
            let (ctx, mut state) = lift();
            for (deg, want, not) in [
                (45.0, WEIGHT_HURTING_COLOR, WEIGHT_HELPING_COLOR),
                (135.0, WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR),
            ] {
                pose_at(&mut state, deg);
                let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));
                assert!(colors.contains(&want), "{deg} deg: an arrow in {want:?}");
                assert!(!colors.contains(&not), "{deg} deg: no arrow in {not:?}");
                assert!(!colors.contains(&WEIGHT_COLOR), "{deg} deg: every weight is classified");
            }
            pose_at(&mut state, 90.0);
            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));
            assert!(colors.contains(&WEIGHT_NEUTRAL_COLOR), "90 deg: the weights move sideways");
        }

        #[test]
        fn a_weight_the_sweep_has_not_seen_gets_a_weight_coloured_arrow() {
            let (ctx, mut state) = lift();
            pose_at(&mut state, 45.0);
            // Moving W1 rebuilds; the sweep is only recomputed later (debounced).
            assert!(state.move_point_mass("rocker", "W1", "rocker", [0.05, 0.0]));

            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));

            assert!(colors.contains(&WEIGHT_COLOR), "W1's arrow waits for the new sweep");
            assert!(colors.contains(&WEIGHT_HURTING_COLOR), "W2 keeps its colour");
        }

        #[test]
        fn no_weight_arrows_without_gravity() {
            let (ctx, mut state) = lift();
            state.gravity_magnitude = 0.0;
            state.sync_gravity();
            state.compute_sweep();
            pose_at(&mut state, 45.0);

            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));

            for color in [WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR, WEIGHT_NEUTRAL_COLOR, WEIGHT_COLOR] {
                assert!(!colors.contains(&color), "no arrow in {color:?}");
            }
        }

        #[test]
        fn weights_have_no_permanent_label() {
            let (ctx, mut state, _, _) = setup();
            state.compute_sweep();
            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            assert!(!texts.iter().any(|t| t.contains("kg")), "{texts:?}");
        }

        /// Texts drawn while the pointer rests at `at`: a tooltip shows from
        /// its second frame (egui lays it out unseen first).
        fn texts_hovering(ctx: &egui::Context, state: &mut AppState, at: Pos2) -> Vec<String> {
            frame(ctx, state, vec![egui::Event::PointerMoved(at)]);
            drawn_texts(&frame(ctx, state, Vec::new()))
        }

        #[test]
        fn hovering_a_weight_shows_its_name_mass_and_share() {
            let (ctx, mut state, body, _) = setup();
            state.compute_sweep();
            let at = weight_screen(&state, &body, "W1");

            let texts = texts_hovering(&ctx, &mut state, at);

            assert!(texts.iter().any(|t| t == "W1"), "{texts:?}");
            assert!(texts.iter().any(|t| t == "Mass: 2 kg"), "{texts:?}");
            assert!(texts.iter().any(|t| t.starts_with("Torque share: ")), "four-bar: driver torque shares: {texts:?}");

            let texts = texts_hovering(&ctx, &mut state, Pos2::new(120.0, 700.0));
            assert!(!texts.iter().any(|t| t == "Mass: 2 kg"), "no tooltip away from the weight: {texts:?}");
        }

        #[test]
        fn the_selected_weight_shows_its_readout_next_to_it() {
            let (ctx, mut state, body, _) = setup();
            state.compute_sweep();
            state.selected = Some(SelectedEntity::Weight { body_id: body.clone(), weight_id: "W1".to_string() });

            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            let card = texts.iter().find(|t| t.starts_with("W1\nMass: 2 kg\n")).expect("the readout card");
            assert!(card.contains("Torque share: "), "{card:?}");

            // Hovering the selected weight adds no tooltip on top of its card.
            let at = weight_screen(&state, &body, "W1");
            let texts = texts_hovering(&ctx, &mut state, at);
            assert!(!texts.iter().any(|t| t == "Mass: 2 kg"), "{texts:?}");
            assert!(texts.iter().any(|t| t.starts_with("W1\nMass: 2 kg\n")), "the card stays: {texts:?}");
        }

        #[test]
        fn the_actuator_label_names_the_direction_and_who_does_the_work() {
            let (ctx, mut state) = lift();
            for (deg, word) in [(45.0, ", motoring"), (135.0, ", braking")] {
                pose_at(&mut state, deg);
                let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
                assert!(
                    texts.iter().any(|t| (t.contains(" push") || t.contains(" pull")) && t.ends_with(word)),
                    "{deg} deg: a label ending in {word:?} in {texts:?}"
                );
            }
        }
    }
```

1e. `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`: the label tests. `actuator_label_words_follow_the_plotted_required_force_and_power_in_every_mode` is the guard against a label that reads `F_required - F_stored`: it runs at every sample with stored force 0, the sample's 50 N and one above every required force, and asserts the words against the plotted force and power and against `WeightBreakdown::braking`.

Find in `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`:

```rust
    let t = (body_max / global_max) as f32;
    Some(heat_color(t))
}
```

Add after it:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};
    use crate::gui::test_support::{set_actuator_stored_force, swept_lift};

    /// Every force sign (push, pull, shown as zero) against every power
    /// case (motoring, braking, inside the tolerance, unknown).
    #[test]
    fn format_actuator_label_words_for_every_sign_combination() {
        let tol = 1.0;
        let cases: [(f64, Option<f64>, &str); 16] = [
            (1234.0, Some(500.0), "1.2 kN push, motoring"),
            (1234.0, Some(-500.0), "1.2 kN push, braking"),
            (-1234.0, Some(500.0), "1.2 kN pull, motoring"),
            (-1234.0, Some(-500.0), "1.2 kN pull, braking"),
            (875.0, Some(0.5), "875 N push"),
            (-875.0, Some(-0.5), "875 N pull"),
            (875.0, Some(1.0), "875 N push"),
            (875.0, Some(-1.0), "875 N push"),
            (875.0, None, "875 N push"),
            (-875.0, None, "875 N pull"),
            (875.0, Some(f64::NAN), "875 N push"),
            (0.0, Some(500.0), "0.00 N, motoring"),
            (0.004, Some(-500.0), "0.00 N, braking"),
            (-0.004, None, "0.00 N"),
            (-0.0, Some(0.0), "0.00 N"),
            (0.005, None, "0.01 N push"),
        ];
        for (force, power, want) in cases {
            assert_eq!(format_actuator_label(force, power, tol), want, "force {force}, power {power:?}");
        }
    }

    #[test]
    fn format_actuator_label_switches_units_at_the_rounding_edges() {
        let label = |f: f64| format_actuator_label(f, None, 0.0);
        assert_eq!(label(0.35), "0.35 N push");
        assert_eq!(label(9.994), "9.99 N push");
        assert_eq!(label(9.995), "10 N push");
        assert_eq!(label(999.4), "999 N push");
        assert_eq!(label(999.5), "1.0 kN push");
        assert_eq!(label(-12_345.0), "12.3 kN pull");
    }

    #[test]
    fn format_actuator_label_shows_a_dash_for_a_non_finite_force() {
        assert_eq!(format_actuator_label(f64::NAN, Some(5.0), 1.0), "-");
        assert_eq!(format_actuator_label(f64::INFINITY, None, 0.0), "-");
    }

    fn first_actuator(state: &AppState) -> LinearActuatorElement {
        state
            .mechanism
            .as_ref()
            .unwrap()
            .forces()
            .iter()
            .find_map(|f| match f {
                ForceElement::LinearActuator(la) => Some(la.clone()),
                _ => None,
            })
            .expect("the sample has a LinearActuator")
    }

    /// Since BL-026 the Actuator Force and Actuator Power plots draw the
    /// REQUIRED force and power whatever the element's stored force, and the
    /// label's words follow them: push/pull from the plotted force's sign,
    /// motoring/braking from the plotted power, i.e. the braking bands
    /// (`WeightBreakdown::braking`). Checked at every sample in sizing mode
    /// (stored force 0), with the sample's stored force, and with a stored
    /// force above every required force, where a label that read
    /// F_required - F_stored would say "pull" at every push sample.
    #[test]
    fn actuator_label_words_follow_the_plotted_required_force_and_power_in_every_mode() {
        let mut state = swept_lift();
        let shipped = first_actuator(&state).force;
        assert!(shipped > 0.0, "fixture: the sample stores a force");
        set_actuator_stored_force(&mut state, 0.0);
        state.compute_sweep();
        let required = state.sweep_data.as_ref().unwrap().actuator_forces.clone().unwrap();
        let force_tol = 1e-9 * max_abs_finite(&required);

        for stored in [0.0, shipped, 2.0 * max_abs_finite(&required)] {
            set_actuator_stored_force(&mut state, stored);
            state.compute_sweep();
            let la = first_actuator(&state);
            let sweep = state.sweep_data.clone().unwrap();
            let forces = sweep.actuator_forces.clone().unwrap();
            let power = sweep.actuator_power.clone().unwrap();
            let breakdown = sweep.weight_breakdown.clone().unwrap();
            let power_tol = BRAKE_TOL_REL * max_abs_finite(&breakdown.total_power);
            let (mut push, mut motoring, mut braking) = (0, 0, 0);
            for (k, &deg) in sweep.angles_deg.iter().enumerate() {
                if deg >= 360.0 || !forces[k].is_finite() {
                    continue; // 360 is the pose of 0; failed samples show the stored force
                }
                let what = format!("stored {stored} N, {deg} deg");
                assert!((forces[k] - required[k]).abs() <= force_tol, "{what}: plotted {} N, required {} N", forces[k], required[k]);
                state.driver_angle = deg.to_radians();
                let label = actuator_label_text(&state, &la);
                let what = format!("{what}: {label:?}");
                assert_eq!(label.contains(" push"), forces[k] >= SHOWN_AS_ZERO_N, "{what}");
                assert_eq!(label.contains(" pull"), forces[k] <= -SHOWN_AS_ZERO_N, "{what}");
                assert_eq!(label.ends_with(", braking"), breakdown.braking[k], "{what}: the braking bands");
                assert_eq!(label.ends_with(", motoring"), breakdown.total_power[k] > power_tol, "{what}");
                if power[k].is_finite() && power[k].abs() > 2.0 * power_tol {
                    assert_eq!(label.ends_with(", motoring"), power[k] > 0.0, "{what}: plotted power {} W", power[k]);
                    assert_eq!(label.ends_with(", braking"), power[k] < 0.0, "{what}: plotted power {} W", power[k]);
                }
                push += usize::from(label.contains(" push"));
                motoring += usize::from(label.ends_with(", motoring"));
                braking += usize::from(label.ends_with(", braking"));
            }
            assert!(
                push > 10 && motoring > 10 && braking > 10,
                "fixture, stored {stored} N: the lift pushes ({push}), motors ({motoring}) and brakes ({braking})"
            );
        }
    }

    #[test]
    fn actuator_label_falls_back_to_the_stored_force_without_a_sweep() {
        let mut state = swept_lift();
        let la = first_actuator(&state);
        state.sweep_data = None;
        assert_eq!(actuator_label_text(&state, &la), format!("{} (stored)", format_actuator_label(la.force, None, 0.0)));
        assert_eq!(actuator_label_text(&state, &la), "50 N push (stored)", "the sample stores 50 N");
    }
}
```

1f. `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`: tests of the arrow length, the gravity direction under mounting angles, `weight_at_pose` against the breakdown (including entries the sweep predates), the readout lines (including the reversal dash and driver-torque shares) and the share format. The new tests go inside `mod tests`, after `place_mass_hint_names_the_next_weight_and_the_field_mass` and before the closing brace of `mod tests`.

Find in `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`:

```rust
    use super::*;
    use crate::gui::samples::SampleMechanism;
```

Replace with:

```rust
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::sweep::WeightBreakdown;
    use crate::gui::test_support::{sample_at, swept_lift};
    use Classification::{Helping, Hurting, Neutral};
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`:

```rust
            "W1 is taken, and adding it made 1.25 kg the last mass used"
        );
    }
```

Add after it:

```rust
    #[test]
    fn weight_arrow_length_scales_linearly_up_to_the_heaviest_weight() {
        assert_eq!(weight_arrow_length_px(50.0, 50.0), WEIGHT_ARROW_MAX_PX);
        assert_eq!(weight_arrow_length_px(25.0, 50.0), WEIGHT_ARROW_MAX_PX / 2.0);
        assert_eq!(weight_arrow_length_px(1.0, 50.0), WEIGHT_ARROW_MIN_PX, "light weights keep a visible arrow");
        assert_eq!(weight_arrow_length_px(80.0, 50.0), WEIGHT_ARROW_MAX_PX, "never longer than the maximum");
        for (mass, max) in [(0.0, 50.0), (-1.0, 50.0), (f64::NAN, 50.0), (5.0, 0.0), (5.0, f64::INFINITY)] {
            assert_eq!(weight_arrow_length_px(mass, max), WEIGHT_ARROW_MIN_PX, "mass {mass}, max {max}");
        }
    }

    fn assert_down(dir: Option<Vec2>, what: &str) {
        let dir = dir.unwrap_or_else(|| panic!("{what}: gravity has a direction"));
        assert!(dir.x.abs() < 1e-5 && (dir.y - 1.0).abs() < 1e-5, "{what}: {dir:?} should point down the screen");
    }

    /// The GUI rotates gravity by the mounting angle (`sync_gravity`) and the
    /// view rotates the world by the same angle, so on screen the arrows
    /// always point straight down.
    #[test]
    fn gravity_points_down_the_screen_whatever_the_mounting_angle() {
        let mut view = ViewTransform::default();
        assert_down(gravity_screen_dir(&view, [0.0, -9.81]), "level");
        for theta in [0.3_f64, -1.2, std::f64::consts::PI] {
            view.mounting_angle = theta;
            let g = [-9.81 * theta.sin(), -9.81 * theta.cos()];
            assert_down(gravity_screen_dir(&view, g), &format!("mounting angle {theta}"));
        }
        view.mounting_angle = 0.3;
        let unrotated = gravity_screen_dir(&view, [0.0, -9.81]).unwrap();
        assert!(unrotated.x.abs() > 0.1, "a world vector turns with the view: {unrotated:?}");
    }

    #[test]
    fn gravity_screen_dir_is_none_when_gravity_is_off() {
        let view = ViewTransform::default();
        assert_eq!(gravity_screen_dir(&view, [0.0, 0.0]), None);
        assert_eq!(gravity_screen_dir(&view, [f64::NAN, -9.81]), None);
    }

    fn breakdown(state: &AppState) -> &WeightBreakdown {
        state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().unwrap()
    }

    fn source_index(b: &WeightBreakdown, id: &str) -> usize {
        b.sources.iter().position(|s| s.id == id).unwrap()
    }

    /// Put the driver at `deg` (the readouts only read the sweep sample).
    fn at_deg(state: &mut AppState, deg: f64) -> usize {
        state.driver_angle = deg.to_radians();
        sample_at(state, deg)
    }

    /// A sample where W1's force share is NaN (stroke reversal).
    fn reversal_sample(state: &AppState) -> usize {
        let b = breakdown(state);
        let angles = &state.sweep_data.as_ref().unwrap().angles_deg;
        (0..angles.len())
            .find(|&k| b.force_share[source_index(b, "W1")][k].is_nan() && b.gravity_power[0][k].is_finite())
            .expect("fixture: the lift reverses its stroke")
    }

    #[test]
    fn weight_at_pose_reads_the_breakdown_at_the_current_sample() {
        let mut state = swept_lift();
        let reversal_deg = state.sweep_data.as_ref().unwrap().angles_deg[reversal_sample(&state)];
        for (deg, want) in [(45.0, Some(Hurting)), (135.0, Some(Helping)), (90.0, Some(Neutral)), (reversal_deg, None)] {
            let k = at_deg(&mut state, deg);
            for (body, id) in [("rocker", "W1"), ("coupler", "W2")] {
                let pm = state.find_point_mass(body, id).unwrap().clone();
                let at = weight_at_pose(&state, body, &pm).unwrap_or_else(|| panic!("{id} at {deg} deg"));
                let b = breakdown(&state);
                let i = source_index(b, id);
                assert_eq!(at.classification, b.classification(i, k), "{id} at {deg} deg");
                let share = b.force_share[i][k];
                assert!(
                    at.force_share == share || (at.force_share.is_nan() && share.is_nan()),
                    "{id} at {deg} deg: {} vs {share}",
                    at.force_share
                );
                assert_eq!(at.basis, ShareBasis::ActuatorForce);
                assert!(!at.is_stroke);
                if let Some(want) = want {
                    assert_eq!(at.classification, want, "{id} at {deg} deg (lifting hurts, lowering helps, sideways is neutral)");
                }
            }
        }
    }

    #[test]
    fn weight_at_pose_is_none_without_a_matching_breakdown_entry() {
        let mut state = swept_lift();
        at_deg(&mut state, 45.0);
        let pm = state.find_point_mass("rocker", "W1").unwrap().clone();
        assert!(weight_at_pose(&state, "rocker", &pm).is_some());

        let moved = PointMassJson { local_pos: [0.1, 0.0], ..pm.clone() };
        assert_eq!(weight_at_pose(&state, "rocker", &moved), None, "the sweep predates the move");
        let heavier = PointMassJson { mass: 60.0, ..pm.clone() };
        assert_eq!(weight_at_pose(&state, "rocker", &heavier), None, "the sweep predates the mass edit");
        assert_eq!(weight_at_pose(&state, "coupler", &pm), None, "not on that link");

        state.sweep_data = None;
        assert_eq!(weight_at_pose(&state, "rocker", &pm), None, "no sweep");
    }

    #[test]
    fn readout_lines_show_the_title_mass_and_current_share() {
        let mut state = swept_lift();
        let k = at_deg(&mut state, 45.0);
        let share = breakdown(&state).force_share[source_index(breakdown(&state), "W1")][k];
        assert!(share.is_finite(), "fixture: a defined share at 45 deg");
        assert_eq!(
            weight_readout_lines(&state, "rocker", "W1").unwrap(),
            vec!["W1".to_string(), "Mass: 50 kg".to_string(), format!("Force share: {} (hurting)", format_share(share, "N"))]
        );

        // A name edit rebuilds without a new sweep: the share still shows.
        assert!(state.set_point_mass_label("rocker", "W1", Some("Robot torso".to_string())));
        let lines = weight_readout_lines(&state, "rocker", "W1").unwrap();
        assert_eq!(lines[0], "Robot torso (W1)");
        assert!(lines[2].ends_with("(hurting)"), "{lines:?}");

        let k = reversal_sample(&state);
        let deg = state.sweep_data.as_ref().unwrap().angles_deg[k];
        at_deg(&mut state, deg);
        let lines = weight_readout_lines(&state, "rocker", "W1").unwrap();
        assert!(lines[2].starts_with("Force share: - ("), "near stroke reversal: {lines:?}");

        state.sweep_data = None;
        assert_eq!(weight_readout_lines(&state, "rocker", "W1").unwrap()[2], "Force share: -");
        assert_eq!(weight_readout_lines(&state, "rocker", "W9"), None);
    }

    #[test]
    fn readout_shows_driver_torque_shares_without_an_actuator() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_point_mass("coupler", 2.0, [0.03, 0.02]).unwrap();
        state.compute_sweep();
        let k = at_deg(&mut state, 60.0);
        let b = breakdown(&state);
        let i = source_index(b, "W1");
        let want = format!(
            "Torque share: {} ({})",
            format_share(b.force_share[i][k], "N\u{00b7}m"),
            classification_word(b.classification(i, k))
        );
        assert_eq!(weight_readout_lines(&state, "coupler", "W1").unwrap()[2], want);
    }

    #[test]
    fn format_share_signs_the_value_and_shows_a_dash_when_undefined() {
        assert_eq!(format_share(874.2, "N"), "+874 N");
        assert_eq!(format_share(-1234.0, "N"), "-1.2 kN");
        assert_eq!(format_share(0.35, "N\u{00b7}m"), "+0.35 N\u{00b7}m");
        assert_eq!(format_share(0.0, "N"), "+0.00 N");
        assert_eq!(format_share(f64::NAN, "N"), "-");
        assert_eq!(format_share(f64::NEG_INFINITY, "N"), "-");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

From `linkage-sim-rs/`:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib --no-run
```

Expected: the lib test build fails with 61 errors (`could not compile linkage-sim-rs (lib test)`): E0425 `cannot find function` for `format_actuator_label`, `actuator_label_text`, `weight_arrow_length_px`, `gravity_screen_dir`, `weight_at_pose`, `weight_readout_lines`, `format_share` and `classification_word`; E0425 `cannot find value` for `SHOWN_AS_ZERO_N`, `WEIGHT_ARROW_MAX_PX` and `WEIGHT_ARROW_MIN_PX`; E0599 `no method named current_sweep_index found for struct AppState` (from `pose_at`, the state tests and the readout tests) and the same for `actuator_label_power`; and in the `weights.rs` tests the unresolved `Vec2`, `ViewTransform`, `ShareBasis`, `PointMassJson` and `Classification` (E0412, E0433, E0422, E0432). Nothing runs until the implementation exists.

- [ ] **Step 3: Implement**

3a. `linkage-sim-rs/src/gui/state/mod.rs`: `actuator_label_force` reads the sample through the new `current_sweep_index` (the driver-angle or stroke lookup that used to live inline), and the label's power comes from the sweep's breakdown with the braking bands' tolerance.

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
        let computed = self.sweep_data.as_ref().and_then(|sweep| {
            let forces = sweep.actuator_forces.as_ref()?;
            let driver_value = if sweep.sweep_mode.is_stroke() {
                self.driver_stroke()
            } else {
                self.driver_angle
            };
            let f = *forces.get(sweep.index_at_driver(driver_value)?)?;
            f.is_finite().then_some(f)
        });
```

Replace with:

```rust
        let computed = self.sweep_data.as_ref().and_then(|sweep| {
            let f = *sweep.actuator_forces.as_ref()?.get(self.current_sweep_index()?)?;
            f.is_finite().then_some(f)
        });
```

Find in `linkage-sim-rs/src/gui/state/mod.rs`:

```rust
        match computed {
            Some(f) => ActuatorLabelForce::Computed(f),
            None => ActuatorLabelForce::Stored(la.force),
        }
    }
```

Add after it:

```rust
    /// Index of the sweep sample nearest the current driver parameter: the
    /// driver angle in an angle sweep, the stroke in a stroke sweep. It is
    /// the sample the plot cursor marks and the one every canvas readout
    /// reads (actuator label, weight arrows and weight readouts). `None`
    /// without a sweep or without a finite sample.
    pub fn current_sweep_index(&self) -> Option<usize> {
        let sweep = self.sweep_data.as_ref()?;
        let driver_value = if sweep.sweep_mode.is_stroke() { self.driver_stroke() } else { self.driver_angle };
        sweep.index_at_driver(driver_value)
    }

    /// Required actuator power (W) at the current pose and the braking
    /// tolerance, `(total_power[k], BRAKE_TOL_REL * max|total_power|)`, from
    /// the sweep's weight breakdown. The canvas actuator label calls power
    /// above the tolerance motoring and below minus it braking: the rule of
    /// `WeightBreakdown::braking` and the plots' braking bands. `None` when
    /// the sweep has no breakdown, its shares are driver shares (no
    /// actuator), or the sample's power is not finite.
    pub fn actuator_label_power(&self) -> Option<(f64, f64)> {
        use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};

        let breakdown = self.sweep_data.as_ref()?.weight_breakdown.as_ref()?;
        if breakdown.basis != crate::gui::sweep::ShareBasis::ActuatorForce {
            return None;
        }
        let power = *breakdown.total_power.get(self.current_sweep_index()?)?;
        power
            .is_finite()
            .then(|| (power, BRAKE_TOL_REL * max_abs_finite(&breakdown.total_power)))
    }
```

3b. `linkage-sim-rs/src/gui/canvas/colors.rs`: the arrow sizes. They sit directly below `WEIGHT_HIT_RADIUS`, with no blank line.

Find in `linkage-sim-rs/src/gui/canvas/colors.rs`:

```rust
pub const WEIGHT_HIT_RADIUS: f32 = 8.0;
```

Add after it:

```rust
/// Length range (screen px) of a weight's gravity arrow: the heaviest weight
/// gets the maximum, lighter ones scale down with mass to the minimum.
pub const WEIGHT_ARROW_MIN_PX: f32 = 12.0;
pub const WEIGHT_ARROW_MAX_PX: f32 = 40.0;
pub const WEIGHT_ARROW_WIDTH: f32 = 2.0;
```

3c. `linkage-sim-rs/src/gui/canvas/interaction.rs`: the tooltip in `weights.rs` uses the same gate as the hover ring.

Find in `linkage-sim-rs/src/gui/canvas/interaction.rs`:

```rust
fn weights_interactive(state: &AppState) -> bool {
```

Replace with:

```rust
pub(super) fn weights_interactive(state: &AppState) -> bool {
```

3d. `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs`: `draw_arrowhead` is the one head of every straight canvas arrow. `draw_arrow` (shaft plus a head clamped to the arrow length) is what the weight arrows use. The force and external-force arrows and the Fx/Fy components keep their 8 px head (`ARROW_HEAD_LEN_PX`) through `draw_arrowhead` and are not routed through `draw_arrow`, whose clamp would shrink the heads of arrows as short as `FORCE_ARROW_MIN_PX` (3 px). The torque arc's head has a different angle (0.5 rad, 5 px) and keeps its own code.

Find in `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs`:

```rust
    // Shaft.
    painter.line_segment(
        [tail, tip],
        Stroke::new(FORCE_ARROW_WIDTH, EXT_FORCE_COLOR),
    );

    // Arrowhead.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44;
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment(
            [tip, head_end],
            Stroke::new(FORCE_ARROW_WIDTH, EXT_FORCE_COLOR),
        );
    }
```

Replace with:

```rust
    // Shaft and arrowhead (a full-size head even on a short arrow).
    let stroke = Stroke::new(FORCE_ARROW_WIDTH, EXT_FORCE_COLOR);
    painter.line_segment([tail, tip], stroke);
    draw_arrowhead(painter, tip, Vec2::new(dx, dy), ARROW_HEAD_LEN_PX, stroke);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs`:

```rust
/// Draw a force arrow at a joint location.
```

Replace with:

```rust
/// Default length (px) and half-angle (rad, about 25 degrees) of an
/// arrowhead: force arrows, the actuator line, weight arrows.
pub(super) const ARROW_HEAD_LEN_PX: f32 = 8.0;
const ARROW_HEAD_HALF_ANGLE: f32 = 0.44;

/// A two-line arrowhead at `tip` pointing along the unit screen vector
/// `dir`: two `head_len` px lines at +/-`ARROW_HEAD_HALF_ANGLE` from the
/// shaft, drawn back from the tip. The one arrowhead of every straight
/// canvas arrow; the caller draws the shaft.
pub(super) fn draw_arrowhead(painter: &egui::Painter, tip: Pos2, dir: Vec2, head_len: f32, stroke: Stroke) {
    let back = -dir;
    for angle in [-ARROW_HEAD_HALF_ANGLE, ARROW_HEAD_HALF_ANGLE] {
        let (sin, cos) = angle.sin_cos();
        let head = Vec2::new(back.x * cos - back.y * sin, back.x * sin + back.y * cos);
        painter.line_segment([tip, tip + head * head_len], stroke);
    }
}

/// A straight arrow from `tail` to `tip` with a two-line head at the tip,
/// never longer than the arrow. Nothing for a zero-length or non-finite
/// arrow.
pub(super) fn draw_arrow(painter: &egui::Painter, tail: Pos2, tip: Pos2, stroke: Stroke) {
    let delta = tip - tail;
    let length = delta.length();
    if !(length.is_finite() && length > 0.0) {
        return;
    }
    painter.line_segment([tail, tip], stroke);
    draw_arrowhead(painter, tip, delta / length, ARROW_HEAD_LEN_PX.min(length), stroke);
}

/// Draw a force arrow at a joint location.
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs`:

```rust
    // Shaft line.
    painter.line_segment(
        [origin, tip],
        Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR),
    );

    // Arrowhead: two lines at +/-25 degrees from the shaft, 8px long.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44; // ~25 degrees in radians
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment(
            [tip, head_end],
            Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR),
        );
    }
```

Replace with:

```rust
    // Shaft and arrowhead (a full-size head even on a short arrow).
    let stroke = Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR);
    painter.line_segment([origin, tip], stroke);
    draw_arrowhead(painter, tip, Vec2::new(dx, dy), ARROW_HEAD_LEN_PX, stroke);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/primitives.rs`:

```rust
    // Solid arrowhead in both modes — keeps the arrow legible regardless
    // of dash phase at the tip.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44;
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment([tip, head_end], stroke);
    }
```

Replace with:

```rust
    // Solid arrowhead in both modes — keeps the arrow legible regardless
    // of dash phase at the tip.
    draw_arrowhead(painter, tip, Vec2::new(dx, dy), ARROW_HEAD_LEN_PX, stroke);
```

3e. `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`: the label helpers (`format_magnitude`, `SHOWN_AS_ZERO_N`, `format_actuator_label`, `actuator_label_text`), the actuator label call, and the actuator line's midpoint head and the force-zone arrows through `draw_arrowhead` (the zone keeps its 6 px head).

Find in `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`:

```rust
/// Compute the current world-space application point of a `ForceZoneElement`.
```

Replace with:

```rust
/// A magnitude (`value` >= 0) in `unit` for a canvas label: one decimal in
/// k<unit> from 999.5 ("1.2 kN"), whole units from 9.995 ("875 N"), two
/// decimals below ("0.35 N").
pub(super) fn format_magnitude(value: f64, unit: &str) -> String {
    if value >= 999.5 {
        format!("{:.1} k{unit}", value / 1000.0)
    } else if value >= 9.995 {
        format!("{value:.0} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

/// A force (N) whose size is below this shows as "0.00 N" and gets no
/// push/pull word.
const SHOWN_AS_ZERO_N: f64 = 0.005;

/// Canvas actuator label: force magnitude, push or pull, motoring or
/// braking, e.g. "1.2 kN push, braking".
///
/// Push/pull follows the element's sign convention (positive = extension,
/// `forces/elements/evaluation.rs`); a force that shows as zero gets no
/// direction. `power_w` is the required actuator power at this pose
/// (`AppState::actuator_label_power`): motoring above `brake_tol_w`,
/// braking below `-brake_tol_w` (the braking bands' rule), no word in
/// between (the actuator is momentarily still) or when it is `None` or not
/// finite. A non-finite force shows as "-".
pub fn format_actuator_label(force_n: f64, power_w: Option<f64>, brake_tol_w: f64) -> String {
    if !force_n.is_finite() {
        return "-".to_string();
    }
    let mut label = format_magnitude(force_n.abs(), "N");
    if force_n >= SHOWN_AS_ZERO_N {
        label.push_str(" push");
    } else if force_n <= -SHOWN_AS_ZERO_N {
        label.push_str(" pull");
    }
    match power_w {
        Some(p) if p > brake_tol_w => label.push_str(", motoring"),
        Some(p) if p < -brake_tol_w => label.push_str(", braking"),
        _ => {}
    }
    label
}

/// Text of the canvas label of actuator `la`: the statics force the
/// Actuator Force plot draws at this pose (`AppState::actuator_label_force`)
/// with motoring/braking from `AppState::actuator_label_power`; without a
/// usable sweep sample, the element's stored force marked "(stored)".
pub(super) fn actuator_label_text(state: &AppState, la: &LinearActuatorElement) -> String {
    match state.actuator_label_force(la) {
        ActuatorLabelForce::Computed(force) => match state.actuator_label_power() {
            Some((power, tol)) => format_actuator_label(force, Some(power), tol),
            None => format_actuator_label(force, None, 0.0),
        },
        ActuatorLabelForce::Stored(force) => format!("{} (stored)", format_actuator_label(force, None, 0.0)),
    }
}

/// Compute the current world-space application point of a `ForceZoneElement`.
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`:

```rust
                    painter.line_segment(
                        [start, end],
                        Stroke::new(2.0, ACTUATOR_COLOR),
                    );
                    // Arrowhead at midpoint pointing A -> B.
                    let mid = Pos2::new(
                        start.x + delta.x * 0.5,
                        start.y + delta.y * 0.5,
                    );
                    let head_len = 8.0_f32;
                    let head_angle = 0.44_f32;
                    let back_dx = -dir.x;
                    let back_dy = -dir.y;
                    for sign in [-1.0_f32, 1.0] {
                        let cos_a = head_angle.cos();
                        let sin_a = head_angle.sin() * sign;
                        let hx = back_dx * cos_a - back_dy * sin_a;
                        let hy = back_dx * sin_a + back_dy * cos_a;
                        let head_end = Pos2::new(mid.x + hx * head_len, mid.y + hy * head_len);
                        painter.line_segment(
                            [mid, head_end],
                            Stroke::new(2.0, ACTUATOR_COLOR),
                        );
                    }
                    // Force magnitude label: the same statics sample the
                    // Actuator Force plot draws at this pose, or the stored
                    // force when no sweep value is available. Lookup lives in
                    // `AppState::actuator_label_force` (unit-tested).
                    let label = match state.actuator_label_force(la) {
                        ActuatorLabelForce::Computed(f) => format!("{:.0} N (computed)", f),
                        ActuatorLabelForce::Stored(f) => format!("{:.0} N", f),
                    };
```

Replace with:

```rust
                    let stroke = Stroke::new(2.0, ACTUATOR_COLOR);
                    painter.line_segment([start, end], stroke);
                    // Arrowhead at midpoint pointing A -> B.
                    let mid = start + delta * 0.5;
                    draw_arrowhead(painter, mid, dir, ARROW_HEAD_LEN_PX, stroke);
                    // Force label: magnitude, push/pull, motoring/braking
                    // (`actuator_label_text`, unit-tested).
                    let label = actuator_label_text(state, la);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`:

```rust
        let head_len = 6.0_f32;
        let head_angle = 0.44_f32;
```

Replace with:

```rust
        let head_len = 6.0_f32;
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`:

```rust
            // Arrow shaft.
            painter.line_segment([start, tip], Stroke::new(1.5, FORCE_ZONE_COLOR));

            // Arrowhead: two small lines forming a V at the tip.
            let back_x = -dir_x;
            let back_y = -dir_y;
            for sign in [-1.0_f32, 1.0] {
                let cos_a = head_angle.cos();
                let sin_a = head_angle.sin() * sign;
                let hx = back_x * cos_a - back_y * sin_a;
                let hy = back_x * sin_a + back_y * cos_a;
                let head_pt = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
                painter.line_segment([tip, head_pt], Stroke::new(1.5, FORCE_ZONE_COLOR));
            }
```

Replace with:

```rust
            // Arrow shaft and a small arrowhead at the tip.
            let stroke = Stroke::new(1.5, FORCE_ZONE_COLOR);
            painter.line_segment([start, tip], stroke);
            draw_arrowhead(painter, tip, Vec2::new(dir_x, dir_y), head_len, stroke);
```

3f. `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`: the module now draws the weights. `weight_at_pose` only trusts a breakdown source with the weight's current id, body, position and mass (rebuilds keep the old sweep until the debounced recompute), otherwise the arrow keeps the weight colour and the share reads "-". `gravity_screen_dir` maps `g` through the view, so the arrows point straight down on screen under any mounting angle.

Find in `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`:

```rust
//! Canvas weights (point masses): placement hint.

use crate::gui::state::{format_mass_kg, AppState};
use crate::io::next_point_mass_id;
```

Replace with:

```rust
//! Canvas weights (point masses): markers, gravity arrows, the hover and
//! selection readout, and the placement hint (payload weights, spec Track 2
//! section 3, "Canvas readout at the current pose").
//!
//! Each weight draws its marker and an arrow in the gravity direction whose
//! length grows with its mass, coloured by whether the weight helps (green),
//! hurts (red) or is neutral (gray) at the current pose:
//! `WeightBreakdown::classification` at `AppState::current_sweep_index`,
//! through `canvas::classification_color`, the palette the Weight Breakdown
//! plot uses. Before the sweep has a breakdown for the weight the arrow is
//! drawn in the weight colour. Hovering a weight shows its name, mass and
//! current force share in a tooltip; the selected weight shows the same
//! readout next to its marker. There are no permanent labels.

use eframe::egui::{self, Pos2, Stroke, Vec2};

use crate::analysis::gravity_breakdown::{gravity_vector, point_mass_title, Classification};
use crate::core::state::GROUND_ID;
use crate::gui::state::{format_mass_kg, AppState, SelectedEntity, ViewTransform};
use crate::gui::sweep::ShareBasis;
use crate::io::{next_point_mass_id, PointMassJson};

use super::super::colors::*;
use super::super::hit_testing::{find_point_mass_at, point_mass_screen_pos};
use super::super::interaction::weights_interactive;
use super::draw_pill_label;
use super::force_render::format_magnitude;
use super::primitives::draw_arrow;
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/weights.rs`:

```rust
    format!(
        "Click to place weight {next_id} ({}) on '{body_id}' (Esc to cancel)",
        format_mass_kg(state.last_point_mass_kg)
    )
}
```

Add after it:

```rust
/// Length (px) of the gravity arrow of a weight of `mass` when the heaviest
/// drawn weight is `max_mass`: `WEIGHT_ARROW_MAX_PX * mass / max_mass`, at
/// least `WEIGHT_ARROW_MIN_PX` so a light weight's arrow stays visible. The
/// minimum when either mass is not positive and finite.
pub(super) fn weight_arrow_length_px(mass: f64, max_mass: f64) -> f32 {
    if !(mass.is_finite() && mass > 0.0 && max_mass.is_finite() && max_mass > 0.0) {
        return WEIGHT_ARROW_MIN_PX;
    }
    let fraction = (mass / max_mass).min(1.0) as f32;
    (WEIGHT_ARROW_MAX_PX * fraction).max(WEIGHT_ARROW_MIN_PX)
}

/// Unit screen direction of the gravity vector `g` (world, m/s^2) in
/// `view`, which rotates world directions by the mounting angle and flips
/// y. `None` when gravity is off or not finite.
pub(super) fn gravity_screen_dir(view: &ViewTransform, g: [f64; 2]) -> Option<Vec2> {
    let norm = g[0].hypot(g[1]);
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    let [x0, y0] = view.world_to_screen(0.0, 0.0);
    let [x1, y1] = view.world_to_screen(g[0] / norm, g[1] / norm);
    let dir = Vec2::new(x1 - x0, y1 - y0);
    let length = dir.length();
    (length.is_finite() && length > 0.0).then(|| dir / length)
}

/// Where a weight stands at the current pose, read from the sweep's
/// weight breakdown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct WeightAtPose {
    pub(super) classification: Classification,
    /// Force share at the current sample; NaN near stroke reversal.
    pub(super) force_share: f64,
    pub(super) basis: ShareBasis,
    pub(super) is_stroke: bool,
}

/// The breakdown entry of weight `pm` on `body_id` at the current sweep
/// sample (`AppState::current_sweep_index`). `None` without a breakdown or
/// a current sample, and when the breakdown does not have the weight with
/// its current mass and position (the sweep is recomputed after an edit).
pub(super) fn weight_at_pose(state: &AppState, body_id: &str, pm: &PointMassJson) -> Option<WeightAtPose> {
    let sweep = state.sweep_data.as_ref()?;
    let breakdown = sweep.weight_breakdown.as_ref()?;
    let sample = state.current_sweep_index()?;
    let source = breakdown.sources.iter().position(|s| {
        !s.is_link_self_weight
            && s.id == pm.id
            && s.body_id == body_id
            && s.local_pos == pm.local_pos
            && s.mass == pm.mass
    })?;
    Some(WeightAtPose {
        classification: breakdown.classification(source, sample),
        force_share: *breakdown.force_share.get(source)?.get(sample)?,
        basis: breakdown.basis,
        is_stroke: sweep.sweep_mode.is_stroke(),
    })
}

fn classification_word(class: Classification) -> &'static str {
    match class {
        Classification::Helping => "helping",
        Classification::Hurting => "hurting",
        Classification::Neutral => "neutral",
    }
}

/// Readout name and unit of a force share: actuator force (N), driver
/// torque (N*m) with a revolute driver, driver force (N) with a linear one.
fn share_name_and_unit(basis: ShareBasis, is_stroke: bool) -> (&'static str, &'static str) {
    match basis {
        ShareBasis::ActuatorForce => ("Force share", "N"),
        ShareBasis::DriverTorque if is_stroke => ("Force share", "N"),
        ShareBasis::DriverTorque => ("Torque share", "N\u{00b7}m"),
    }
}

/// A signed share for the readout ("+874 N", "-1.2 kN"); "-" when it is
/// not finite (near stroke reversal).
fn format_share(value: f64, unit: &str) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }
    let sign = if value < 0.0 { "-" } else { "+" };
    format!("{sign}{}", format_magnitude(value.abs(), unit))
}

/// Readout of weight `weight_id` on `body_id` at the current pose: its
/// title ("W1", "Robot (W1)"), its mass, and its current force share with
/// its classification ("Force share: +874 N (helping)"). The share reads
/// "-" near stroke reversal and while the sweep has no breakdown for the
/// weight. `None` when the weight does not exist.
pub(super) fn weight_readout_lines(state: &AppState, body_id: &str, weight_id: &str) -> Option<Vec<String>> {
    let pm = state.find_point_mass(body_id, weight_id)?;
    let share = match weight_at_pose(state, body_id, pm) {
        Some(at) => {
            let (name, unit) = share_name_and_unit(at.basis, at.is_stroke);
            format!("{name}: {} ({})", format_share(at.force_share, unit), classification_word(at.classification))
        }
        None => "Force share: -".to_string(),
    };
    Some(vec![point_mass_title(pm), format!("Mass: {}", format_mass_kg(pm.mass)), share])
}

/// Draw every weight on a moving link: its gravity arrow (see the module
/// docs), its marker (faded while it is dragged, ringed while selected)
/// and, on top, the readout of the selected weight. Weights whose screen
/// position is not finite are skipped.
pub(super) fn draw_weights(painter: &egui::Painter, state: &AppState) {
    let Some(bp) = &state.blueprint else { return };
    let gravity_dir = state.mechanism.as_ref().and_then(|m| gravity_screen_dir(&state.view, gravity_vector(m)));
    let max_mass = bp
        .bodies
        .iter()
        .filter(|(id, _)| id.as_str() != GROUND_ID)
        .flat_map(|(_, body)| body.point_masses.iter().map(|pm| pm.mass))
        .filter(|m| m.is_finite() && *m > 0.0)
        .fold(0.0, f64::max);
    let marker_color = state.nc(WEIGHT_COLOR);
    let selected_ring = Stroke::new(2.0, state.nc(BODY_SELECTED_COLOR));

    for (body_id, body) in &bp.bodies {
        if body_id == GROUND_ID {
            continue;
        }
        for pm in &body.point_masses {
            let Some(center) = point_mass_screen_pos(state, body_id, pm.local_pos) else { continue };
            // A weight being dragged fades where it is; the drag preview
            // (canvas interaction) draws it at the drop point.
            let dragged = state.weight_drag.as_ref().is_some_and(|d| d.body_id == *body_id && d.weight_id == pm.id);
            let fade = |c: egui::Color32| if dragged { c.linear_multiply(0.35) } else { c };
            if let Some(dir) = gravity_dir {
                let class_color = weight_at_pose(state, body_id, pm)
                    .map_or(WEIGHT_COLOR, |at| classification_color(at.classification));
                let tail = center + dir * WEIGHT_RADIUS;
                let tip = tail + dir * weight_arrow_length_px(pm.mass, max_mass);
                draw_arrow(painter, tail, tip, Stroke::new(WEIGHT_ARROW_WIDTH, fade(state.nc(class_color))));
            }
            painter.circle_filled(center, WEIGHT_RADIUS, fade(marker_color));
            let entity = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: pm.id.clone() };
            if state.selected.as_ref() == Some(&entity) || state.multi_selected.contains(&entity) {
                painter.circle_stroke(center, WEIGHT_RADIUS + 3.0, selected_ring);
            }
        }
    }

    if let Some(SelectedEntity::Weight { body_id, weight_id }) = &state.selected {
        let center = state
            .find_point_mass(body_id, weight_id)
            .and_then(|pm| point_mass_screen_pos(state, body_id, pm.local_pos));
        if let (Some(center), Some(lines)) = (center, weight_readout_lines(state, body_id, weight_id)) {
            let offset = WEIGHT_HIT_RADIUS + 4.0;
            draw_pill_label(painter, center + Vec2::new(offset, -offset), &lines.join("\n"), marker_color, egui::Align2::LEFT_BOTTOM);
        }
    }
}

/// Hover readout: a tooltip with the readout of the weight under
/// `hover_pos`, while weights answer the pointer (plain Select mode, no
/// weight drag). The selected weight already shows its readout on the
/// canvas and gets no tooltip. Returns true when a weight is under the
/// pointer, so the caller shows no joint or link tooltip beneath it.
pub(super) fn show_weight_tooltip(ui: &egui::Ui, state: &AppState, hover_pos: Pos2) -> bool {
    if !weights_interactive(state) || state.weight_drag.is_some() {
        return false;
    }
    let Some((body_id, weight_id)) = find_point_mass_at(state, hover_pos, WEIGHT_HIT_RADIUS) else {
        return false;
    };
    let hovered = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: weight_id.clone() };
    if state.selected.as_ref() == Some(&hovered) {
        return true;
    }
    let Some(lines) = weight_readout_lines(state, &body_id, &weight_id) else { return false };
    egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::new("weight_tooltip"), egui::PopupAnchor::Pointer)
        .show(|ui: &mut egui::Ui| {
            let mut lines = lines.into_iter();
            if let Some(title) = lines.next() {
                ui.label(egui::RichText::new(title).strong());
            }
            for line in lines {
                ui.label(line);
            }
        });
    true
}
```

3g. `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`: `draw_weights` replaces the point-mass block of `render_mechanism` (markers without the permanent "2.00 kg" label), and the weight tooltip goes ahead of the joint tooltip, as in click selection (weights sit on pins).

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
use super::hit_testing::{point_mass_screen_pos, AttachmentHit, BodySegment};
```

Replace with:

```rust
use super::hit_testing::{AttachmentHit, BodySegment};
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
    // ── Draw point masses from blueprint ───────────────────────────
    if let Some(bp) = &state.blueprint {
        let point_mass_color = gc(WEIGHT_COLOR);
        let selected_ring = Stroke::new(2.0, gc(BODY_SELECTED_COLOR));
        for (body_id, bp_body) in &bp.bodies {
            if body_id == GROUND_ID {
                continue;
            }
            for pm in &bp_body.point_masses {
                let Some(screen_pos) = point_mass_screen_pos(state, body_id, pm.local_pos) else {
                    continue;
                };
                // A weight being dragged fades where it is; the drag preview
                // (canvas interaction) draws it at the drop point.
                let dragged = state.weight_drag.as_ref()
                    .is_some_and(|d| d.body_id == *body_id && d.weight_id == pm.id);
                let fill = if dragged { point_mass_color.linear_multiply(0.35) } else { point_mass_color };
                painter.circle_filled(screen_pos, WEIGHT_RADIUS, fill);
                let entity = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: pm.id.clone() };
                if selected.as_ref() == Some(&entity) || state.multi_selected.contains(&entity) {
                    painter.circle_stroke(screen_pos, WEIGHT_RADIUS + 3.0, selected_ring);
                }
                painter.text(
                    screen_pos + Vec2::new(8.0, -8.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{:.2} kg", pm.mass),
                    FontId::proportional(9.0),
                    point_mass_color,
                );
            }
        }
    }
```

Replace with:

```rust
    // ── Weights (point masses): markers, gravity arrows, readout ────
    weights::draw_weights(painter, state);
```

Find in `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`:

```rust
            let mut shown_tooltip = false;

            // Check joints first (they're drawn on top)
```

Replace with:

```rust
            // Weights first, as in click selection (they sit on pins).
            let mut shown_tooltip = weights::show_weight_tooltip(ui, state, hover_pos);

            // Then joints (they're drawn on top of links)
```

3h. Docs (the docs change with the code). Keep a colon followed by a space out of the wording of the YAML items: inside a plain scalar it breaks the parse of the whole file, so the new lines below avoid it. `docs/ai/03-structure.yaml` must still load after this task (Step 4 checks it); `docs/ai/02-system.yaml` already fails to load as one document on main (line 15, unrelated to this task), so only its new item is kept free of `: `.

`docs/FEATURES.md`, Actuator Sizing, after the Braking bands bullet (the two new bullets follow it directly, with no blank line):

Find in `docs/FEATURES.md`:

```markdown
- **Braking bands** -- the Actuator Force and Actuator Power plots shade the crank ranges where the load drives the actuator (negative required power)
```

Add after it:

```markdown
- **Actuator label words** -- the canvas actuator label reads like "1.2 kN push, braking": push (extension) or pull (retraction) from the sign of the required force the Actuator Force plot draws at the current pose, motoring or braking from the required power (the braking bands' rule); both follow the plots whatever the stored force. Without a sweep sample it shows the stored force marked "(stored)"
- **Weight arrows** -- every weight draws an arrow in the gravity direction, longer for heavier weights, green where it helps (coming down), red where the actuator lifts it, gray where it moves sideways, at the current pose. Hovering a weight, or selecting it, shows its name, mass and current force share (a dash near stroke reversal); there are no permanent weight labels
```

`docs/architecture/ENGINEERING_OUTPUTS.md`, after the **Braking:** paragraph of the per-weight breakdown section:

Find in `docs/architecture/ENGINEERING_OUTPUTS.md`:

```markdown
**Braking:** the actuator brakes (the load drives it) where the required actuator power is below `-1e-6 * max|P|` over the sweep. The Actuator Force and Actuator Power plots shade those driver ranges; a band runs halfway to the neighbouring samples on either side. In the quasi-static model the force to hold a load is the same up and down; gravity helping shows up as negative power (braking), not as a smaller force.
```

Add after it:

```markdown
**Canvas readout:** at the current pose (the sweep sample the plot cursor marks) each weight's gravity arrow takes its helping/hurting/neutral colour, hovering or selecting a weight shows its force share, and the actuator label reads like "1.2 kN push, braking" (push/pull from the sign of the plotted required force, motoring/braking from the braking rule above; both are required quantities, so the words do not depend on the stored force).
```

`docs/ai/02-system.yaml`, the readout invariant (the new item follows the previous item directly, with no blank line). Its actuator-label paragraph states the BL-026 rule directly: both words come from required quantities in both modes, not from `F_required - F_stored`.

Find in `docs/ai/02-system.yaml`:

```yaml
    the pick radius of the pointer). The link editor's Add weight button
    adds last_point_mass_kg at the link's base cg_local and selects it.
```

Add after it:

```yaml
  - canvas_readouts_read_the_current_sweep_sample — the canvas actuator
    label and the weight arrows/readouts all read the sweep at
    AppState::current_sweep_index (index_at_driver of the driver angle, or
    of the stroke in a stroke sweep, i.e. the sample the plot cursor marks).
    The actuator label (force_render.rs actuator_label_text calling
    format_actuator_label) shows the magnitude ("1.2 kN", "875 N", "0.35 N"),
    push for force >= SHOWN_AS_ZERO_N (0.005 N), pull for <= -0.005 N
    (positive = extension), then ", motoring" or ", braking" from
    AppState::actuator_label_power, which returns
    (WeightBreakdown::total_power[k], BRAKE_TOL_REL * max|total_power|),
    the rule of WeightBreakdown::braking and the braking bands. No word
    without an actuator breakdown; "(stored)" when the label falls back to
    the element's stored force (no finite sweep sample). Both words derive
    from REQUIRED quantities in both modes since BL-026. The force is
    sweep.actuator_forces (the plotted required force) and the power is
    the required power, so the words match the Actuator Force and Actuator
    Power plots whatever the stored force (tested at every sample with
    stored force 0, the sample's 50 N, and one above every required force).
    Weights (canvas/rendering/weights.rs draw_weights) draw a gold marker
    plus an arrow in the gravity direction (gravity_screen_dir maps g
    through the view, so it points straight down on screen under any
    mounting angle; no arrow when gravity is off), 12-40 px scaled by mass
    relative to the heaviest weight, coloured by
    classification_color(WeightBreakdown::classification(source, k)).
    weight_at_pose only uses a breakdown source with the weight's current
    id, body, local_pos and mass (rebuilds keep the old sweep until the
    debounced recompute); otherwise the arrow is WEIGHT_COLOR and the share
    reads "-". No permanent weight labels. Hover (show_weight_tooltip, same
    gate as the hover ring) or selection (card next to the marker) shows
    the title, the mass (format_mass_kg) and the signed force share with
    its classification word (a Torque share in N*m without an actuator),
    "-" near stroke reversal (weight_readout_lines). Every straight canvas arrowhead goes through
    primitives::draw_arrowhead (draw_arrow adds the shaft and clamps the
    head to the arrow length).
```

`docs/ai/03-structure.yaml`, the `primitives`, `force_render` and `weights` lines and the `test_support` line:

Find in `docs/ai/03-structure.yaml`:

```yaml
        primitives: canvas/rendering/primitives.rs (springs, dampers, arrows, arcs, markers, alignment guides)
        force_render: canvas/rendering/force_render.rs (force element visualization + load-path heat map; actuator label value comes from AppState::actuator_label_force)
        weights: canvas/rendering/weights.rs (payload weights on the canvas - place_mass_hint)
```

Replace with:

```yaml
        primitives: canvas/rendering/primitives.rs (springs, dampers, arrows - draw_arrowhead is the one straight-arrow head, draw_arrow = shaft + head; arcs, markers, alignment guides)
        force_render: canvas/rendering/force_render.rs (force element visualization + load-path heat map; actuator label text actuator_label_text / format_actuator_label, with the required force from AppState::actuator_label_force and motoring/braking from AppState::actuator_label_power; format_magnitude)
        weights: canvas/rendering/weights.rs (payload weights on the canvas - draw_weights draws markers + gravity arrows coloured by classification at AppState::current_sweep_index; show_weight_tooltip and the selected-weight card show weight_readout_lines; weight_at_pose, weight_arrow_length_px, gravity_screen_dir; place_mass_hint)
```

Find in `docs/ai/03-structure.yaml`:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press, typed, the headless frame central_panel_frame, painted-output inspection drawn_texts, drew_text, text_rect; extend it instead of re-implementing fixtures per module)
```

Replace with:

```yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press, typed, the headless frame central_panel_frame, painted-output inspection drawn_texts, drew_text, text_rect, drawn_line_colors; the robot-lift fixture swept_lift with sample_at and pose_at; extend it instead of re-implementing fixtures per module)
```

`docs/ai/05-update-tracker.md`, a new entry at the top:

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Add after it:

```markdown
## 2026-09-29 — Payload weights Task 9: canvas readout
- `AppState::current_sweep_index` (the sample the plot cursor marks;
  `actuator_label_force` now uses it) and `AppState::actuator_label_power`
  (required power + braking tolerance from `WeightBreakdown`).
- Actuator label (`force_render.rs actuator_label_text`,
  `format_actuator_label`): "1.2 kN push, braking" - push/pull by sign
  (positive = extension), motoring/braking by the braking-band rule;
  "(stored)" on the stored-force fallback; "(computed)" is gone. Since
  BL-026 both words derive from REQUIRED quantities: the force is
  `sweep.actuator_forces` (the plotted required force in both modes) and
  the power is `WeightBreakdown::total_power`. The drafts' BL-026 caveat
  (push/pull from F_required - F_stored) is gone, and `swept_lift` no
  longer zeroes the stored force to dodge it.
- `canvas/rendering/weights.rs`: `draw_weights` replaces the point-mass
  block of `render_mechanism`: gold marker plus a gravity-direction arrow
  (`gravity_screen_dir`, 12-40 px by mass, `WEIGHT_ARROW_*` in colors.rs)
  coloured by `classification_color` of the weight's classification at the
  current sample (`weight_at_pose`; gold until the sweep has the weight at
  its current mass and position). The permanent "2.00 kg" labels are gone:
  hovering a weight shows a tooltip (`show_weight_tooltip`, weights before
  joints) and the selected weight a card with the title, mass and force
  share ("-" near stroke reversal). `interaction::weights_interactive` is
  `pub(super)`.
- Arrowheads (DRY): `primitives::draw_arrowhead(tip, dir, head_len,
  stroke)` is the one head of every straight canvas arrow. `draw_arrow`
  (shaft plus a head clamped to the arrow length, used by the weight
  arrows) calls it, and so do `draw_force_arrow`,
  `draw_external_force_arrow`, the Fx/Fy component arrows, the actuator
  line's midpoint head and the force-zone arrows (6 px head), each with
  its old head length, so their geometry is unchanged. The torque arc's
  head (0.5 rad, 5 px) has a different angle and keeps its own code.
- Test helpers (DRY): `test_support` gains `drawn_line_colors`,
  `swept_lift`, `sample_at` and `pose_at`, and reuses
  `set_actuator_stored_force` (no second copy). The canvas
  `weight_readout` tests reuse `weight_clicks::frame` (now returns the
  frame output) and `hit_testing::tests::weight_screen`, and the
  duplicate `weight_screen` in `weight_clicks` is gone.
- Tests: `force_render::tests` (every force/power sign combination, unit
  edges, non-finite force,
  `actuator_label_words_follow_the_plotted_required_force_and_power_in_every_mode`
  at every sample with stored force 0, the sample's 50 N and one above
  every required force, stored fallback "50 N push (stored)"),
  `rendering::weights::tests` (arrow
  length, gravity direction under mounting angles, weight_at_pose vs the
  breakdown incl. stale entries, readout lines incl. reversal dash and
  driver-torque shares, share format), `gui::state::tests`
  (current_sweep_index, actuator_label_power), `gui::canvas::tests::
  weight_readout` (arrow colours at 45/135/90 deg, gold for an unswept
  weight, no arrows without gravity, no permanent labels, hover tooltip,
  selected card, actuator label words on the canvas).
- Mutation check: a label that reads F_required - F_stored fails the
  every-mode test at the sample's own 50 N ("50 N pull" at 90 deg).
- Docs: FEATURES (actuator label words, weight arrows), ENGINEERING_OUTPUTS
  (canvas readout), 02-system `canvas_readouts_read_the_current_sweep_sample`
  (BL-026 text rewritten), 03-structure (primitives, force_render,
  rendering/weights, test_support). The spec checklist lines for the
  readout land with the checklist in Task 10.
```

- [ ] **Step 4: Run the tests to verify they pass**

The task's tests, the whole lib suite and the BL-010 label integration tests:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib -- rendering::weights force_render weight_readout current_sweep_index actuator_label_power weight_clicks
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --test actuator_force_label
```

Expected: the filtered run passes 45 tests (`test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 881 filtered out`): 5 in `force_render::tests`, 9 in `rendering::weights::tests` (the 8 new ones and `place_mass_hint_names_the_next_weight_and_the_field_mass`), 3 in `gui::state::tests` (`current_sweep_index_is_the_sample_nearest_the_driver`, `actuator_label_power_is_the_required_power_and_braking_tolerance`, `actuator_label_power_is_none_without_an_actuator`), 7 in `weight_readout`, and the 21 `weight_clicks` tests (the weight selection, drag and place-mass tests, which now go through the shared `frame` and `weight_screen`). The full lib run passes with `test result: ok. 926 passed; 0 failed` (903 before this task plus its 23 new tests). The BL-010 label integration tests (`label_matches_plot_sample_on_full_sweep`, `label_matches_plot_sample_on_stroke_sweep`, `label_matches_plot_sample_on_seam_crossing_range_sweep`, `stored_force_mode_label_reads_sweep_not_stored_value`) still pass, 4 passed, now through `current_sweep_index`.

Mutation check of the BL-026 guard: make the label read `F_required - F_stored`, confirm the every-mode test fails, then restore the file.

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && sed -i.bak -e 's/format_actuator_label(force, Some(power), tol)/format_actuator_label(force - la.force, Some(power), tol)/' -e 's/None => format_actuator_label(force, None, 0.0),/None => format_actuator_label(force - la.force, None, 0.0),/' src/gui/canvas/rendering/force_render.rs
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib actuator_label_words_follow
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && mv src/gui/canvas/rendering/force_render.rs.bak src/gui/canvas/rendering/force_render.rs
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib actuator_label_words_follow
```

Expected: the mutated run fails, `test result: FAILED. 0 passed; 1 failed`, with `assertion left == right failed: stored 50 N, 90 deg: "50 N pull"` (with stored force 0 the mutant is invisible; it shows at the sample's own 50 N). After the file is restored the test passes (`1 passed`).

Check that the structure map still parses as YAML:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload && python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('03-structure.yaml parses')"
```

Expected: `03-structure.yaml parses`.

Then the gate, and restore the PNGs it rewrites:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
```

Expected: `GATE PASS` with 926 lib tests, every integration suite green, and clippy unchanged from the previous task: `linkage-sim-rs (lib) generated 279 warnings` and `(lib test) generated 295 warnings`, no new warnings.

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add linkage-sim-rs/src/gui/test_support.rs linkage-sim-rs/src/gui/state/mod.rs linkage-sim-rs/src/gui/state/tests.rs linkage-sim-rs/src/gui/canvas/colors.rs linkage-sim-rs/src/gui/canvas/hit_testing.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/canvas/mod.rs linkage-sim-rs/src/gui/canvas/rendering/mod.rs linkage-sim-rs/src/gui/canvas/rendering/primitives.rs linkage-sim-rs/src/gui/canvas/rendering/force_render.rs linkage-sim-rs/src/gui/canvas/rendering/weights.rs docs/FEATURES.md docs/architecture/ENGINEERING_OUTPUTS.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/05-update-tracker.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload status --short
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 9: canvas readout

Weights draw a gold marker and a gravity-direction arrow (12-40 px by
mass) coloured by their helping/hurting/neutral classification at the
current sweep sample (AppState::current_sweep_index); the permanent
"2.00 kg" labels are replaced by a hover tooltip and a selected-weight
card with the title, mass and force share. The actuator label reads like
"1.2 kN push, braking".

BL-026 is fixed on main, so push/pull comes from the plotted required
force and motoring/braking from the required power in both modes. The
label test now runs at every sample with stored force 0, the sample's
50 N and one above every required force, and swept_lift keeps the
sample's stored force.

primitives::draw_arrowhead is the one head of every straight canvas
arrow: draw_arrow, the force and external-force arrows, the Fx/Fy
components, the actuator line and the force-zone arrows use it with
their old head lengths. Test helpers are shared through test_support and
the canvas weight_clicks module.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: `git status --short` lists exactly the 16 files added above (nothing under `docs/chebyshev_lambda/`), and the commit succeeds. The spec's hands-on checklist lines for the readout land with the checklist in Task 10.

### Task 10: Final docs and the hands-on checklist

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md` (append the section "Hands-on checklist (robot-lift model)": 182 lines, 35 checkboxes)
- Modify: `README.md` (the plot-tab bullet, line 170)
- Modify: `docs/FEATURES.md` (the plot-tab bullet, line 44)
- Modify: `docs/ai/02-system.yaml` (two new `known_limitations` entries, before `risks:`)
- Modify: `docs/ai/04-memory.yaml` (one `active_issues` line, one `open_questions` item)
- Modify: `docs/ai/05-update-tracker.md` (a new newest entry)

**Interfaces:**
- Consumes (nothing is changed; the checklist exercises these and the docs name them):
  - `pub const EPS_REL_LDOT: f64 = 0.01;` and `pub fn force_share(p_g: f64, rate: f64, max_abs_rate: f64) -> f64` in `linkage-sim-rs/src/analysis/gravity_breakdown.rs`: force shares are NaN where `|rate| < EPS_REL_LDOT * max_abs_rate`. The checklist's stroke-reversal steps and the first new `known_limitations` entry describe this band.
  - `pub fn reassign_driver(&mut self, joint_id: &str)` in `linkage-sim-rs/src/gui/state/driver_ops.rs` and `pub current_sample: Option<SampleMechanism>` in `linkage-sim-rs/src/gui/state/mod.rs`: driver reassignment on a loaded sample rebuilds the stock sample and drops the weights. The checklist caution and the second new `known_limitations` entry describe this.
  - `enum PlotTab` in `linkage-sim-rs/src/gui/plot_panel/mod.rs` (15 variants, from `CouplerTrace` to `ActuatorPower, WeightBreakdown, OutputForce`): the source of the "15 plot tabs" wording in README and FEATURES.
  - `pub fn current_sweep_index(&self) -> Option<usize>` and `pub fn actuator_label_power(&self) -> Option<(f64, f64)>` on `AppState` (`linkage-sim-rs/src/gui/state/mod.rs`), `pub(super) fn actuator_label_text(state: &AppState, la: &LinearActuatorElement) -> String` (`linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`), `pub fn classification_color(class: Classification) -> Color32` (`linkage-sim-rs/src/gui/canvas/colors.rs`), `pub(super) fn braking_bands(xs: &[f64], braking: &[bool]) -> Vec<(f64, f64)>` (`linkage-sim-rs/src/gui/plot_panel/weights.rs`): the canvas label words, arrow colours and braking bands that sections 3 and 4 of the checklist tick.
  - `pub(crate) fn swept_lift() -> AppState` in `linkage-sim-rs/src/gui/test_support.rs`: the fixture the verification probe ran on (the sample "Parallelogram + Actuator", swept 0-360 deg in 1 deg steps, with W1 on the rocker tip and W2 on the coupler).
  - Behaviour of Tasks 5-9 that the checklist walks through: the toolbar mass field (defaults to `AppState::last_point_mass_kg`), `W<n>` names, the weight editor and the link editor's **Add weight** button, weight hit testing and shift-click, drag/reattach/cancel/delete as one undo step, the push/pull and motoring/braking label words, the hover tooltip and selected readout, the gold stale arrow, the braking bands and the two Weight Breakdown views (`AppState::weight_breakdown_show_power`).
- Produces: documentation only. No Rust signature changes; the lib test total stays at 926 (the diff of `*.rs` against the previous commit is empty).

This is the last task of Track 2: a single hands-on pass for the user over the finished feature, before merge. Every number in the checklist was checked against the code with a throwaway probe test on `test_support::swept_lift` (removed, never committed): labels "7.9 kN push, motoring" at 45 deg, "1.3 kN push, braking" at 135 deg and "0.00 N" at 90 deg; W1's force share +5.5 kN at 45 deg, halving to +2.7 kN at the rocker midpoint; W2's share unchanged when W2 moves; braking at 91-269 deg; Other loads below 4e-7 N; Total equal to the plotted force; stored force 50 N to 0 changing everything by 3e-11 at most; mounting angle 30 deg moving the gray angles to 60/240 deg; driver reassignment followed by one undo restoring the weights. Copy the text below as given; do not round or re-derive the numbers. The probe corrected two claims of the earlier draft, and the text below already carries the corrections:

1. Force shares are NaN only at the 56-57 deg reversal. The reversal at 236.3 deg, where the actuator is at its shortest, has no sample inside the 1 % band, so W1's share jumps from about +45 kN to about -20 kN with no gap. The checklist says so (sections 3 and 4), and the first new `known_limitations` entry records it.
2. On this sample the change-point glitch is at 360 deg on the plots (a one-sample braking band and a pull force), not a band break at 180 deg. The canvas at 360 deg reads sample 0.

The checklist also drops the stale pre-BL-026 lines. There is no "set F to 0" in section 1 (it leaves F at the sample's 50 N), and section 6 says the plots, the Total and the label do not change with F. It folds in the two old Task 8 patches (the readout stays while the weight is selected; the gold arrow until the sweep is recomputed) and the old Task 7 patch (the link editor's "Weights (1)" section with **Add weight**). A caution at the top and a section 6 check cover the one known trap: on a loaded sample, driver reassignment rebuilds the stock sample and discards the weights (one Ctrl+Z restores them, and a model reopened from a file keeps them).

- [ ] **Step 1: Write the failing check**

This task changes only Markdown and YAML, so its test is a docs-presence check. Paste it into a shell; it needs PyYAML (`python -c "import yaml"`). It checks the checklist heading, its 35 boxes and its 7 sections (0-6), the caution, the absence of the stale "set F to 0" step in section 1, the plot-tab count, the two new 02-system entries (and that they parse as YAML on their own; the whole file does not parse, at HEAD either, so it is not parsed as a whole), the 04-memory additions (and that the whole file parses) and the tracker entry.

```bash
ROOT=/c/Users/Cole/source/repos/linkage_simulation-payload
SPEC=$ROOT/docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md
SYS=$ROOT/docs/ai/02-system.yaml
MEM=$ROOT/docs/ai/04-memory.yaml
TRACKER=$ROOT/docs/ai/05-update-tracker.md
fail=0
has() { grep -q -- "$1" "$2" || { echo "MISSING in ${2#$ROOT/}: $1"; fail=1; }; }
lacks() { ! grep -q -- "$1" "$2" || { echo "STALE in ${2#$ROOT/}: $1"; fail=1; }; }

has '^## Hands-on checklist (robot-lift model)' "$SPEC"
boxes=$(grep -c '^- \[ \]' "$SPEC")
[ "$boxes" -eq 35 ] || { echo "checklist boxes in spec: $boxes (want 35)"; fail=1; }
sections=$(sed -n '/^## Hands-on checklist/,$p' "$SPEC" | grep -c '^### [0-6]\. ')
[ "$sections" -eq 7 ] || { echo "checklist sections 0-6 in spec: $sections (want 7)"; fail=1; }
has 'Caution (known issue)' "$SPEC"
has 'stock sample and discards every weight' "$SPEC"
sed -n '/^### 1\. /,/^### 2\. /p' "$SPEC" | grep -q '\*\*F\*\* to 0' && { echo "STALE in spec section 1: set F to 0"; fail=1; }

has '15 plot tabs' "$ROOT/README.md"
has '15 plot tabs' "$ROOT/docs/FEATURES.md"
lacks '10 plot tabs' "$ROOT/README.md"
lacks '10 plot tabs' "$ROOT/docs/FEATURES.md"

has 'The stroke-reversal blank-out is sample based\.' "$SYS"
has 'Driver reassignment on a loaded sample' "$SYS"
{ echo "known_limitations:"; sed -n '/^  - The stroke-reversal blank-out is sample based\./,/^risks:/p' "$SYS" | sed '$d'; } \
  | python -c "import sys, yaml; d = yaml.safe_load(sys.stdin); assert len(d['known_limitations']) == 2, d" 2>/dev/null \
  || { echo "the two new 02-system known_limitations items do not parse as YAML"; fail=1; }

has 'Payload weights Track 2: before merge' "$MEM"
has 'Payload weights Track 2 decisions left to the user' "$MEM"
python -c "import sys, yaml; d = yaml.safe_load(open(sys.argv[1], encoding='utf-8')); assert len(d['active_issues']) == 2 and len(d['open_questions']) == 3, d" "$MEM" 2>/dev/null \
  || { echo "04-memory.yaml does not parse with 2 active_issues and 3 open_questions"; fail=1; }

has '^## 2026-09-29 .* Payload weights Task 10: hands-on checklist and plot-tab counts' "$TRACKER"

[ "$fail" -eq 0 ] && echo "DOCS OK" || echo "DOCS MISSING"
```

- [ ] **Step 2: Run it to verify it fails**

Run the Step 1 script. Expected, on the tree before this task: every line below, ending with `DOCS MISSING`. The README and FEATURES lines say STALE because both still have the old count.

```console
MISSING in docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md: ^## Hands-on checklist (robot-lift model)
checklist boxes in spec: 0 (want 35)
checklist sections 0-6 in spec: 0 (want 7)
MISSING in docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md: Caution (known issue)
MISSING in docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md: stock sample and discards every weight
MISSING in README.md: 15 plot tabs
MISSING in docs/FEATURES.md: 15 plot tabs
STALE in README.md: 10 plot tabs
STALE in docs/FEATURES.md: 10 plot tabs
MISSING in docs/ai/02-system.yaml: The stroke-reversal blank-out is sample based\.
MISSING in docs/ai/02-system.yaml: Driver reassignment on a loaded sample
the two new 02-system known_limitations items do not parse as YAML
MISSING in docs/ai/04-memory.yaml: Payload weights Track 2: before merge
MISSING in docs/ai/04-memory.yaml: Payload weights Track 2 decisions left to the user
04-memory.yaml does not parse with 2 active_issues and 3 open_questions
MISSING in docs/ai/05-update-tracker.md: ^## 2026-09-29 .* Payload weights Task 10: hands-on checklist and plot-tab counts
DOCS MISSING
```

- [ ] **Step 3: Implement**

The files in the working tree use CRLF line endings (`core.autocrlf`; git stores LF). Make each edit with a tool that keeps the file's existing line endings, and do not rewrite whole files: Step 4 checks the diff size, which shows a line-ending rewrite at once.

(a) Payload spec: append the checklist. The current last line of the file is the last item of "Out of scope".

Find in `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`:

```markdown
- Per-weight split of the "With Inertia" curve.
```

Add after it (one blank line first, then the whole section, 182 lines, 35 boxes):

```markdown

## Hands-on checklist (robot-lift model)

Run this after the automated gate passes, before merging Track 2. Tick each
box in the real GUI. A box that fails goes into `docs/ai/backlog.yaml` with
the angle and the numbers seen. The numbers quoted come from the sample's
default sweep (0-360 deg in 1 deg steps) with W1 at the rocker tip.

**Caution (known issue):** while the model is still the loaded sample, that
is until section 6 reopens it from a file, changing the driver rebuilds the
stock sample and discards every weight and every other edit. The driver
changes on right-click of a ground pivot joint (**Set as Driver**, **Set
Driver to …**) or when a load case with another driver joint is applied.
Ctrl+Z brings everything back, and a model reopened from a file keeps its
weights. Section 6 checks the undo.

### 0. Build and smoke test

1. From `linkage-sim-rs/`: `bash scripts/gate.sh` prints `GATE PASS`. The
   tests rewrite `docs/chebyshev_lambda/*.png`; restore them with
   `git checkout -- ../docs/chebyshev_lambda/`.
2. Build and serve the WASM app from `linkage-sim-rs/`:
   `bash scripts/build_web.sh` (needs the `wasm32-unknown-unknown` target
   and `wasm-bindgen-cli`), then `bash scripts/serve_web.sh` in a second
   shell. It serves http://localhost:8080; leave it running.
3. In Claude Code, run the `gui-smoke` skill (workflow
   `.claude/workflows/gui-smoke.js`; default URL http://localhost:8080,
   override with args `{"url": "..."}`). It must report `passed: true`: the
   page loaded, a `<canvas>` is present, and the console shows no errors.
4. Do sections 1-6 in the served page, or natively with
   `cargo run --release --bin linkage-gui` from `linkage-sim-rs/`.

### 1. Model the lift

- [ ] File > Load Sample > **Parallelogram + Actuator**. Leave the
      actuator's stored force **F** at the sample's 50 N: every plot and the
      actuator label show the required force whatever F is (section 6
      checks this).
- [ ] Click **+ Mass**: a mass field appears in the toolbar. Set it to 50 kg
      and click the rocker. The canvas hint reads "Click to place weight W1
      (50 kg) on 'rocker' (Esc to cancel)", and a preview circle shows where
      the weight lands (on the grid when snapping is on). Click the rocker's
      tip, the end joined to the coupler: **W1** appears, selected, and the
      tool returns to Select. If its **Position** does not read X 0 mm,
      Y 0 mm (the tip), type those values.
- [ ] Click **+ Mass** again: the field shows 50 kg, the last mass used. Set
      20 kg, click the coupler, then a point on it: **W2** appears. Type
      "Robot torso" in its **Name** field and press Enter.
- [ ] With W2 selected, the property panel shows "Weight Robot torso (W2)",
      "Link: coupler" and its Name, Mass and body-local Position fields.
      Pick the coupler in the Link Editor: its **Weights (1)** section lists
      W2 and has an **Add weight** button. The rocker's lists W1.
- [ ] Pick the rocker in the Link Editor and click **Add weight**: W3
      (20 kg, the last mass used) appears at the rocker's centre of mass.
      One Ctrl+Z removes it.
- [ ] Click W2's Mass field, type 25 and press Esc: the mass stays 20 kg.
      Type 25 and press Enter: it is 25 kg, and one Ctrl+Z puts back 20 kg.
      While typing, Backspace edits the text and does not delete the
      selected weight.

### 2. Select, drag, reattach, delete

- [ ] Hover a weight: it gets a highlight ring and a grab cursor. Click W1
      at the rocker tip: W1 is selected, not the joint under it. Shift+click
      W2: both are selected; Shift+click W2 again drops it from the
      selection.
- [ ] Drag W2 along the coupler: a dashed line and a marker follow the
      pointer (on the grid when snapping is on), and the mechanism does not
      re-solve until release. After release, one Ctrl+Z puts W2 back.
- [ ] Drag W2 over the rocker and release nearer the rocker than the
      coupler, within 60 px: the rocker is highlighted while dragging, and W2
      moves to the rocker at the drop point with the same id, name and mass.
      One Ctrl+Z reverts it.
- [ ] Start dragging W2, then press Esc, or release outside the canvas:
      nothing changes.
- [ ] Select W1 and press Delete: it is removed. Ctrl+Z restores it as W1.

### 3. Canvas readout at the current pose

Set the crank angle with the **Crank Angle** slider, or by clicking a plot.

- [ ] At 45 deg both weight arrows are red (being lifted), W1's longer than
      W2's (the length grows with the mass). The actuator label reads
      "7.9 kN push, motoring".
- [ ] At 135 deg both arrows are green (coming down); the label reads
      "1.3 kN push, braking".
- [ ] At 90 deg the arrows are gray (moving sideways); the label reads
      "0.00 N", with no push/pull or motoring/braking word.
- [ ] Hover W1 at 45 deg: a tooltip shows "W1", "Mass: 50 kg" and
      "Force share: +5.5 kN (hurting)". Select it: the same readout stays
      next to the weight. With neither hover nor selection, no weight label
      is drawn.
- [ ] Hover W1 at 56 deg, a stroke reversal (the actuator speed passes zero
      between 56 and 57 deg): the share reads "Force share: - (hurting)".
      The other reversal, between 236 and 237 deg, shows no dash: no sample
      falls inside the 1 % speed band there, so W1's share jumps from about
      +45 kN to about -20 kN instead.
- [ ] The label's push/pull word follows the sign of the Actuator Force plot
      at the cursor: positive = push (extension), negative = pull
      (retraction). From 57 to 89 deg, for example, it reads pull.
- [ ] Drag W1 elsewhere on the rocker: its arrow is gold until the sweep is
      recomputed (a moment later), then green, red or gray again. Ctrl+Z
      puts it back at the tip.

### 4. Plots

The force passes through infinity at the stroke reversals (about 227 kN at
56 deg), so those spikes set the y range of the force plots. Scroll to zoom
in; double-click resets the view.

- [ ] **Actuator Power**: shaded bands cover exactly the angles where the
      red statics curve is below zero (91-269 deg). The legend entry
      "Braking" hides and shows them. The tab tooltip mentions the bands.
- [ ] **Actuator Force**: the same bands. The tab tooltip says "Positive =
      extension". Enter a **Rated Force**: the lines are labelled
      "Rated (push)" and "Rated (pull)".
- [ ] At 0, 180 and 360 deg the parallelogram's links are collinear (change
      points) and the velocity is not unique, so one sample there can glitch
      in every channel: the 0 and 180 deg samples dip, and at 360 deg the
      plots show a one-sample braking band, a pull force and mixed line
      colours. That is a solver artifact of this sample, not a payload bug.
      (The canvas at 360 deg reads the 0 deg sample.)
- [ ] **Weight Breakdown**, Force share: the legend lists "coupler (link)",
      "crank (link)", "rocker (link)", "Robot torso (W2)", "W1", "Other
      loads" and "Total". Other loads stays at 0 (gravity is the only load),
      and Total lies on the Actuator Force statics curve. Each weight's line
      is red where that weight rises, green where it falls, and gray where
      it moves sideways (90 and 270 deg).
- [ ] The weight lines and Other loads have a gap at 56-57 deg (force shares
      are blank near stroke reversal), while Total spikes there. At 236-237
      deg the weight lines spike and change sign without a gap (see
      section 3).
- [ ] Near 200 deg W1's line, and the rocker's own, is green (helping) while
      its force share is positive (W1 about +1.0 kN): the retracting
      actuator pushes harder to hold the load back (see "Key physics
      statement").
- [ ] **Power share**: each weight's line is below zero where it is green
      and above zero where it is red, with no gaps; Total lies on the
      Actuator Power statics curve.
- [ ] At three angles, hover the lines: the weight shares plus Other loads
      add up to Total.

### 5. Physical intuition

Read the shares at 45 deg in the Force share view.

- [ ] Set W1's Position to X 1000 mm, Y 0 mm, halfway from the tip (X 0) to
      the rocker pivot (X 2000 mm): its share halves, from about +5.5 kN to
      about +2.7 kN. A rocker point's speed is proportional to its distance
      from the pivot, in the same direction.
- [ ] Drag W2 anywhere on the coupler: its share does not change. The
      parallelogram coupler translates, so every coupler point has the same
      velocity.
- [ ] Set W1's mass to 100 kg: its shares double; the other weights' lines
      do not move.

### 6. Modes, driver and persistence

- [ ] In the property panel, set the actuator's **F** to 0 (sizing mode),
      then back to 50 N: the Actuator Force and Actuator Power plots, the
      Weight Breakdown Total and the actuator label do not change (since
      BL-026 they all show the required force).
- [ ] Open **Mounting Angle** in the input panel and set 30 deg: at 90 deg
      the weights are no longer gray, because gravity now has a component
      along their sideways motion; they turn gray near 60 and 240 deg
      instead. Set it back to 0.
- [ ] Turn on View > **Nathan Mode**: the green, red and gray lines and
      arrows stay distinguishable by brightness. Turn it off.
- [ ] Right-click the rocker's ground pivot joint (J4) and pick **Set as
      Driver**: the stock sample comes back without the weights (the known
      reset in the caution above). One Ctrl+Z restores W1 and W2 with their
      ids, names and masses, and J1 as the driver.
- [ ] Save and reopen: natively File > **Save As...**, then File > **Open
      JSON...**; in the browser File > **Download JSON...**, then File >
      **Recent Mechanisms**. Then use File > **Share via URL** and open the
      copied link in the browser. Each time, the weights keep their ids,
      names and masses, and each link's Mass in the Link Editor still reads
      1 kg (no double counting).
- [ ] Load **4-Bar Crank-Rocker**, which has no actuator: the Weight
      Breakdown shows driver torque shares ("Driver Torque Share (N·m)") for
      crank, coupler and rocker, and the Actuator Force, Actuator Speed and
      Actuator Power tabs are disabled.
```

(b) `README.md`, the plot-tab bullet.

Find in `README.md`:

```markdown
- **10 plot tabs**: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration
```

Replace with:

```markdown
- **15 plot tabs**: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration, actuator force, actuator speed, actuator power, weight breakdown (each weight's force or power share), output force (force zones)
```

(c) `docs/FEATURES.md`, the plot-tab bullet.

Find in `docs/FEATURES.md`:

```markdown
- 10 plot tabs: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration
```

Replace with:

```markdown
- 15 plot tabs: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration, actuator force, actuator speed, actuator power, weight breakdown (each weight's force or power share), output force (force zones)
```

(d) `docs/ai/02-system.yaml`: two entries at the end of `known_limitations`, right before the blank line and `risks:`.

Find in `docs/ai/02-system.yaml`:

```yaml
    use the world center transformed into local coords; if the target body
    is rotated at import time the geometry rectangle inherits that rotation.
```

Add after it:

```yaml
  - The stroke-reversal blank-out is sample based.
    gravity_breakdown::force_share returns NaN only where |dL/dt| <
    EPS_REL_LDOT (1 %) of the sweep max, so a reversal where dL/dt changes
    fast can fall between two samples with neither inside the band. On
    ParallelogramActuator at 1 deg steps the 56.3 deg reversal is blanked
    (56 and 57 deg) but the 236.3 deg one (actuator at its shortest) is
    not; W1 of the hands-on checklist jumps from about +45 kN to -20 kN
    between 236 and 237 deg.
  - Driver reassignment on a loaded sample (AppState::current_sample is
    Some) rebuilds the stock sample (state/driver_ops.rs reassign_driver)
    and drops every blueprint edit, weights included. It is undoable, and a
    model loaded from a file (current_sample None) keeps its weights. The
    payload spec's hands-on checklist warns about it.
```

(e) `docs/ai/04-memory.yaml`: an active issue and an open question.

Find in `docs/ai/04-memory.yaml`:

```yaml
  - "Agentic loop queue: see docs/ai/backlog.yaml (open items tracked there, not here)"
```

Add after it:

```yaml
  - "Payload weights Track 2: before merge the user runs the hands-on checklist at the end of docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md (its section 0 is the gate and gui-smoke)."
```

Find in `docs/ai/04-memory.yaml`:

```yaml
    `FirmwareAdapter` trait — implement on demand once a target controller
    is selected.
```

Add after it (the file's trailing blank line stays after the new item):

```yaml
  - |
    Payload weights Track 2 decisions left to the user. (1) The sweep also
    builds a DriverTorque-basis breakdown for stroke-mode LinearDriver
    sweeps without a LinearActuator ("Driver Force Share (N)"), which the
    spec's section 2 and Out of scope exclude for v1: keep it and update the
    spec, or drop it (weight_breakdown None there)? (2) The spec text still
    differs from the code in small ways: SweepData::weight_shares is
    weight_breakdown, the actuator label reads "1.2 kN push, braking" (spec
    uses a middle dot), the reversal dash is "-" (spec an em dash), and an
    arrow is gold while the sweep lacks the weight (the spec names only
    green/red/gray). Align the spec or the code?
```

(f) `docs/ai/05-update-tracker.md`: a new newest entry, directly below the `---` under the header and above the Task 9 entry, with one blank line before and after it.

Find in `docs/ai/05-update-tracker.md`:

```markdown
Reverse chronological (newest at top).

---
```

Add after it (after the blank line that follows `---`; keep one blank line between this entry and the Task 9 heading):

```markdown
## 2026-09-29 — Payload weights Task 10: hands-on checklist and plot-tab counts
- Payload spec: appended "Hands-on checklist (robot-lift model)", one pass
  over the finished feature. Section 0 is the gate, the WASM build/serve
  scripts and the `gui-smoke` workflow. Sections 1-6 cover placement (toolbar
  mass field, hint, W<n> names), the weight editor and Add weight, Esc/Enter
  commits, select/drag/reattach/cancel/delete, the canvas readout (arrows,
  tooltip, selected readout, label words, gold stale arrow), braking bands,
  the Weight Breakdown views, the physical-intuition checks, stored-force
  invariance, mounting angle, Nathan Mode, save/reopen/share and a
  no-actuator sample. It folds in the checklist patches from the old Task
  7/8 drafts. It drops the pre-BL-026 steps: no "set F to 0 first", and the
  Actuator Force plot no longer shows F_required - F_stored.
- A caution at the top of the checklist and a new 02-system
  known_limitations entry: on a loaded sample, driver reassignment rebuilds
  the stock sample and drops every weight. It is undoable, and a model
  reopened from a file keeps its weights.
- A throwaway probe on `test_support::swept_lift` (not committed) checked
  every number and claim in the checklist against the code. Labels:
  "7.9 kN push, motoring" at 45 deg, "1.3 kN push, braking" at 135 and
  "0.00 N" at 90. W1's share is +5.5 kN at 45 and halves at the rocker
  midpoint. Braking runs 91-269 deg, Other loads stays below 4e-7 N, Total
  equals the plotted force, and going from stored force 50 N to 0 changes
  nothing (3e-11). A 30 deg mounting angle moves gray to 60/240 deg.
  Reassign-then-undo restores the weights.
- The probe corrected two draft claims. (1) Force shares are blank only at
  56-57 deg. The 236.3 deg reversal, where the actuator is at its shortest,
  has no sample inside the 1 % band, so W1's share jumps from about +45 kN
  to -20 kN with no gap (new 02-system known_limitations entry). (2) The
  change-point glitch on this sample is at 360 deg on the plots (a
  one-sample braking band and a pull force), not a band break at 180.
- README and FEATURES: 15 plot tabs (was "10"). The list now includes
  actuator force/speed/power, weight breakdown and output force.
- 04-memory: active issue (the checklist runs before merge) and an open
  question: stroke-mode driver-share scope and the small spec/code
  wording mismatches.
```

- [ ] **Step 4: Run the checks to verify they pass**

1. Run the Step 1 script again. Expected: `DOCS OK`, with no other line.

2. Check that the edits are additive and kept the line endings:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload diff --stat
```

Expected: six files changed, `245 insertions(+), 2 deletions(-)` (the only deletions are the two plot-tab lines that were replaced, in README and FEATURES). Hundreds of deletions mean an edit rewrote a file's line endings; redo that edit.

3. The lib tests are unchanged (no Rust file is touched):

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && cargo test --lib
```

Expected: `test result: ok. 926 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.

4. Run the full gate. This task edits only Markdown and YAML. None of it is compiled: no source file pulls a docs page in with `include_str!` (the only `include_str!` in `linkage-sim-rs/src` reads `samples/custom_6bar.json`), so the spec and `docs/architecture/ENGINEERING_OUTPUTS.md` add no doctests. The gate's `cargo test --all` still runs the doctests and every other suite, so a green gate only confirms that nothing regressed.

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-payload/linkage-sim-rs && bash scripts/gate.sh
```

Expected: `cargo test --all` (lib 926 passed, 0 failed, and every other suite green), clippy and the WASM check pass, and the last line is `GATE PASS`.

5. The gate's tests rewrite `docs/chebyshev_lambda/*.png`. Restore them, then confirm that only the six intended files are modified:

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload checkout -- docs/chebyshev_lambda/
git -C /c/Users/Cole/source/repos/linkage_simulation-payload status --short
```

Expected: exactly these six lines and nothing else (no `.png`).

```console
 M README.md
 M docs/FEATURES.md
 M docs/ai/02-system.yaml
 M docs/ai/04-memory.yaml
 M docs/ai/05-update-tracker.md
 M docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md
```

- [ ] **Step 5: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-payload add README.md docs/FEATURES.md docs/ai/02-system.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md
git -C /c/Users/Cole/source/repos/linkage_simulation-payload commit -F - <<'EOF'
task 10: final docs and the hands-on checklist

Payload spec: append the hands-on robot-lift checklist, one pass over
the finished feature. Section 0 covers the gate, the WASM build/serve
scripts and the gui-smoke workflow. Sections 1-6 cover placement, the
weight editor, drag/reattach/delete, the canvas readout, the braking
bands, the Weight Breakdown, the physical-intuition checks,
stored-force invariance, mounting angle, Nathan Mode,
save/reopen/share and a no-actuator sample. It folds in the old Task
7/8 checklist patches and drops the pre-BL-026 steps. A caution warns
that driver reassignment on a loaded sample discards the weights
(undoable).

A throwaway probe checked every number and claim against the code.
It corrected two draft claims: shares are blank only at the 56-57 deg
reversal (at 236-237 deg they jump with no gap), and the change-point
glitch is at 360 deg, not 180.

README/FEATURES: 15 plot tabs, now listing the actuator and Weight
Breakdown tabs. 02-system: two known_limitations entries (the
sample-based reversal blank-out, driver reassignment discarding
weights). 04-memory: the pending checklist and the open Track 2
decisions. Tracker entry.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```
