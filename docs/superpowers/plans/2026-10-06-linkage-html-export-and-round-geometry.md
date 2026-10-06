# Linkage website: round shapes, the contact point, and an animated HTML export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Part A gives a link's visual geometry a circle shape and a force zone a third application mode, the shape's contact point, so a wheel's load acts (and is drawn) at the wheel's bottom at every pose. Part B adds File -> "Export Animation (HTML)...", a self-contained animated page of the current mechanism like the user's press animation.

**Architecture:** Part A: `BodyGeometry` gains a `shape` (rectangle, the default and left out of files, or circle, its diameter in `width` with `height` equal) and three methods (`centre_world`, `outline_world`, `extreme_point_world`); every place that today builds the rectangle's world corners for a force zone (the force evaluation, the overlap ratio, the independent equilibrium check, the canvas marker) calls one new helper, `force_zone_application`, which returns the overlap and the application point by the zone's mode (overlap centroid, locked body point, or contact point = the shape's extreme point against the zone's force). The property panel switches shapes and adds them; the force editor picks the mode. Part B (Tasks 5 and 6, written against `main` after Part A merged): the export rebuilds every solved sweep sample's pose from the sweep's own records (no second solve), computes outlines, force visuals, reactions and readouts in Rust, and embeds them as JSON in an HTML template with a small JS player.

**Tech Stack:** Rust 2024, egui 0.32 (headless tests), nalgebra, serde/serde_json; no new dependency. Playwright for the browser checks.

**Spec:** the user's requests: 2026-10-05 "Can the square instead be a circle whose bottom center point is located where the F is located at full extension of actuator. 5.2" diameter, centered as is"; "In the website why is the F locked shown as the centroid now?"; asked whether to add round shapes and a lowest-point contact to the website: "Yes add to 10:45 task"; and "Schedule task at 10:45am to update website to be able to export html animation that is this but for the current layout if possible" (the reference: `Lift_Linkage_Animation_2026-10-06.html`, the press animation checked against the website solver). Physics the user confirmed the same day: the wheel and the mass move rigidly with the link; the ground's vertical push acts at the wheel's lowest point, which slides round the rim relative to the link ("It still acts vertically, just on a different part of wheel relative to link").

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-shapes`, branch `linkage/round-geometry`, created by Task 0 from `main` at `f9df219` with LF line endings and a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed without the user's go.

**Process (the user's lean process, memory `project_magcoupling_lean_pacing`):** Part A is Tasks 0 to 4. Tasks 1 to 3 give the exact code, so implementers transcribe on `sonnet` with a `sonnet` review per task; Task 4 (docs and the browser check) also on `sonnet`; the whole-branch review after Task 4 on the session model. No pre-flight scan: every old block quotes `main` at `f9df219` (or the file as earlier tasks leave it). Part B (Tasks 5 and 6) got its exact code in this file after Part A merged (decision H-8) and runs the same way, from its own worktree (see Part B).

## Decisions to confirm

**Confirmed (user, 2026-10-06): the recommended option on all of R-1 to R-7 and H-1 to H-8.** The plan implements the recommended option of each (code and docs cite the ids).

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| R-1 | How a circle is stored | **A `shape` field on `BodyGeometry`:** `rectangle` (the default, left out of files, so every existing file keeps its bytes) or `circle`, whose diameter is `width`, with `height` written equal to it (a build before schema 1.2.0 ignores `shape` and draws the circle's bounding square) | (b) `BodyGeometry` as an enum of shapes (a cleaner type; a new file format and every use rewritten); (c) a separate optional circle field on the body | 1 |
| R-2 | How the user adds and switches shapes | **The property panel:** "Add rectangle" and "Add circle" buttons on a link without geometry (sized from the link: a rectangle as long as the link and a quarter as deep, a circle half the link's length, both centred on the link), and a Rectangle / Circle switch plus a Diameter slider on a link with geometry. The "Draw Geometry" and "Redraw" buttons go: their canvas tool is an unfinished stub that does nothing (BL-044) | (b) also finish the canvas draw tool (drag out a rectangle or a circle): a separate task; (c) circles from files only | 3 |
| R-3 | What "contact point" means | **The shape's extreme point against the zone's force:** where a surface pushing in the force's direction first meets the shape; for an upward force, the shape's lowest point (a wheel's bottom, straight below its hub at every pose); for a downward force, its top; a tilted force follows. A level rectangle edge gives the edge's midpoint | (b) the lowest point in the world frame whatever the force; (c) the lowest point along gravity (follows the mounting angle) | 2 |
| R-4 | How the force editor shows it | **A three-way choice, "Apply at: Overlap centre / Locked point / Contact point",** replacing the "Lock to body-local point" checkbox and the "Reset to auto" button. Choosing Locked keeps an earlier locked point, else seeds the shape's centre (today it seeds the body origin, which for a DXF-imported body is the world origin, far from the link). Choosing Contact keeps the locked point, so switching back restores it. Dragging the F marker switches to Locked at the drop point | (b) keep the checkbox and add a separate "at contact point" checkbox | 2, 3 |
| R-5 | The schema version | **1.2.0:** the loader checks only the major number, so files written now still open in older builds (which draw circles as squares and ignore the contact mode) | (b) stay 1.1.0 (the format change goes unmarked) | 1 |
| R-6 | Circle maths | **A 64-sided polygon for the overlap test and the overlap centroid; a true circle on the canvas; the exact centre plus radius for the contact point** | (b) an exact circle-rectangle intersection (more code for a binary on/off test) | 1, 2 |
| R-7 | What proves the contact point right | **A sample linkage, not the user's press:** the Parallelogram Press with a wheel on its coupler gives the same driver torque, sample for sample, with the force at the wheel's contact point as with it locked at the hub (a vertical force on the hub's vertical line). The repo is public, so the user's press model is not committed; its numbers (21,506 N at crank 28 deg, 5,776 N at 65 deg) are checked locally in Task 4 and live after the push | (b) commit the user's press model as a test fixture | 2, 4 |
| H-1 | Where the export lives | **File -> "Export animation (HTML)...",** after "Generate Report (HTML)..." | (b) also a button on the plot panel | B3 |
| H-2 | A mechanism without an actuator | **The driver torque takes the actuator force's place** in the readout and the chart | (b) no chart without an actuator | B1, B2 |
| H-3 | Samples and file size | **The sweep's own samples** (1 deg over the sweep range, 1 mm for a stroke drive; about 0.3 MB for the press over 28-65 deg) | (b) 0.25 deg (smoother, four times the file) | B1 |
| H-4 | Units | **The app's units (N, N m, and mm or m from Display Units) with lbf beside every force** (the app itself has no lbf; the user works in lbf) | (b) the app's units only; (c) a units switch inside the page | B1, B2 |
| H-5 | Samples with no solution | **Left out:** the slider covers solved samples only; the chart breaks there | (b) shown as a "no solution" frame | B1, B2 |
| H-6 | Self-contained | **One file, no network:** inline SVG and JS, no CDN (the existing HTML report loads Plotly and KaTeX from CDNs) | (b) reuse Plotly from its CDN for the chart | B2 |
| H-7 | Desktop app too | **Yes:** the web download and the desktop save dialog, through `export::download::download_text` like every export | (b) web only | B3 |
| H-8 | Shipping order | **Part A first:** built, reviewed, merged and (with the user's go) pushed; then Part B's code is written here against that `main`, built and pushed | (b) both parts on one branch, one push | all |

## Global Constraints

- Base: `main` at `f9df219` ("merge: BL-042 weights draw over force zones"). Worktree `C:/Users/Cole/source/repos/lsim-shapes`, branch `linkage/round-geometry`. Never modify `C:/Users/Cole/source/repos/linkage_simulation` itself beyond the `worktree add` of Task 0 Step 1.
- Existing files keep loading and keep their meaning: a geometry without `shape` is a rectangle; a force zone without `at_contact_point` keeps its locked point or centroid; files with only rectangles and no contact mode are written with the same keys as before (only `schema_version` changes, to `1.2.0`).
- The magcoupling crate is untouched: no file under `magcoupling-rs/` changes, and gates 4 to 12 stay green.
- No new dependency; both `Cargo.lock` files unchanged.
- Formatting: never run `cargo fmt` in `linkage-sim-rs` (not rustfmt-clean at `f9df219`); new code is written in rustfmt's layout.
- UI text is ASCII where a glyph could be missing from egui's default fonts: no diameter sign (write "diameter").
- Blocks quote the files with LF line endings, as git stores them. "Create `path`:" writes a new file with the block's text and a final newline. "In `path`, replace: ... with: ..." is one exact replacement whose old block occurs exactly once, as whole lines; "replace both occurrences" says so explicitly. Apply a step's blocks in order. If an old block is not found, stop and escalate; never improvise a match.
- Gate runs (Task 0 and every task before its commit): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-shapes checkout -- docs/chebyshev_lambda` (the tests rewrite those PNGs; never commit them).
- Commits: one per task, subjects `feat(linkage): ...` (Task 4: `docs(linkage): ...`). Every commit message ends with exactly two lines: `Co-Authored-By: <the model writing the commit> <noreply@anthropic.com>`, then `Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg`. Nothing is pushed.
- Docs change with code (repo rule): Task 4 updates `docs/reference/FORCE_ELEMENTS.md`, `docs/FEATURES.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` and `backlog.yaml`, and commits this plan. Read `docs/ai/*.yaml` first (Task 0).
- Model tiers (CLAUDE.md section 5): Tasks 0 to 4 run on `sonnet` (`model: 'sonnet'`); a `sonnet` attempt that ends blocked, leaves a check red or is rejected in review escalates every retry to the session model. The whole-branch review runs on the session model. Task 2 touches the statics path (the force's application point); its maths is fixed by the decisions and pinned by tests, so it is transcription, but its review must check the residual test and the sweep-equality test ran, not only compiled.

## Review Focus

The inputs and conditions the requests imply but no feature test would naturally hit, most likely first, each pinned by a test in the task that owns the code:

1. **A file written before this change** (a geometry without `shape`, a zone without `at_contact_point`). Expected: it loads as a rectangle and keeps its locked point or centroid; written back, it gains no new keys. Tests: `a_rectangle_is_written_without_a_shape_key_and_old_files_load_as_rectangles` (Task 1), `at_contact_point_is_left_out_when_false_and_defaults_to_false` (Task 2).
2. **A contact point while the shape is outside the zone.** Expected: no force (the overlap gate still decides), but the canvas still marks where the force will act. Test: `the_contact_point_still_needs_the_shape_in_the_zone` (Task 2).
3. **Dragging the F marker of a zone in contact mode.** Expected: the zone switches to Locked at the drop point (today the drag would set a locked point that the contact mode then ignores). Test: `dragging_the_contact_marker_locks_it_where_it_drops` (Task 3).
4. **A level rectangle against a vertical force** (two corners equally low). Expected: the contact point is the bottom edge's midpoint, not one arbitrary corner; tilted, the lowest corner. Test: `a_rectangle_s_extreme_point_is_its_farthest_corner_or_its_edge_midpoint` (Task 1).
5. **A sideways or zero force in contact mode.** Expected: a sideways push meets the shape's side; a zero force falls back to the lowest point (and applies nothing). Tests: `a_sideways_force_meets_the_side_of_the_wheel` (Task 2), `a_zero_direction_gives_the_shape_s_centre` (Task 1, the geometry layer; the zone layer passes a downward direction for a zero force).

## File Structure

Part A (Tasks 1 to 3):

- `linkage-sim-rs/src/core/body.rs`: `GeometryShape`, `CIRCLE_SEGMENTS`, `BodyGeometry.shape`, `BodyGeometry::circle`, `centre_world`, `outline_world`, `extreme_point_world`; `area` covers circles (Task 1).
- `linkage-sim-rs/src/io/schema.rs`: `SCHEMA_VERSION` 1.2.0 (Task 1).
- `linkage-sim-rs/src/forces/elements/element_types.rs`: `ForceZoneElement.at_contact_point`, `ZoneAppMode`, `ForceZoneElement::app_mode`, `with_app_mode`, `is_false` (Task 2).
- `linkage-sim-rs/src/forces/elements/evaluation.rs`: `ZoneApplication`, `force_zone_application`; `evaluate_force_zone` and `force_zone_overlap_ratio` call it (Task 2).
- `linkage-sim-rs/src/solver/reactions.rs`: the independent check's force-zone branch calls it; a contact-mode validation test (Task 2).
- `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`: `force_zone_app_point_world` and the overlap highlight and F marker call it; "F (contact)" (Task 2).
- `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`: the geometry drawn as a rectangle or a circle; the body tooltip (Task 3).
- `linkage-sim-rs/src/gui/property_panel/mod.rs`, `pending_edits.rs`: the shape switch, Diameter, Add rectangle / Add circle; one `edit_geometry` helper for the blueprint and mechanism copies; `default_geometry` (Task 3).
- `linkage-sim-rs/src/gui/property_panel/force_editor.rs`: the three-way application point (Task 3).
- `linkage-sim-rs/src/gui/canvas/interaction.rs`: dropping the F marker locks it (Task 3); the zone tool's literal gains the new field (Task 2).
- Literal sites of `BodyGeometry` (Task 1) and `ForceZoneElement` (Task 2): `pending_edits.rs`, `dxf_import.rs`, `reactions.rs` tests, `samples/fourbar.rs`, `interaction.rs`, `tests/force_zone_tests.rs`.
- Tests: `tests/geometry_tests.rs` (Task 1), `tests/force_zone_tests.rs` and the new `tests/wheel_contact_sweep.rs` (Task 2), `pending_edits.rs` and `gui/canvas/mod.rs` test modules (Task 3).

Part B (Tasks 5 and 6): `linkage-sim-rs/src/gui/sweep/mod.rs` (`sweep_time`), `linkage-sim-rs/src/gui/export/animation.rs` and `animation_template.html` (new), `export/mod.rs`, `menu_bar.rs`; see "Part B: animated HTML export" at the end.

## Tasks

### Task 0: Worktree, baseline and context

**Files:** none changed (a worktree is created).

- [ ] **Step 1: Create the LF worktree**

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b linkage/round-geometry C:/Users/Cole/source/repos/lsim-shapes f9df219
git -C C:/Users/Cole/source/repos/lsim-shapes config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-shapes log --oneline -1
```

Expected: `f9df219 merge: BL-042 weights draw over force zones`.

- [ ] **Step 2: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-shapes/docs/ai/01-meta.yaml`, `02-system.yaml` (the force-zone and geometry entries), `03-structure.yaml` (the `geometry` and `forces` modules) and `04-memory.yaml`.

- [ ] **Step 3: Baseline gate**

Run the gate command from Global Constraints. Expected: `GATE PASS`. Record the linkage `cargo test` pass count (gate 1's last `test result` lines). Then `git -C C:/Users/Cole/source/repos/lsim-shapes checkout -- docs/chebyshev_lambda`.

- [ ] **Step 4: Confirm the old blocks exist**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs
grep -c 'body_local_app_point: None,' tests/force_zone_tests.rs
grep -n 'pub const SCHEMA_VERSION: &str = "1.1.0";' src/io/schema.rs
```

Expected: `9`, and one line at 18.

- [ ] **Step 5: Report** the baseline counts. No commit.

- [x] **Step 6: Decisions.** Confirmed by the user on 2026-10-06: the recommended option on all of R-1..R-7 and H-1..H-8 (recorded under Decisions to confirm).

---

### Task 1: Circle shapes in the core

**Files:**
- Modify: `linkage-sim-rs/src/core/body.rs:17-42`
- Modify: `linkage-sim-rs/src/io/schema.rs:18`
- Modify (struct literals): `linkage-sim-rs/src/gui/property_panel/pending_edits.rs:134-147`, `linkage-sim-rs/src/gui/dxf_import.rs:1119-1123`, `linkage-sim-rs/src/solver/reactions.rs:1581-1585` and `:1933-1937`
- Test: `linkage-sim-rs/tests/geometry_tests.rs`

**Interfaces:**
- Produces: `pub enum GeometryShape { Rectangle, Circle }` (serde `snake_case`, `Default` = `Rectangle`, `Copy`, `PartialEq`); `pub const CIRCLE_SEGMENTS: usize = 64`; `BodyGeometry { width, height, offset, shape }`; `BodyGeometry::circle(diameter: f64, offset: Vector2<f64>) -> Result<BodyGeometry, BodyError>`; `fn centre_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vector2<f64>`; `fn outline_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vec<Vector2<f64>>` (CCW; 4 corners or 64 vertices); `fn extreme_point_world(&self, body_x: f64, body_y: f64, body_theta: f64, direction: Vector2<f64>) -> Vector2<f64>`. All in `crate::core::body`.

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/tests/geometry_tests.rs`, replace:

```rust
use linkage_sim_rs::core::body::BodyGeometry;
```

with:

```rust
use linkage_sim_rs::core::body::{BodyGeometry, GeometryShape, CIRCLE_SEGMENTS};
```

Then append to the end of the file:

```rust

// ── Round shapes (schema 1.2.0, decisions R-1, R-3, R-5, R-6) ───────────

#[test]
fn a_circle_keeps_its_diameter_as_width_and_height() {
    let geo = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    assert_eq!(geo.shape, GeometryShape::Circle);
    assert_relative_eq!(geo.width, 0.13208);
    assert_relative_eq!(geo.height, 0.13208);
    assert_relative_eq!(
        geo.area(),
        std::f64::consts::PI * 0.13208 * 0.13208 / 4.0,
        epsilon = 1e-15
    );
}

#[test]
fn a_circle_needs_a_positive_finite_diameter() {
    for d in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        assert!(BodyGeometry::circle(d, Vector2::zeros()).is_err(), "diameter {d}");
    }
}

#[test]
fn a_rectangle_outline_is_its_four_corners() {
    let offset = Vector2::new(0.05, 0.0);
    let geo = BodyGeometry::new(0.2, 0.1, offset).unwrap();
    assert_eq!(geo.shape, GeometryShape::Rectangle);
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    let corners = body_rect_to_world(1.0, 2.0, FRAC_PI_2, 0.2, 0.1, &offset);
    assert_eq!(outline.len(), 4);
    for (a, b) in outline.iter().zip(corners.iter()) {
        assert_relative_eq!(a.x, b.x, epsilon = 1e-12);
        assert_relative_eq!(a.y, b.y, epsilon = 1e-12);
    }
}

#[test]
fn a_circle_outline_is_a_polygon_on_the_circle_turned_with_the_body() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    assert_eq!(outline.len(), CIRCLE_SEGMENTS);
    let centre = geo.centre_world(1.0, 2.0, FRAC_PI_2);
    assert_relative_eq!(centre.x, 1.0, epsilon = 1e-12);
    assert_relative_eq!(centre.y, 2.2, epsilon = 1e-12);
    for p in &outline {
        assert_relative_eq!((p - centre).norm(), 0.05, epsilon = 1e-12);
    }
    // Counter-clockwise, like the rectangle's corners: positive signed area.
    let n = outline.len();
    let signed: f64 = (0..n)
        .map(|i| {
            let (a, b) = (outline[i], outline[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        / 2.0;
    assert!(signed > 0.0, "signed area {signed}");
    assert_relative_eq!(polygon_area(&outline), signed, epsilon = 1e-15);
}

#[test]
fn a_circle_s_extreme_point_is_its_centre_plus_the_radius_along_the_direction() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    // However the body turns, the point does not turn with it.
    for theta in [0.0, 0.7, -2.0] {
        let centre = geo.centre_world(1.0, 2.0, theta);
        let low = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(0.0, -3.0));
        assert_relative_eq!(low.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(low.y, centre.y - 0.05, epsilon = 1e-12);
        let corner = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(1.0, 1.0));
        let r = 0.05 / 2f64.sqrt();
        assert_relative_eq!(corner.x, centre.x + r, epsilon = 1e-12);
        assert_relative_eq!(corner.y, centre.y + r, epsilon = 1e-12);
    }
}

#[test]
fn a_rectangle_s_extreme_point_is_its_farthest_corner_or_its_edge_midpoint() {
    let geo = BodyGeometry::new(0.2, 0.1, Vector2::zeros()).unwrap();
    // Level: the whole bottom edge meets a surface below; its midpoint.
    let level = geo.extreme_point_world(0.0, 0.0, 0.0, Vector2::new(0.0, -1.0));
    assert_relative_eq!(level.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(level.y, -0.05, epsilon = 1e-12);
    // Tilted: one corner is lowest.
    let theta = 0.3_f64;
    let tilted = geo.extreme_point_world(0.0, 0.0, theta, Vector2::new(0.0, -1.0));
    let corners = body_rect_to_world(0.0, 0.0, theta, 0.2, 0.1, &Vector2::zeros());
    let lowest = corners.iter().min_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
    assert_relative_eq!(tilted.x, lowest.x, epsilon = 1e-12);
    assert_relative_eq!(tilted.y, lowest.y, epsilon = 1e-12);
}

#[test]
fn a_zero_direction_gives_the_shape_s_centre() {
    let shapes = [
        BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap(),
        BodyGeometry::circle(0.1, Vector2::new(0.05, 0.0)).unwrap(),
    ];
    for geo in shapes {
        for direction in [Vector2::zeros(), Vector2::new(f64::NAN, 1.0)] {
            let p = geo.extreme_point_world(1.0, 2.0, 0.4, direction);
            let c = geo.centre_world(1.0, 2.0, 0.4);
            assert_relative_eq!(p.x, c.x, epsilon = 1e-15);
            assert_relative_eq!(p.y, c.y, epsilon = 1e-15);
        }
    }
}

#[test]
fn a_rectangle_is_written_without_a_shape_key_and_old_files_load_as_rectangles() {
    let rect = BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap();
    let json = serde_json::to_value(&rect).unwrap();
    assert!(json.get("shape").is_none(), "{json}");
    let old: BodyGeometry =
        serde_json::from_str(r#"{"width":0.2,"height":0.1,"offset":[0.05,0.0]}"#).unwrap();
    assert_eq!(old.shape, GeometryShape::Rectangle);
}

#[test]
fn a_circle_round_trips_and_keeps_width_and_height_for_older_builds() {
    let circle = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    let json = serde_json::to_value(&circle).unwrap();
    assert_eq!(json["shape"], "circle");
    // A build before 1.2.0 ignores "shape" and draws the circle's bounding square.
    assert_eq!(json["width"], json["height"]);
    let back: BodyGeometry = serde_json::from_value(json).unwrap();
    assert_eq!(back.shape, GeometryShape::Circle);
    assert_relative_eq!(back.width, 0.13208);
}

#[test]
fn files_with_shapes_are_schema_1_2_0() {
    assert_eq!(linkage_sim_rs::io::SCHEMA_VERSION, "1.2.0");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test geometry_tests`
Expected: FAIL to compile ("unresolved imports `GeometryShape`, `CIRCLE_SEGMENTS`").

- [ ] **Step 3: Implement the shape**

In `linkage-sim-rs/src/core/body.rs`, replace:

```rust
/// Visual rectangular geometry attached to a body.
/// Used for force zone overlap computation and canvas rendering.
/// Dimensions are in body-local frame, centered on `offset` from body origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis
    pub height: f64,           // meters, extent along body's local y-axis
    pub offset: Vector2<f64>,  // local-frame offset from body origin
}

impl BodyGeometry {
    /// Create a new body geometry. Width and height must be > 0.
    pub fn new(width: f64, height: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(BodyError::InvalidGeometry {
                reason: format!("width ({}) and height ({}) must both be > 0", width, height),
            });
        }
        Ok(Self { width, height, offset })
    }

    /// Total area of the rectangle (m^2).
    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}
```

with:

```rust
/// The shape of a body's visual geometry (decision R-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryShape {
    /// A `width` x `height` rectangle; every file written before schema 1.2.0.
    #[default]
    Rectangle,
    /// A circle of diameter `width`. `height` is kept equal to it, so a build
    /// before schema 1.2.0, which ignores the shape, draws its bounding square.
    Circle,
}

impl GeometryShape {
    /// Rectangles are written without a `shape` key, so files without circles
    /// keep the keys they had before schema 1.2.0.
    fn is_rectangle(&self) -> bool {
        *self == GeometryShape::Rectangle
    }
}

/// Number of sides of the polygon that stands in for a circle in the force-zone
/// overlap test and its centroid; the canvas draws a true circle (decision R-6).
pub const CIRCLE_SEGMENTS: usize = 64;

/// Visual geometry attached to a body: a rectangle or a circle.
/// Used for force zone overlap computation and canvas rendering.
/// Dimensions are in body-local frame, centered on `offset` from body origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis (a circle's diameter)
    pub height: f64,           // meters, extent along body's local y-axis (a circle: equal to width)
    pub offset: Vector2<f64>,  // local-frame offset from body origin
    /// Rectangle (the default, left out of files) or circle.
    #[serde(default, skip_serializing_if = "GeometryShape::is_rectangle")]
    pub shape: GeometryShape,
}

impl BodyGeometry {
    /// Create a rectangle. Width and height must be > 0.
    pub fn new(width: f64, height: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(BodyError::InvalidGeometry {
                reason: format!("width ({}) and height ({}) must both be > 0", width, height),
            });
        }
        Ok(Self { width, height, offset, shape: GeometryShape::Rectangle })
    }

    /// Create a circle of `diameter` centred on `offset`. The diameter must be
    /// a finite number > 0.
    pub fn circle(diameter: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if !(diameter > 0.0 && diameter.is_finite()) {
            return Err(BodyError::InvalidGeometry {
                reason: format!("diameter ({}) must be a finite number > 0", diameter),
            });
        }
        Ok(Self { width: diameter, height: diameter, offset, shape: GeometryShape::Circle })
    }

    /// Area (m^2): width x height for a rectangle, pi d^2 / 4 for a circle.
    pub fn area(&self) -> f64 {
        match self.shape {
            GeometryShape::Rectangle => self.width * self.height,
            GeometryShape::Circle => std::f64::consts::PI * self.width * self.width / 4.0,
        }
    }

    /// The shape's centre in the world frame, for a body at (`body_x`, `body_y`)
    /// turned by `body_theta`.
    pub fn centre_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vector2<f64> {
        local_to_world(body_x, body_y, body_theta, self.offset)
    }

    /// The outline in the world frame, counter-clockwise: the rectangle's four
    /// corners (bottom-left first), or a `CIRCLE_SEGMENTS`-gon inscribed in the
    /// circle.
    pub fn outline_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vec<Vector2<f64>> {
        match self.shape {
            GeometryShape::Rectangle => crate::geometry::body_rect_to_world(
                body_x,
                body_y,
                body_theta,
                self.width,
                self.height,
                &self.offset,
            )
            .to_vec(),
            GeometryShape::Circle => {
                let r = self.width / 2.0;
                (0..CIRCLE_SEGMENTS)
                    .map(|k| {
                        let a = std::f64::consts::TAU * k as f64 / CIRCLE_SEGMENTS as f64;
                        let local = self.offset + Vector2::new(r * a.cos(), r * a.sin());
                        local_to_world(body_x, body_y, body_theta, local)
                    })
                    .collect()
            }
        }
    }

    /// The point of the shape farthest along `direction` (world frame) for a
    /// body at (`body_x`, `body_y`) turned by `body_theta`: where a surface
    /// approaching from that side first touches it (decision R-3). For a
    /// circle, the centre plus the radius along the direction, wherever the
    /// body has turned; for a rectangle, its farthest corner, or the midpoint
    /// of the edge when two corners tie (an edge square to the direction). A
    /// zero or non-finite direction gives the centre.
    pub fn extreme_point_world(
        &self,
        body_x: f64,
        body_y: f64,
        body_theta: f64,
        direction: Vector2<f64>,
    ) -> Vector2<f64> {
        let centre = self.centre_world(body_x, body_y, body_theta);
        let norm = direction.norm();
        if !(norm > 0.0 && norm.is_finite()) {
            return centre;
        }
        let d = direction / norm;
        match self.shape {
            GeometryShape::Circle => centre + d * (self.width / 2.0),
            GeometryShape::Rectangle => {
                let corners = self.outline_world(body_x, body_y, body_theta);
                let best = corners.iter().map(|c| c.dot(&d)).fold(f64::NEG_INFINITY, f64::max);
                let tol = 1e-9 * self.width.max(self.height);
                let tied: Vec<&Vector2<f64>> =
                    corners.iter().filter(|c| best - c.dot(&d) <= tol).collect();
                tied.iter().fold(Vector2::zeros(), |sum, c| sum + **c) / tied.len() as f64
            }
        }
    }
}

/// A body-local point in the world frame, for a body at (`x`, `y`) turned by
/// `theta`.
fn local_to_world(x: f64, y: f64, theta: f64, p: Vector2<f64>) -> Vector2<f64> {
    let (sin_t, cos_t) = theta.sin_cos();
    Vector2::new(x + cos_t * p.x - sin_t * p.y, y + sin_t * p.x + cos_t * p.y)
}
```

- [ ] **Step 4: Bump the schema version (decision R-5)**

In `linkage-sim-rs/src/io/schema.rs`, replace:

```rust
/// Current schema version for the JSON format.
pub const SCHEMA_VERSION: &str = "1.1.0";
```

with:

```rust
/// Current schema version for the JSON format. 1.2.0 added circle geometry
/// (`BodyGeometry.shape`) and a force zone's contact point
/// (`ForceZoneElement.at_contact_point`); both are left out of a file that does
/// not use them, and the loader checks only the major number.
pub const SCHEMA_VERSION: &str = "1.2.0";
```

- [ ] **Step 5: Give the struct literals the shape**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace both occurrences (the blueprint's and the mechanism's) of:

```rust
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                        });
```

with:

```rust
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
```

(Task 3 replaces this handler; this step only keeps the crate compiling.)

In `linkage-sim-rs/src/gui/dxf_import.rs`, replace:

```rust
            body.geometry = Some(BodyGeometry {
                width,
                height,
                offset: offset_local,
            });
```

with:

```rust
            body.geometry = Some(BodyGeometry {
                width,
                height,
                offset: offset_local,
                shape: crate::core::body::GeometryShape::Rectangle,
            });
```

In `linkage-sim-rs/src/solver/reactions.rs`, replace both occurrences (the tests `validation_verified_with_force_zone` and `validation_verified_force_zone_centroid_branch`) of:

```rust
        coupler.geometry = Some(BodyGeometry {
            width: 3.0,
            height: 0.4,
            offset: Vector2::new(1.5, 0.0),
        });
```

with:

```rust
        coupler.geometry = Some(BodyGeometry::new(3.0, 0.4, Vector2::new(1.5, 0.0)).unwrap());
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test geometry_tests`
Expected: PASS (all tests, the ten new ones included).

Then run the gate (Global Constraints). Expected: `GATE PASS`, gate 1's pass count = Task 0's + 10. Restore `docs/chebyshev_lambda`.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/core/body.rs linkage-sim-rs/src/io/schema.rs linkage-sim-rs/src/gui/property_panel/pending_edits.rs linkage-sim-rs/src/gui/dxf_import.rs linkage-sim-rs/src/solver/reactions.rs linkage-sim-rs/tests/geometry_tests.rs
git commit -m "feat(linkage): circle body geometry (decisions R-1, R-5, R-6)" -m "BodyGeometry gains a shape: rectangle (the default, left out of files) or circle (diameter in width, height equal), with centre_world, outline_world and extreme_point_world. Schema 1.2.0." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 2: A force zone's contact point, one shared helper

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements/element_types.rs:301-328`
- Modify: `linkage-sim-rs/src/forces/elements/evaluation.rs:1-11` (imports), `:445-554`
- Modify: `linkage-sim-rs/src/solver/reactions.rs:284-326`, `:1618-1625`, `:1968-1975`, and a new test after `validation_verified_force_zone_centroid_branch`
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs:77-117`, `:464-550`
- Modify (literals): `linkage-sim-rs/src/gui/canvas/interaction.rs:1113-1120`, `linkage-sim-rs/src/gui/samples/fourbar.rs:350-357`, `linkage-sim-rs/tests/force_zone_tests.rs` (9 literals)
- Test: `linkage-sim-rs/tests/force_zone_tests.rs`, create `linkage-sim-rs/tests/wheel_contact_sweep.rs`

**Interfaces:**
- Consumes (Task 1): `BodyGeometry::outline_world`, `extreme_point_world`, `centre_world`, `BodyGeometry::circle`.
- Produces: `ForceZoneElement.at_contact_point: bool`; `pub enum ZoneAppMode { Centroid, Locked, Contact }`; `ForceZoneElement::app_mode(&self) -> ZoneAppMode`; `ForceZoneElement::with_app_mode(&self, mode: ZoneAppMode, seed: [f64; 2]) -> ForceZoneElement`; `pub struct ZoneApplication { pub overlap: Vec<Vector2<f64>>, pub active: bool, pub point: Option<Vector2<f64>>, pub mode: ZoneAppMode }`; `pub fn force_zone_application(fz: &ForceZoneElement, geo: &BodyGeometry, pose: (f64, f64, f64)) -> ZoneApplication`. All re-exported from `crate::forces::elements`.

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/tests/force_zone_tests.rs`, replace:

```rust
use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{ForceElement, ForceZoneElement};
```

with:

```rust
use approx::assert_relative_eq;
use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{
    force_zone_application, ForceElement, ForceZoneElement, ZoneAppMode,
};
```

Then give every literal the new field: after each of the 9 lines `        body_local_app_point: None,` insert the line `        at_contact_point: false,`:

```bash
cd C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs
sed -i 's/^        body_local_app_point: None,$/        body_local_app_point: None,\n        at_contact_point: false,/' tests/force_zone_tests.rs
grep -c '        at_contact_point: false,' tests/force_zone_tests.rs
```

Expected: `9`.

Then append to the end of the file:

```rust

// ── Contact point (schema 1.2.0, decisions R-3, R-4) ─────────────────────

/// The bar of `build_test_mechanism` carrying a wheel (a 0.05 m circle centred
/// 0.1 m along it), turned to `theta`. Positions are not re-solved: the zone
/// reads the pose straight from q.
fn wheel_bar_at(theta: f64) -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let mut bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0);
    bar.geometry = Some(BodyGeometry::circle(0.05, Vector2::new(0.1, 0.0)).unwrap());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let mut q = DVector::zeros(mech.state().n_coords());
    q[mech.state().get_index("bar").unwrap().theta_idx()] = theta;
    (mech, q)
}

/// A zone over the whole wheel, its force at the contact point.
fn contact_zone(force: [f64; 2]) -> ForceZoneElement {
    ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force,
        label: None,
        body_local_app_point: None,
        at_contact_point: true,
    }
}

#[test]
fn the_contact_point_stays_at_the_bottom_of_the_wheel_as_it_turns() {
    for theta in [0.0, 0.6, 1.4, -0.9] {
        let (mech, q) = wheel_bar_at(theta);
        let geo = mech.bodies()["bar"].geometry.clone().unwrap();
        let pose = mech.state().get_pose("bar", &q);
        let app = force_zone_application(&contact_zone([0.0, 500.0]), &geo, pose);
        assert_eq!(app.mode, ZoneAppMode::Contact);
        assert!(app.active, "theta {theta}");
        let centre = geo.centre_world(pose.0, pose.1, pose.2);
        let point = app.point.unwrap();
        assert_relative_eq!(point.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(point.y, centre.y - 0.025, epsilon = 1e-12);
    }
}

#[test]
fn the_contact_point_loads_the_body_like_the_same_force_at_the_hub() {
    use linkage_sim_rs::forces::helpers::point_force_to_q;
    let (mech, q) = wheel_bar_at(0.6);
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(contact_zone([0.0, 500.0]))
        .evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    // The force at the wheel's bottom, given in the body frame.
    let (bx, by, th) = mech.state().get_pose("bar", &q);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let bottom = geo.centre_world(bx, by, th) - Vector2::new(0.0, 0.025);
    let (s, c) = th.sin_cos();
    let (dx, dy) = (bottom.x - bx, bottom.y - by);
    let local = Vector2::new(c * dx + s * dy, -s * dx + c * dy);
    let force = Vector2::new(0.0, 500.0);
    let want = point_force_to_q(mech.state(), "bar", &local, &force, &q);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(*g, *w, epsilon = 1e-12);
    }
    // A vertical force on the hub's vertical line: the same Q as at the hub.
    let at_hub = point_force_to_q(mech.state(), "bar", &Vector2::new(0.1, 0.0), &force, &q);
    for (g, h) in got.iter().zip(at_hub.iter()) {
        assert_relative_eq!(*g, *h, epsilon = 1e-9);
    }
}

#[test]
fn a_sideways_force_meets_the_side_of_the_wheel() {
    let (mech, q) = wheel_bar_at(0.0);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let pose = mech.state().get_pose("bar", &q);
    let app = force_zone_application(&contact_zone([-500.0, 0.0]), &geo, pose);
    // Pushing towards -x: the surface comes from +x and meets the rightmost point.
    let p = app.point.unwrap();
    assert_relative_eq!(p.x, 0.125, epsilon = 1e-12);
    assert_relative_eq!(p.y, 0.0, epsilon = 1e-12);
    // A zero force falls back to the lowest point.
    let zero = force_zone_application(&contact_zone([0.0, 0.0]), &geo, pose);
    let p = zero.point.unwrap();
    assert_relative_eq!(p.x, 0.1, epsilon = 1e-12);
    assert_relative_eq!(p.y, -0.025, epsilon = 1e-12);
}

#[test]
fn the_contact_point_still_needs_the_shape_in_the_zone() {
    let (mech, q) = wheel_bar_at(0.0);
    let mut fz = contact_zone([0.0, 500.0]);
    fz.zone_min = [10.0, 10.0];
    fz.zone_max = [11.0, 11.0];
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let app = force_zone_application(&fz, &geo, mech.state().get_pose("bar", &q));
    assert!(!app.active);
    assert!(app.point.is_some(), "the canvas still marks where the force will act");
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(fz).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    assert!(got.iter().all(|v| v.abs() < 1e-12), "{got:?}");
}

#[test]
fn the_contact_point_wins_over_a_locked_point_which_is_kept() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.body_local_app_point = Some([0.05, 0.0]);
    assert_eq!(fz.app_mode(), ZoneAppMode::Contact);
    fz.at_contact_point = false;
    assert_eq!(fz.app_mode(), ZoneAppMode::Locked);
    fz.body_local_app_point = None;
    assert_eq!(fz.app_mode(), ZoneAppMode::Centroid);
}

#[test]
fn with_app_mode_switches_and_keeps_the_locked_point() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    // Locked with no earlier point: the seed (the panel passes the shape's centre).
    let locked = fz.with_app_mode(ZoneAppMode::Locked, [0.1, 0.0]);
    assert_eq!(locked.app_mode(), ZoneAppMode::Locked);
    assert_eq!(locked.body_local_app_point, Some([0.1, 0.0]));
    // Contact keeps the locked point; Locked again restores it, ignoring the seed.
    let contact = locked.with_app_mode(ZoneAppMode::Contact, [9.0, 9.0]);
    assert_eq!(contact.app_mode(), ZoneAppMode::Contact);
    assert_eq!(contact.body_local_app_point, Some([0.1, 0.0]));
    let back = contact.with_app_mode(ZoneAppMode::Locked, [9.0, 9.0]);
    assert_eq!(back.app_mode(), ZoneAppMode::Locked);
    assert_eq!(back.body_local_app_point, Some([0.1, 0.0]));
    // Overlap centre clears both.
    let centroid = contact.with_app_mode(ZoneAppMode::Centroid, [9.0, 9.0]);
    assert_eq!(centroid.app_mode(), ZoneAppMode::Centroid);
    assert_eq!(centroid.body_local_app_point, None);
    assert!(!centroid.at_contact_point);
}

#[test]
fn at_contact_point_is_left_out_when_false_and_defaults_to_false() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    let json = serde_json::to_value(ForceElement::ForceZone(fz.clone())).unwrap();
    assert!(json.get("at_contact_point").is_none(), "{json}");
    let back: ForceElement = serde_json::from_value(json).unwrap();
    let ForceElement::ForceZone(back) = back else { panic!("a force zone") };
    assert!(!back.at_contact_point);
    fz.at_contact_point = true;
    let json = serde_json::to_value(ForceElement::ForceZone(fz)).unwrap();
    assert_eq!(json["at_contact_point"], true);
}
```

Create `linkage-sim-rs/tests/wheel_contact_sweep.rs`:

```rust
//! A wheel's contact point loads the linkage exactly like the same vertical
//! force locked at the wheel's hub, at every sample of a sweep (decision R-7:
//! a sample linkage stands in for the user's press, which the public repo does
//! not carry).

use linkage_sim_rs::core::body::BodyGeometry;
use linkage_sim_rs::forces::elements::ForceElement;
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test force_zone_tests --test wheel_contact_sweep`
Expected: FAIL to compile (`force_zone_application`, `ZoneAppMode`, `at_contact_point` unknown).

- [ ] **Step 3: The zone's mode**

In `linkage-sim-rs/src/forces/elements/element_types.rs`, replace:

```rust
/// A spatial force zone: applies a constant distributed force to a body
/// proportional to the overlap area between the body's geometry and the zone.
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. When `body_local_app_point` is `None`, the force is
/// applied at the centroid of the zone-geometry overlap polygon, projected
/// into the body's local frame each frame. When `Some`, the force is
/// applied at that fixed body-local point, letting the user pin the
/// application location to a specific contact point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
```

with:

```rust
fn is_false(b: &bool) -> bool {
    !*b
}

/// How a force zone picks its application point (`ForceZoneElement::app_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneAppMode {
    /// The centroid of the geometry's overlap with the zone, found at every pose.
    Centroid,
    /// A fixed body-local point (`body_local_app_point`).
    Locked,
    /// The shape's contact point: its extreme point against the force, found
    /// at every pose (decision R-3).
    Contact,
}

/// A spatial force zone: applies its full constant force to a body whenever
/// the body's geometry overlaps the zone (binary: any overlap, full force).
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. Where the force acts is `app_mode`: the overlap
/// centroid (no locked point), a locked body-local point
/// (`body_local_app_point`), or the shape's contact point
/// (`at_contact_point`), which wins over a locked point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
```

Then replace:

```rust
    /// Optional override for the application point, in body-local
    /// coordinates (meters). When `Some`, replaces the auto-computed
    /// overlap centroid. See struct-level docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_local_app_point: Option<[f64; 2]>,
}
```

with:

```rust
    /// Optional override for the application point, in body-local
    /// coordinates (meters). When `Some`, replaces the auto-computed
    /// overlap centroid. See struct-level docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_local_app_point: Option<[f64; 2]>,
    /// When true, the force acts at the shape's contact point (the bottom of a
    /// wheel for an upward force), found again at every pose, so it slides
    /// round a circle as the body turns. Wins over `body_local_app_point`,
    /// which is kept so switching back restores it. Left out of files when
    /// false; absent from files before schema 1.2.0.
    #[serde(default, skip_serializing_if = "is_false")]
    pub at_contact_point: bool,
}

impl ForceZoneElement {
    /// Where the force acts: the contact point wins, then a locked point, else
    /// the overlap centroid.
    pub fn app_mode(&self) -> ZoneAppMode {
        if self.at_contact_point {
            ZoneAppMode::Contact
        } else if self.body_local_app_point.is_some() {
            ZoneAppMode::Locked
        } else {
            ZoneAppMode::Centroid
        }
    }

    /// This zone switched to `mode` (decision R-4): Centroid clears the locked
    /// point and the contact flag; Locked keeps an earlier locked point, else
    /// locks at `seed` (body-local); Contact sets the flag and keeps the locked
    /// point, so switching back restores it.
    pub fn with_app_mode(&self, mode: ZoneAppMode, seed: [f64; 2]) -> Self {
        let mut zone = self.clone();
        zone.at_contact_point = mode == ZoneAppMode::Contact;
        match mode {
            ZoneAppMode::Centroid => zone.body_local_app_point = None,
            ZoneAppMode::Locked => {
                zone.body_local_app_point = self.body_local_app_point.or(Some(seed));
            }
            ZoneAppMode::Contact => {}
        }
        zone
    }
}
```

- [ ] **Step 4: The shared helper and its callers**

In `linkage-sim-rs/src/forces/elements/evaluation.rs`, replace:

```rust
use crate::core::body::Body;
```

with:

```rust
use crate::core::body::{Body, BodyGeometry};
```

Then replace:

```rust
// ── Force zone evaluation ────────────────────────────────────────────────────

pub fn evaluate_force_zone(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid};

    let n = state.n_coords();
    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return DVector::zeros(n),
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return DVector::zeros(n),
    };

    // Get body position from state vector via BodyIndex
    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return DVector::zeros(n),
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    // Transform body rectangle to world frame
    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);

    // Clip against zone AABB
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);

    if overlap_area < 1e-15 {
        return DVector::zeros(n);
    }

    // Binary overlap: any contact → full force.
    let force_global = Vector2::new(fz.force[0], fz.force[1]);

    // Application point: either the user-pinned body-local override, or
    // the centroid of the overlap polygon projected into body-local
    // coords. The override is useful when the actual contact point isn't
    // the overlap centroid (e.g., a specific contact pad location).
    let local_point = if let Some(lp) = fz.body_local_app_point {
        Vector2::new(lp[0], lp[1])
    } else {
        let centroid_world = polygon_centroid(&clipped);
        let cos_t = btheta.cos();
        let sin_t = btheta.sin();
        let dx = centroid_world.x - bx;
        let dy = centroid_world.y - by;
        Vector2::new(
            cos_t * dx + sin_t * dy,
            -sin_t * dx + cos_t * dy,
        )
    };

    point_force_to_q(state, &fz.body_id, &local_point, &force_global, q)
}

/// Check whether any part of a force zone's target body overlaps the zone.
///
/// Returns 1.0 if any part of the body geometry intersects the zone, 0.0
/// otherwise. Binary semantics: partial overlap applies the full force.
pub fn force_zone_overlap_ratio(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> f64 {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area};

    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return 0.0,
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return 0.0,
    };

    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return 0.0,
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);

    if overlap_area < 1e-15 {
        0.0
    } else {
        1.0
    }
}
```

with:

```rust
// ── Force zone evaluation ────────────────────────────────────────────────────

/// Where a force zone's force acts on its body at one pose.
#[derive(Debug, Clone)]
pub struct ZoneApplication {
    /// The body's geometry clipped to the zone (world frame); empty without overlap.
    pub overlap: Vec<Vector2<f64>>,
    /// True when the geometry overlaps the zone: the full force applies.
    pub active: bool,
    /// The application point (world frame) by `mode`: the contact point and a
    /// locked point always exist (the canvas marks them before the body
    /// reaches the zone); the overlap centroid only with overlap.
    pub point: Option<Vector2<f64>>,
    pub mode: ZoneAppMode,
}

/// The overlap gate and the application point of `fz` on a body whose
/// geometry is `geo`, at pose `(x, y, theta)`. The force evaluation, the
/// overlap ratio, the independent equilibrium check and the canvas all call
/// this, so they cannot disagree.
pub fn force_zone_application(
    fz: &ForceZoneElement,
    geo: &BodyGeometry,
    pose: (f64, f64, f64),
) -> ZoneApplication {
    use crate::geometry::{clip_polygon_to_aabb, polygon_area, polygon_centroid};

    let (bx, by, btheta) = pose;
    let outline = geo.outline_world(bx, by, btheta);
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let overlap = clip_polygon_to_aabb(&outline, &zone_min, &zone_max);
    // Binary overlap: any contact -> full force.
    let active = polygon_area(&overlap) >= 1e-15;
    let mode = fz.app_mode();
    let point = match mode {
        ZoneAppMode::Contact => {
            // A surface pushing along the force meets the shape's extreme point
            // against it: the lowest point for an upward push. A zero force
            // falls back to the lowest point.
            let against = Vector2::new(-fz.force[0], -fz.force[1]);
            let direction = if against.norm() > 0.0 { against } else { Vector2::new(0.0, -1.0) };
            Some(geo.extreme_point_world(bx, by, btheta, direction))
        }
        ZoneAppMode::Locked => {
            let lp = fz.body_local_app_point.unwrap_or([0.0, 0.0]);
            let (sin_t, cos_t) = btheta.sin_cos();
            Some(Vector2::new(bx + cos_t * lp[0] - sin_t * lp[1], by + sin_t * lp[0] + cos_t * lp[1]))
        }
        ZoneAppMode::Centroid => active.then(|| polygon_centroid(&overlap)),
    };
    ZoneApplication { overlap, active, point, mode }
}

/// The body's pose in `q` and its geometry, or `None` when the body is
/// unknown or has no geometry.
fn zone_body<'a>(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &'a HashMap<String, Body>,
    q: &DVector<f64>,
) -> Option<(&'a BodyGeometry, (f64, f64, f64))> {
    let geo = bodies.get(&fz.body_id)?.geometry.as_ref()?;
    let bi = state.get_index(&fz.body_id).ok()?;
    Some((geo, (q[bi.x_idx()], q[bi.y_idx()], q[bi.theta_idx()])))
}

pub fn evaluate_force_zone(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    let n = state.n_coords();
    let Some((geo, pose)) = zone_body(fz, state, bodies, q) else {
        return DVector::zeros(n);
    };
    let app = force_zone_application(fz, geo, pose);
    let (true, Some(world)) = (app.active, app.point) else {
        return DVector::zeros(n);
    };
    // The application point in the body's local frame, as point_force_to_q takes it.
    let (bx, by, btheta) = pose;
    let (sin_t, cos_t) = btheta.sin_cos();
    let (dx, dy) = (world.x - bx, world.y - by);
    let local_point = Vector2::new(cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy);
    let force_global = Vector2::new(fz.force[0], fz.force[1]);
    point_force_to_q(state, &fz.body_id, &local_point, &force_global, q)
}

/// Check whether any part of a force zone's target body overlaps the zone.
///
/// Returns 1.0 if any part of the body geometry intersects the zone, 0.0
/// otherwise. Binary semantics: partial overlap applies the full force.
pub fn force_zone_overlap_ratio(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> f64 {
    match zone_body(fz, state, bodies, q) {
        Some((geo, pose)) if force_zone_application(fz, geo, pose).active => 1.0,
        _ => 0.0,
    }
}
```

In `linkage-sim-rs/src/solver/reactions.rs`, replace:

```rust
    // For force zones, two decisions are SHARED with production, not re-derived:
    // the binary overlap gate (polygon_area < 1e-15) and the unpinned-app-point
    // centroid (polygon_centroid). A bug in those shared geometry primitives
    // would corrupt both sides identically and cancel — narrow blast radius
    // (wrong app-POINT or overlap DECISION only, not a moment-arm projection).
    for fe in mech.forces() {
        match fe {
            ForceElement::ForceZone(fz) => {
                use crate::geometry::{
                    body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid,
                };
                let Some(body) = mech.bodies().get(&fz.body_id) else { continue };
                let Some(geo) = body.geometry.as_ref() else { continue };
                let (bx, by, bth) = state.get_pose(&fz.body_id, q);
                let corners = body_rect_to_world(bx, by, bth, geo.width, geo.height, &geo.offset);
                let zmin = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
                let zmax = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
                let clipped = clip_polygon_to_aabb(&corners, &zmin, &zmax);
                // Binary overlap: any contact → full force (matches
                // evaluate_force_zone).
                if polygon_area(&clipped) < 1e-15 {
                    continue;
                }
                let force = Vector2::new(fz.force[0], fz.force[1]);
                // World application point: pinned body-local override, else
                // the overlap-polygon centroid.
                let app = if let Some(lp) = fz.body_local_app_point {
                    let (c, s) = (bth.cos(), bth.sin());
                    Vector2::new(
                        bx + c * lp[0] - s * lp[1],
                        by + s * lp[0] + c * lp[1],
                    )
                } else {
                    polygon_centroid(&clipped)
                };
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &fz.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
```

with:

```rust
    // For force zones, two decisions are SHARED with production, not re-derived:
    // the binary overlap gate and the world application point (the overlap
    // centroid, the locked point or the shape's contact point), both from
    // force_zone_application. A bug there would corrupt both sides identically
    // and cancel — narrow blast radius (wrong app-POINT or overlap DECISION
    // only, not a moment-arm projection).
    for fe in mech.forces() {
        match fe {
            ForceElement::ForceZone(fz) => {
                use crate::forces::elements::force_zone_application;
                let Some(body) = mech.bodies().get(&fz.body_id) else { continue };
                let Some(geo) = body.geometry.as_ref() else { continue };
                let zone = force_zone_application(fz, geo, state.get_pose(&fz.body_id, q));
                // Binary overlap: any contact → full force (matches
                // evaluate_force_zone).
                let (true, Some(app)) = (zone.active, zone.point) else { continue };
                let force = Vector2::new(fz.force[0], fz.force[1]);
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &fz.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
```

In the same file, replace:

```rust
            label: None,
            body_local_app_point: Some([2.5, 0.0]),
        }));
```

with:

```rust
            label: None,
            body_local_app_point: Some([2.5, 0.0]),
            at_contact_point: false,
        }));
```

Then replace:

```rust
            force: [400.0, -1000.0],
            label: None,
            body_local_app_point: None,
        }));
```

with:

```rust
            force: [400.0, -1000.0],
            label: None,
            body_local_app_point: None,
            at_contact_point: false,
        }));
```

Then add the contact-mode check after the centroid test. Replace:

```rust
    /// A mechanism containing an element type the independent check does
    /// not model must report `Unverified` — never a false `Verified`/
    /// `Failed`. Two actuators trips the ">1 actuator" guard.
```

with:

```rust
    /// Pin the force-zone CONTACT branch (schema 1.2.0, decision R-3): a wheel
    /// (circle) on the coupler, the force at its extreme point against a
    /// tilted force, so the check's contact point must agree with production's.
    #[test]
    fn validation_verified_force_zone_contact_branch() {
        use crate::core::body::BodyGeometry;
        use crate::forces::elements::ForceZoneElement;
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        coupler.geometry = Some(BodyGeometry::circle(0.8, Vector2::new(2.0, 0.3)).unwrap());
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        }));
        // A zone over the whole wheel; the force tilted so the contact point
        // sits off the wheel's vertical line and its moment is real.
        mech.add_force(ForceElement::ForceZone(ForceZoneElement {
            body_id: "coupler".to_string(),
            zone_min: [-10.0, -10.0],
            zone_max: [10.0, 10.0],
            force: [300.0, 1000.0],
            label: None,
            body_local_app_point: None,
            at_contact_point: true,
        }));
        mech.build().unwrap();

        let q = seed_pose_at_angle(&mech, PI / 3.0);
        let r = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Verified,
            "gravity + actuator + a contact-point zone should independently verify",
        );
        let resid = body_equilibrium_residual(&mech, &q, PI / 3.0, &r.reactions, r.actuator_force)
            .expect("modeled");
        assert!(resid < 1e-9, "contact-point equilibrium residual {resid}");
    }

    /// A mechanism containing an element type the independent check does
    /// not model must report `Unverified` — never a false `Verified`/
    /// `Failed`. Two actuators trips the ">1 actuator" guard.
```

In `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`, replace:

```rust
/// Compute the current world-space application point of a `ForceZoneElement`.
///
/// - If `body_local_app_point` is `Some`, the override point is transformed
///   by the body's current pose.
/// - Otherwise, the overlap polygon (body geometry ∩ zone AABB) is computed
///   and its centroid is returned.
///
/// Returns `None` when the target body is missing, has no geometry, or
/// there is no overlap and no override.
pub fn force_zone_app_point_world(
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let body = mech.bodies().get(&fz.body_id)?;
    let geo = body.geometry.as_ref()?;
    let (bx, by, btheta) = mech_state.get_pose(&fz.body_id, q);
    let cos_t = btheta.cos();
    let sin_t = btheta.sin();

    if let Some(lp) = fz.body_local_app_point {
        return Some(nalgebra::Vector2::new(
            bx + cos_t * lp[0] - sin_t * lp[1],
            by + sin_t * lp[0] + cos_t * lp[1],
        ));
    }

    let corners = crate::geometry::body_rect_to_world(
        bx, by, btheta, geo.width, geo.height, &geo.offset,
    );
    let zmin = nalgebra::Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zmax = nalgebra::Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = crate::geometry::clip_polygon_to_aabb(&corners, &zmin, &zmax);
    if clipped.len() >= 3 {
        Some(crate::geometry::polygon_centroid(&clipped))
    } else {
        None
    }
}
```

with:

```rust
/// Compute the current world-space application point of a `ForceZoneElement`
/// (`force_zone_application`): the contact point or the locked point at the
/// body's current pose, else the overlap centroid.
///
/// Returns `None` when the target body is missing, has no geometry, or the
/// zone uses the centroid and there is no overlap.
pub fn force_zone_app_point_world(
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let geo = mech.bodies().get(&fz.body_id)?.geometry.as_ref()?;
    force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q)).point
}
```

In the same file, replace:

```rust
    // 5. Active overlap highlight + application-point marker.
    if let Some(body) = mech.bodies().get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            let (bx, by, btheta) = mech_state.get_pose(&fz.body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                bx, by, btheta, geo.width, geo.height, &geo.offset,
            );
            let zone_min_v = nalgebra::Vector2::new(fz.zone_min[0], fz.zone_min[1]);
            let zone_max_v = nalgebra::Vector2::new(fz.zone_max[0], fz.zone_max[1]);
            let clipped = crate::geometry::clip_polygon_to_aabb(
                &corners, &zone_min_v, &zone_max_v,
            );
            let has_overlap = clipped.len() >= 3;
            if has_overlap {
                let screen_verts: Vec<Pos2> = clipped
                    .iter()
                    .map(|v| {
                        let sp = view.world_to_screen(v.x, v.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_verts,
                    FORCE_ZONE_OVERLAP_FILL,
                    Stroke::new(1.0, FORCE_ZONE_OVERLAP_STROKE),
                ));
            }

            // Application-point marker. Always drawn when the zone has
            // any overlap (or when the user has pinned a body-local
            // override, regardless of overlap, so they can still see
            // where the force would act once the mechanism reaches the
            // zone). Shown as a crosshair + dot. A locked override uses
            // a stronger color; the auto-centroid uses a softer color.
            let cos_t = btheta.cos();
            let sin_t = btheta.sin();

            let (app_world, is_locked) = if let Some(lp) = fz.body_local_app_point {
                // Override: world = body_pose + R(theta) * local
                let wx = bx + cos_t * lp[0] - sin_t * lp[1];
                let wy = by + sin_t * lp[0] + cos_t * lp[1];
                (Some(nalgebra::Vector2::new(wx, wy)), true)
            } else if has_overlap {
                (Some(crate::geometry::polygon_centroid(&clipped)), false)
            } else {
                (None, false)
            };

            if let Some(app_w) = app_world {
                let sp = view.world_to_screen(app_w.x, app_w.y);
                let center = Pos2::new(sp[0], sp[1]);
                let color = if is_locked {
                    Color32::from_rgb(255, 165, 80)
                } else {
                    Color32::from_rgb(255, 215, 120)
                };
```

with:

```rust
    // 5. Active overlap highlight + application-point marker.
    if let Some(body) = mech.bodies().get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            let zone = force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q));
            if zone.active {
                let screen_verts: Vec<Pos2> = zone
                    .overlap
                    .iter()
                    .map(|v| {
                        let sp = view.world_to_screen(v.x, v.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_verts,
                    FORCE_ZONE_OVERLAP_FILL,
                    Stroke::new(1.0, FORCE_ZONE_OVERLAP_STROKE),
                ));
            }

            // Application-point marker. Drawn whenever the point exists: a
            // locked or contact point even before the body reaches the zone,
            // so the user sees where the force will act; the centroid only
            // with overlap. Shown as a crosshair + dot. A locked or contact
            // point uses a stronger color; the auto-centroid a softer one.
            let (color, label) = match zone.mode {
                ZoneAppMode::Centroid => (Color32::from_rgb(255, 215, 120), "F"),
                ZoneAppMode::Locked => (Color32::from_rgb(255, 165, 80), "F (locked)"),
                ZoneAppMode::Contact => (Color32::from_rgb(255, 165, 80), "F (contact)"),
            };
            if let Some(app_w) = zone.point {
                let sp = view.world_to_screen(app_w.x, app_w.y);
                let center = Pos2::new(sp[0], sp[1]);
```

Then replace:

```rust
                // Label — "F" for auto, "F (locked)" for override.
                let label = if is_locked { "F (locked)" } else { "F" };
                painter.text(
```

with:

```rust
                painter.text(
```

(`force_render.rs` already has `use crate::forces::elements::*;`, which brings `force_zone_application` and `ZoneAppMode`.)

- [ ] **Step 5: The other zone literals**

In `linkage-sim-rs/src/gui/canvas/interaction.rs`, replace:

```rust
                        force: [0.0, -100.0],
                        label: None,
                        body_local_app_point: None,
                    });
```

with:

```rust
                        force: [0.0, -100.0],
                        label: None,
                        body_local_app_point: None,
                        at_contact_point: false,
                    });
```

In `linkage-sim-rs/src/gui/samples/fourbar.rs`, replace:

```rust
        label: Some("Force Zone".to_string()),
        body_local_app_point: None,
    }));
```

with:

```rust
        label: Some("Force Zone".to_string()),
        body_local_app_point: None,
        at_contact_point: false,
    }));
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test force_zone_tests --test wheel_contact_sweep`
Expected: PASS (every old and new test).

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib validation_verified`
Expected: PASS, including `validation_verified_with_force_zone`, `validation_verified_force_zone_centroid_branch` and the new `validation_verified_force_zone_contact_branch`.

Then the gate. Expected: `GATE PASS`; gate 1's count = Task 1's + 9 (7 in `force_zone_tests`, 1 in `wheel_contact_sweep`, 1 in `reactions`). Restore `docs/chebyshev_lambda`.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/forces/elements/element_types.rs linkage-sim-rs/src/forces/elements/evaluation.rs linkage-sim-rs/src/solver/reactions.rs linkage-sim-rs/src/gui/canvas/rendering/force_render.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/samples/fourbar.rs linkage-sim-rs/tests/force_zone_tests.rs linkage-sim-rs/tests/wheel_contact_sweep.rs
git commit -m "feat(linkage): a force zone's contact point, one shared helper (decisions R-3, R-4, R-7)" -m "ForceZoneElement.at_contact_point: the force acts at the shape's extreme point against it (a wheel's bottom for an upward force). force_zone_application decides overlap and point for the force evaluation, the overlap ratio, the independent check and the canvas (F (contact))." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 3: Shapes and the application point in the GUI

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs:176-197`, `:1499-1505`
- Modify: `linkage-sim-rs/src/gui/property_panel/mod.rs:277-362`
- Modify: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (variants `:27-32`, handlers `:131-219`, tests)
- Modify: `linkage-sim-rs/src/gui/property_panel/force_editor.rs:903-961`
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs:287-290`
- Test: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (tests module), `linkage-sim-rs/src/gui/canvas/mod.rs` (tests module)

**Interfaces:**
- Consumes (Tasks 1, 2): `GeometryShape`, `BodyGeometry::{new, circle, centre_world, outline_world}`, `ZoneAppMode`, `ForceZoneElement::{app_mode, with_app_mode}`.
- Produces: `PendingPropertyEdit::AddGeometry { body_id: String, shape: GeometryShape }` (replaces `{ body_id, width, height }`), `SetGeometryShape { body_id: String, shape: GeometryShape }`, `UpdateGeometryDiameter { body_id: String, diameter: f64 }`; `fn default_geometry(body: &crate::io::BodyJson, shape: GeometryShape) -> BodyGeometry` and `fn edit_geometry(state: &mut AppState, body_id: &str, edit: impl Fn(&mut Option<BodyGeometry>))` (private to `pending_edits.rs`).

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace:

```rust
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }
}
```

with:

```rust
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }

    // ── Geometry shapes (decision R-2) ──────────────────────────────────

    /// `body`'s geometry in the blueprint and in the built mechanism.
    fn geometry_copies(state: &AppState, body: &str) -> [BodyGeometry; 2] {
        let bp = state.blueprint.as_ref().unwrap().bodies[body].geometry.clone();
        let mech = state.mechanism.as_ref().unwrap().bodies()[body].geometry.clone();
        [bp.expect("blueprint geometry"), mech.expect("mechanism geometry")]
    }

    /// The Four-Bar sample's coupler: its length and its midpoint (body-local).
    fn coupler_span(state: &AppState) -> (f64, [f64; 2]) {
        let points: Vec<[f64; 2]> =
            state.blueprint.as_ref().unwrap().bodies["coupler"].attachment_points.values().cloned().collect();
        assert_eq!(points.len(), 2, "the sample coupler is a two-point bar");
        let (a, b) = (points[0], points[1]);
        let span = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        (span, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
    }

    fn add_geometry(state: &mut AppState, shape: GeometryShape) {
        apply_pending(state, Some(PendingPropertyEdit::AddGeometry { body_id: "coupler".into(), shape }));
    }

    #[test]
    fn add_circle_centres_a_wheel_half_the_link_long_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span / 2.0).abs() < 1e-12 && (geo.height - span / 2.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn add_rectangle_spans_the_link_a_quarter_as_deep() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span / 4.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn switching_shape_keeps_the_width_and_squares_the_height() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, _) = coupler_span(&state);
        let set = |state: &mut AppState, shape| {
            apply_pending(state, Some(PendingPropertyEdit::SetGeometryShape { body_id: "coupler".into(), shape }));
        };
        set(&mut state, GeometryShape::Circle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
        set(&mut state, GeometryShape::Rectangle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
    }

    #[test]
    fn a_diameter_edit_sets_width_and_height_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::UpdateGeometryDiameter { body_id: "coupler".into(), diameter: 0.05 }),
        );
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!((geo.width, geo.height), (0.05, 0.05));
        }
        apply_pending(&mut state, Some(PendingPropertyEdit::RemoveGeometry { body_id: "coupler".into() }));
        assert!(state.blueprint.as_ref().unwrap().bodies["coupler"].geometry.is_none());
        assert!(state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.is_none());
    }
}
```

In `linkage-sim-rs/src/gui/canvas/mod.rs`, replace (the end of the `weight_readout` module and of the tests module, as Task BL-042 left them):

```rust
            let highlight = highlight.expect("the coupler's overlap highlight is drawn");
            let marker = marker.expect("the weight marker is drawn");
            assert!(marker > highlight, "the weight (shape {marker}) is painted under the highlight (shape {highlight})");
        }
    }
}
```

with:

```rust
            let highlight = highlight.expect("the coupler's overlap highlight is drawn");
            let marker = marker.expect("the weight marker is drawn");
            assert!(marker > highlight, "the weight (shape {marker}) is painted under the highlight (shape {highlight})");
        }
    }

    /// Headless canvas frames checking how a circle geometry and a force
    /// zone's contact point are drawn and dragged (decisions R-2, R-3, R-4).
    mod geometry_drawing {
        use eframe::egui::{self, Pos2};

        use crate::core::body::{BodyGeometry, GeometryShape};
        use crate::forces::elements::{ForceElement, ZoneAppMode};
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;
        use crate::gui::test_support::{drawn_texts, primary_button, visit_shapes};
        use super::weight_clicks::frame;

        /// The Parallelogram Press with a 20 mm wheel on its coupler where its
        /// rectangle was, the zone grown over the canvas, the zone's force at
        /// the wheel's contact point when `contact`; after two idle frames.
        fn press_with_wheel(contact: bool) -> (egui::Context, AppState) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramPress);
            let bp = state.blueprint.as_mut().expect("blueprint");
            let coupler = bp.bodies.get_mut("coupler").expect("coupler");
            let hub = coupler.geometry.as_ref().expect("the sample's geometry").offset;
            coupler.geometry = Some(BodyGeometry::circle(0.02, hub).unwrap());
            for force in &mut bp.forces {
                if let ForceElement::ForceZone(zone) = force {
                    zone.zone_min = [-10.0, -10.0];
                    zone.zone_max = [10.0, 10.0];
                    zone.at_contact_point = contact;
                }
            }
            state.rebuild();
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state)
        }

        fn zone(state: &AppState) -> crate::forces::elements::ForceZoneElement {
            state
                .mechanism
                .as_ref()
                .unwrap()
                .forces()
                .iter()
                .find_map(|f| if let ForceElement::ForceZone(z) = f { Some(z.clone()) } else { None })
                .expect("the press's zone")
        }

        #[test]
        fn a_circle_geometry_is_drawn_as_a_circle_of_its_radius() {
            let (ctx, mut state) = press_with_wheel(false);
            let output = frame(&ctx, &mut state, Vec::new());
            let geo = state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.clone().unwrap();
            assert_eq!(geo.shape, GeometryShape::Circle);
            let a = state.view.world_to_screen(0.0, 0.0);
            let b = state.view.world_to_screen(0.01, 0.0);
            let radius_px = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let mut found = false;
            visit_shapes(&output, |shape| {
                if let egui::Shape::Circle(c) = shape {
                    if c.stroke.color == egui::Color32::from_rgb(255, 165, 0) && (c.radius - radius_px).abs() < 0.5 {
                        found = true;
                    }
                }
            });
            assert!(found, "a circle of radius {radius_px} px in the geometry colour");
        }

        #[test]
        fn the_contact_point_marker_reads_f_contact() {
            let (ctx, mut state) = press_with_wheel(true);
            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            assert!(texts.iter().any(|t| t == "F (contact)"), "{texts:?}");
            assert!(!texts.iter().any(|t| t == "F (locked)"), "{texts:?}");
        }

        #[test]
        fn dragging_the_contact_marker_locks_it_where_it_drops() {
            let (ctx, mut state) = press_with_wheel(true);
            let fz = zone(&state);
            let mech = state.mechanism.as_ref().unwrap();
            let world = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the contact point");
            let s = state.view.world_to_screen(world.x, world.y);
            let from = Pos2::new(s[0], s[1]);
            let to = from + egui::vec2(40.0, 30.0);
            // egui starts the drag once the pointer passes its click distance
            // (6 px) and reports the pointer's position then, which must still
            // be within the marker's hit radius (12 px): a first step of 8 px.
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from)]);
            frame(&ctx, &mut state, vec![primary_button(from, true)]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from + egui::vec2(8.0, 0.0))]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(to)]);
            frame(&ctx, &mut state, vec![primary_button(to, false)]);
            let fz = zone(&state);
            assert_eq!(fz.app_mode(), ZoneAppMode::Locked, "the drop locks the point");
            let [wx, wy] = state.view.screen_to_world(to.x, to.y);
            let mech = state.mechanism.as_ref().unwrap();
            let dropped = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the locked point");
            assert!((dropped.x - wx).abs() < 1e-9 && (dropped.y - wy).abs() < 1e-9, "{dropped:?} vs ({wx}, {wy})");
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib geometry`
Expected: FAIL to compile (`AddGeometry { shape }`, `SetGeometryShape`, `UpdateGeometryDiameter` unknown).

- [ ] **Step 3: The edits**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace:

```rust
use nalgebra::Vector2;
use crate::core::body::BodyGeometry;
```

with:

```rust
use nalgebra::Vector2;
use crate::core::body::{BodyGeometry, GeometryShape};
```

Then replace:

```rust
    AddGeometry { body_id: String, width: f64, height: f64 },
    UpdateGeometryWidth { body_id: String, width: f64 },
```

with:

```rust
    /// Give a link without geometry a `shape` sized from the link (decision R-2).
    AddGeometry { body_id: String, shape: GeometryShape },
    /// Switch a link's geometry to `shape`, keeping its width.
    SetGeometryShape { body_id: String, shape: GeometryShape },
    /// Set a circle's diameter (width and height).
    UpdateGeometryDiameter { body_id: String, diameter: f64 },
    UpdateGeometryWidth { body_id: String, width: f64 },
```

Then replace the six geometry handlers (as Task 1 left the first one):

```rust
            PendingPropertyEdit::AddGeometry { body_id, width, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                state.mark_sweep_dirty();
            }
```

with:

```rust
            PendingPropertyEdit::AddGeometry { body_id, shape } => {
                let new = state
                    .blueprint
                    .as_ref()
                    .and_then(|bp| bp.bodies.get(&body_id))
                    .map(|body| default_geometry(body, shape));
                if let Some(new) = new {
                    edit_geometry(state, &body_id, |geo| *geo = Some(new.clone()));
                }
            }
            PendingPropertyEdit::SetGeometryShape { body_id, shape } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.shape = shape;
                        g.height = g.width;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryDiameter { body_id, diameter } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.width = diameter;
                        g.height = diameter;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.width = width;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.height = height;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.x = offset_x;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.y = offset_y;
                    }
                });
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                edit_geometry(state, &body_id, |geo| *geo = None);
            }
```

Then add the two helpers before `apply_pending`. Replace:

```rust
/// Apply a pending property edit.
pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
```

with:

```rust
/// Apply `edit` to `body_id`'s geometry in the blueprint and in the built
/// mechanism (the two copies the GUI keeps), then mark the sweep dirty.
fn edit_geometry(state: &mut AppState, body_id: &str, edit: impl Fn(&mut Option<BodyGeometry>)) {
    if let Some(body) = state.blueprint.as_mut().and_then(|bp| bp.bodies.get_mut(body_id)) {
        edit(&mut body.geometry);
    }
    if let Some(body) = state.mechanism.as_mut().and_then(|mech| mech.body_mut(body_id)) {
        edit(&mut body.geometry);
    }
    state.mark_sweep_dirty();
}

/// A new `shape` for `body` (decision R-2): centred on its attachment points'
/// centroid (body-local) and sized from their span L, the largest distance
/// between two of them (0.1 m with fewer than two): a rectangle L x L/4, a
/// circle of diameter L/2.
fn default_geometry(body: &crate::io::BodyJson, shape: GeometryShape) -> BodyGeometry {
    let points: Vec<Vector2<f64>> =
        body.attachment_points.values().map(|p| Vector2::new(p[0], p[1])).collect();
    let centre = if points.is_empty() {
        Vector2::zeros()
    } else {
        points.iter().sum::<Vector2<f64>>() / points.len() as f64
    };
    let span = points
        .iter()
        .flat_map(|a| points.iter().map(move |b| (a - b).norm()))
        .fold(0.0, f64::max);
    let span = if span > 0.0 { span } else { 0.1 };
    match shape {
        GeometryShape::Rectangle => BodyGeometry::new(span, span / 4.0, centre),
        GeometryShape::Circle => BodyGeometry::circle(span / 2.0, centre),
    }
    .expect("a positive span gives a valid shape")
}

/// Apply a pending property edit.
pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
```

- [ ] **Step 4: The property panel**

In `linkage-sim-rs/src/gui/property_panel/mod.rs`, replace:

```rust
                                    .show(ui, |ui| {
                                        if let Some(ref geo) = bp_body.geometry {
                                            // Width slider (mm display, m internal)
                                            let mut width_mm = geo.width * 1e3;
```

with:

```rust
                                    .show(ui, |ui| {
                                        if let Some(ref geo) = bp_body.geometry {
                                            // Shape switch (decision R-2).
                                            let mut shape = geo.shape;
                                            ui.horizontal(|ui| {
                                                ui.label("Shape:");
                                                ui.selectable_value(&mut shape, GeometryShape::Rectangle, "Rectangle");
                                                ui.selectable_value(&mut shape, GeometryShape::Circle, "Circle");
                                            });
                                            if shape != geo.shape {
                                                pending = Some(PendingPropertyEdit::SetGeometryShape {
                                                    body_id: body_id.clone(),
                                                    shape,
                                                });
                                            }

                                            if geo.shape == GeometryShape::Circle {
                                            // Diameter slider (mm display, m internal)
                                            let mut diameter_mm = geo.width * 1e3;
                                            let dr = ui.add(
                                                egui::Slider::new(&mut diameter_mm, 1.0..=500.0)
                                                    .text("Diameter (mm)")
                                                    .clamping(egui::SliderClamping::Never)
                                                    .logarithmic(true),
                                            ).on_hover_text("Circle diameter in mm (visual geometry for force zones)");
                                            if dr.drag_stopped() || (dr.changed() && !dr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryDiameter {
                                                    body_id: body_id.clone(),
                                                    diameter: diameter_mm * 1e-3,
                                                });
                                            }
                                            } else {
                                            // Width slider (mm display, m internal)
                                            let mut width_mm = geo.width * 1e3;
```

Then replace:

```rust
                                            if hr.drag_stopped() || (hr.changed() && !hr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryHeight {
                                                    body_id: body_id.clone(),
                                                    height: height_mm * 1e-3,
                                                });
                                            }
```

with:

```rust
                                            if hr.drag_stopped() || (hr.changed() && !hr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryHeight {
                                                    body_id: body_id.clone(),
                                                    height: height_mm * 1e-3,
                                                });
                                            }
                                            }
```

Then replace:

```rust
                                            ui.horizontal(|ui| {
                                                if ui.button("Redraw").on_hover_text("Drag on canvas to redefine this body's geometry rectangle").clicked() {
                                                    pending = Some(PendingPropertyEdit::EnterDrawGeometryMode {
                                                        body_id: body_id.clone(),
                                                    });
                                                }
                                                if ui.button("Remove").on_hover_text("Remove the visual geometry rectangle from this body").clicked() {
                                                    pending = Some(PendingPropertyEdit::RemoveGeometry {
                                                        body_id: body_id.clone(),
                                                    });
                                                }
                                            });
                                        } else {
                                            if ui.button("Draw Geometry").on_hover_text("Drag on canvas to define a geometry rectangle for this body (required for force zones)").clicked() {
                                                pending = Some(PendingPropertyEdit::EnterDrawGeometryMode {
                                                    body_id: body_id.clone(),
                                                });
                                            }
                                        }
```

with:

```rust
                                            if ui.button("Remove").on_hover_text("Remove the visual geometry from this body").clicked() {
                                                pending = Some(PendingPropertyEdit::RemoveGeometry {
                                                    body_id: body_id.clone(),
                                                });
                                            }
                                        } else {
                                            // Add a shape sized from the link (decision R-2).
                                            ui.horizontal(|ui| {
                                                if ui.button("Add rectangle").on_hover_text("Give this body a rectangle as long as the link and a quarter as deep, centred on it (required for force zones)").clicked() {
                                                    pending = Some(PendingPropertyEdit::AddGeometry {
                                                        body_id: body_id.clone(),
                                                        shape: GeometryShape::Rectangle,
                                                    });
                                                }
                                                if ui.button("Add circle").on_hover_text("Give this body a circle half the link's length across, centred on it: a wheel (required for force zones)").clicked() {
                                                    pending = Some(PendingPropertyEdit::AddGeometry {
                                                        body_id: body_id.clone(),
                                                        shape: GeometryShape::Circle,
                                                    });
                                                }
                                            });
                                        }
```

Then add the import. In the same file, find the line `use crate::core::state::GROUND_ID;` with `grep -n "^use crate::core" linkage-sim-rs/src/gui/property_panel/mod.rs` and add directly after the first `use crate::core...` line:

```rust
use crate::core::body::GeometryShape;
```

(If the file already imports from `crate::core::body`, add `GeometryShape` to that import instead; `cargo check` must report no unused or duplicate import.)

- [ ] **Step 5: The canvas**

In `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`, replace:

```rust
        // Draw body geometry rectangle (behind the link bar).
        if let Some(ref geo) = body.geometry {
            let pos = mech_state.get_position(body_id, q);
            let theta = mech_state.get_angle(body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                pos.x, pos.y, theta, geo.width, geo.height, &geo.offset,
            );
            let screen_corners: Vec<Pos2> = corners
                .iter()
                .map(|c| {
                    let sp = view.world_to_screen(c.x, c.y);
                    Pos2::new(sp[0], sp[1])
                })
                .collect();
            let geo_fill = Color32::from_rgba_premultiplied(64, 42, 0, 64);
            let geo_stroke = Stroke::new(2.0, Color32::from_rgb(255, 165, 0));
            painter.add(egui::epaint::PathShape::convex_polygon(
                screen_corners,
                geo_fill,
                geo_stroke,
            ));
        }
```

with:

```rust
        // Draw body geometry (behind the link bar): a rectangle, or a circle.
        if let Some(ref geo) = body.geometry {
            let pos = mech_state.get_position(body_id, q);
            let theta = mech_state.get_angle(body_id, q);
            let geo_fill = Color32::from_rgba_premultiplied(64, 42, 0, 64);
            let geo_stroke = Stroke::new(2.0, Color32::from_rgb(255, 165, 0));
            if geo.shape == crate::core::body::GeometryShape::Circle {
                // A true circle: its radius on screen from a world-frame offset,
                // which the view's rotation (mounting angle) does not change.
                let c = geo.centre_world(pos.x, pos.y, theta);
                let sc = view.world_to_screen(c.x, c.y);
                let se = view.world_to_screen(c.x + geo.width / 2.0, c.y);
                let radius = ((se[0] - sc[0]).powi(2) + (se[1] - sc[1]).powi(2)).sqrt();
                painter.circle(Pos2::new(sc[0], sc[1]), radius, geo_fill, geo_stroke);
            } else {
                let screen_corners: Vec<Pos2> = geo
                    .outline_world(pos.x, pos.y, theta)
                    .iter()
                    .map(|c| {
                        let sp = view.world_to_screen(c.x, c.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_corners,
                    geo_fill,
                    geo_stroke,
                ));
            }
        }
```

Then replace:

```rust
        if let Some(ref geo) = body.geometry {
            ui.label(format!(
                "Geometry: {:.1} \u{00d7} {:.1} mm",
                geo.width * 1e3,
                geo.height * 1e3
            ));
        }
```

with:

```rust
        if let Some(ref geo) = body.geometry {
            ui.label(match geo.shape {
                crate::core::body::GeometryShape::Rectangle => format!(
                    "Geometry: {:.1} \u{00d7} {:.1} mm",
                    geo.width * 1e3,
                    geo.height * 1e3
                ),
                crate::core::body::GeometryShape::Circle => {
                    format!("Geometry: circle, diameter {:.1} mm", geo.width * 1e3)
                }
            });
        }
```

- [ ] **Step 6: The force editor and the marker drop (decision R-4)**

In `linkage-sim-rs/src/gui/property_panel/force_editor.rs`, replace:

```rust
            // Application-point override: auto-centroid vs user-pinned.
            ui.separator();
            ui.label("Application point:");
            let mut locked = fz.body_local_app_point.is_some();
            if ui
                .checkbox(&mut locked, "Lock to body-local point")
                .on_hover_text(
                    "Unchecked: force applies at the overlap centroid (auto). Checked: force applies at a fixed point on the body — pin it to the real contact location.",
                )
                .changed()
            {
                let mut updated = fz.clone();
                updated.body_local_app_point = if locked {
                    // Seed the override with the current centroid so the
                    // force doesn't jump when locking.
                    fz.body_local_app_point.or(Some([0.0, 0.0]))
                } else {
                    None
                };
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(updated),
                });
            }

            if let Some(lp) = fz.body_local_app_point {
```

with:

```rust
            // Application point: overlap centre, a locked body point, or the
            // shape's contact point (decision R-4).
            ui.separator();
            ui.label("Application point:");
            let mode = fz.app_mode();
            let mut picked = mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut picked, ZoneAppMode::Centroid, "Overlap centre")
                    .on_hover_text("The force acts at the centroid of the shape's overlap with the zone, found again at every pose.");
                ui.radio_value(&mut picked, ZoneAppMode::Locked, "Locked point")
                    .on_hover_text("The force acts at a fixed point on the body: set it below, or drag the F marker on the canvas.");
                ui.radio_value(&mut picked, ZoneAppMode::Contact, "Contact point")
                    .on_hover_text("The force acts where a surface pushing in the force's direction meets the shape: its lowest point for an upward force. On a wheel it stays straight below the hub as the wheel turns.");
            });
            if picked != mode {
                // A first lock seeds the shape's centre, not the body origin.
                let seed = blueprint
                    .bodies
                    .get(&fz.body_id)
                    .and_then(|b| b.geometry.as_ref())
                    .map_or([0.0, 0.0], |g| [g.offset.x, g.offset.y]);
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(fz.with_app_mode(picked, seed)),
                });
            }

            if let (ZoneAppMode::Locked, Some(lp)) = (mode, fz.body_local_app_point) {
```

Then replace:

```rust
                if ui.button("Reset to auto (overlap centroid)")
                    .on_hover_text("Clear the locked application point and revert to the auto-computed overlap centroid each frame.")
                    .clicked()
                {
                    let mut updated = fz.clone();
                    updated.body_local_app_point = None;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ForceZone(updated),
                    });
                }
                if ap_changed {
```

with:

```rust
                if ap_changed {
```

Then add the import: run `grep -n "^use crate::forces" linkage-sim-rs/src/gui/property_panel/force_editor.rs`; in that import add `ZoneAppMode` (or, if the file uses a glob `crate::forces::elements::*`, nothing to add). `cargo check` must report no unused or missing import.

In `linkage-sim-rs/src/gui/canvas/interaction.rs`, replace:

```rust
                    let mut updated = fz.clone();
                    updated.body_local_app_point = Some(local);
                    state.update_force_element(
```

with:

```rust
                    // Dropping the marker locks the point there, whatever the
                    // zone's mode was (decision R-4).
                    let mut updated = fz.with_app_mode(
                        crate::forces::elements::ZoneAppMode::Locked,
                        local,
                    );
                    updated.body_local_app_point = Some(local);
                    state.update_force_element(
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib geometry`
Expected: PASS (the four `pending_edits` tests and the three `geometry_drawing` tests, plus older tests matching "geometry").

Then the gate. Expected: `GATE PASS`; gate 1's count = Task 2's + 7. Restore `docs/chebyshev_lambda`.

- [ ] **Step 8: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/gui/canvas/rendering/mod.rs linkage-sim-rs/src/gui/property_panel/mod.rs linkage-sim-rs/src/gui/property_panel/pending_edits.rs linkage-sim-rs/src/gui/property_panel/force_editor.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/canvas/mod.rs
git commit -m "feat(linkage): circles and the contact point in the editor and on the canvas (decisions R-2, R-4)" -m "Add rectangle / Add circle, a shape switch and Diameter in the property panel; the force editor's Overlap centre / Locked point / Contact point; the canvas draws circles; dropping the F marker locks it." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 4: Docs, backlog, the press check and the browser check

**Files:**
- Modify: `docs/reference/FORCE_ELEMENTS.md:373-399`, `docs/FEATURES.md` (before `## Planned / Future`), `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`
- Create: `docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md` (this file, copied in)

- [ ] **Step 1: The force-element reference**

In `docs/reference/FORCE_ELEMENTS.md`, replace from the line `Spatial force field defined as an axis-aligned rectangular zone. Applies a distributed force proportional to the overlap area between the zone and the body's geometry.` through the line `- Returns zero if overlap area < \`1e-15\`.` with:

````markdown
Spatial force field defined as an axis-aligned rectangular zone. Applies its full force whenever the body's geometry overlaps the zone (binary: any overlap, full force).

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_id` | String | -- | Target body (must have `BodyGeometry` set) |
| `zone_min` | [f64; 2] | m | World-space bottom-left corner of zone |
| `zone_max` | [f64; 2] | m | World-space top-right corner of zone |
| `force` | [f64; 2] | N | Force vector while the geometry overlaps the zone |
| `label` | Option | -- | Optional display label |
| `body_local_app_point` | Option<[f64; 2]> | m | Locked application point, body-local |
| `at_contact_point` | bool | -- | Apply at the shape's contact point (schema 1.2.0; default false, left out of files when false) |

**Equation:**

```
overlap = polygon_clip(geometry_outline_world, zone_AABB)   (a circle is a 64-gon here)
F = force            if area(overlap) >= 1e-15, else 0
```

**Application point** (`force_zone_application`, shared by the force evaluation, the overlap ratio, the independent equilibrium check and the canvas):
- **Contact point** (`at_contact_point`): the shape's extreme point against the force, found at every pose: for an upward force the lowest point (a wheel's bottom, straight below its hub however the wheel turns); a level rectangle edge gives the edge's midpoint. Wins over a locked point.
- **Locked point** (`body_local_app_point`): a fixed point on the body.
- **Overlap centre** (neither): the centroid of the clipped overlap polygon.

The point is converted to body-local coordinates for the generalized force mapping.

**Special behavior:**
- Body must have `BodyGeometry`: a rectangle (`width`, `height`, `offset`) or a circle (`shape: "circle"`, diameter in `width`). Returns zero if geometry is missing.
- The outline is transformed to world space, then clipped against the axis-aligned zone using Sutherland-Hodgman polygon clipping.
- Returns zero if overlap area < `1e-15`, in every application mode.
````

- [ ] **Step 2: Features**

In `docs/FEATURES.md`, replace:

```markdown
## Planned / Future
```

with:

```markdown
### Schema v1.2.0: round shapes and the contact point

- Body geometry can be a circle (a wheel): "Add circle" / "Add rectangle" in the link's Geometry section, a Rectangle / Circle switch and a Diameter slider; circles draw as circles
- Force zones apply at the overlap centre, a locked point, or the shape's contact point (a wheel's bottom for an upward force, found at every pose); the F marker reads "F (contact)"
- Older builds still open schema 1.2.0 files (circles show as their bounding squares)

## Planned / Future
```

- [ ] **Step 3: docs/ai**

Run `grep -n "Linkage canvas: FORCE_ZONE_OVERLAP_FILL" docs/ai/02-system.yaml` and add, directly above that line (same indentation, two spaces then `- "`), the entry:

```yaml
  - "Linkage force zones: force_zone_application (forces::elements) is the one place that decides a zone's overlap (binary) and its application point (overlap centroid, locked point, or the shape's contact point: BodyGeometry::extreme_point_world against the force). The force evaluation, the overlap ratio, the independent equilibrium check (reactions.rs) and the canvas marker all call it; never rebuild a body's outline for a zone elsewhere. A circle is a CIRCLE_SEGMENTS (64)-gon in the overlap test, a true circle on the canvas, and exact (centre plus radius) for the contact point."
```

In `docs/ai/03-structure.yaml`, replace (part of one line):

```yaml
    responsibility: low-level 2D geometry helpers (polygon clipping, body_rect_to_world, centroid)
```

with:

```yaml
    responsibility: low-level 2D geometry helpers (polygon clipping, body_rect_to_world, centroid); a body's shape (rectangle or circle, core::body::GeometryShape) and its world outline and contact point are BodyGeometry methods (outline_world, centre_world, extreme_point_world)
```

In `docs/ai/04-memory.yaml`, append to the end of the file (same indentation as the file's other open items; run `tail -5 docs/ai/04-memory.yaml` first and match it):

```yaml
  - "DONE 2026-10-06 (Part A of docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md, decisions R-1 to R-7): circle body geometry and a force zone's contact point (schema 1.2.0). NEXT: Part B, the animated HTML export (decisions H-1 to H-8), its code written in that plan against main after Part A merges."
```

In `docs/ai/05-update-tracker.md`, replace:

```markdown
---

## 2026-10-05 — Weights draw over force zones (BL-042, branch fix/weights-over-force-zones)
```

with:

```markdown
---

## 2026-10-06 — Round shapes and the contact point (Part A, branch linkage/round-geometry)
- `BodyGeometry.shape`: rectangle (default, left out of files) or circle (diameter in width); schema 1.2.0.
- Force zones gain a third application mode, the shape's contact point (`at_contact_point`): its extreme point against the force, found at every pose, so a wheel's load acts at its bottom. One helper, `force_zone_application`, now decides overlap and point for the evaluation, the overlap ratio, the independent check and the canvas.
- Property panel: Add rectangle / Add circle, a shape switch, Diameter; the stub Draw Geometry / Redraw buttons removed (BL-044). Force editor: Overlap centre / Locked point / Contact point; dropping the F marker locks it.

## 2026-10-05 — Weights draw over force zones (BL-042, branch fix/weights-over-force-zones)
```

In `docs/ai/backlog.yaml`, append to the end of the file:

```yaml

- id: BL-043
  title: "Scale mechanism scales a force zone's box but not its locked application point"
  dimension: physics
  risk: mechanical
  evidence: "blueprint_ops.rs scale_mechanism (2026-10-06 map of the force-zone code): it multiplies zone_min/zone_max and the body geometry by the factor but leaves ForceZoneElement.body_local_app_point unscaled, so after a scale the locked force acts at the old distance from the body origin."
  acceptance: "scaling a mechanism by k scales body_local_app_point by k; a test scales a zone with a locked point and checks the point and the statics (driver torque scales by k for a force-only load)"
  priority: 3
  status: open
  notes: "Found while planning the round-geometry work (plan 2026-10-06, Part A); not changed there."

- id: BL-044
  title: "The canvas Draw Geometry tool is a stub"
  dimension: gui
  risk: mechanical
  evidence: "interaction.rs handle_draw_body_geometry is '#[allow(dead_code)] // Stubbed: full implementation pending'; EditorTool::DrawBodyGeometry does nothing. Until 2026-10-06 the property panel's Draw Geometry and Redraw buttons entered this mode, so they did nothing; Part A of the round-geometry plan replaced them with Add rectangle / Add circle (decision R-2). The stub, DrawBodyGeometryState and PendingPropertyEdit::EnterDrawGeometryMode remain."
  acceptance: "either the tool draws a rectangle or a circle by dragging (with a headless test), or the stub, its state and the edit variant are removed"
  priority: 4
  status: open
  notes: "Decision R-2 option (b) if the user wants dragging."
```

- [ ] **Step 4: Commit this plan**

```bash
cp C:/Users/Cole/source/repos/linkage_simulation/docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md C:/Users/Cole/source/repos/lsim-shapes/docs/superpowers/plans/
```

(The controller first records the user's decision answers in this file, Task 0 Step 6.)

- [ ] **Step 5: The press check (local, not committed; decision R-7)**

The controller builds the user's press with the wheel as a circle at the hub and the contact mode (from `press_w5.json` in the session scratchpad: set `bodies.link_2.geometry.shape` to `"circle"` and the force zone's `at_contact_point` to `true`), loads it through `AppState::load_from_json_str` in a throwaway test in a scratch copy of the worktree (never committed), and checks the actuator force: 21,506 N at crank 28 deg and 5,776 N at 65 deg (within 1 N), equal to `press_w5` at every sample. The scratch copy is deleted afterwards.

- [ ] **Step 6: Gate and the browser check**

Run the gate. Expected: `GATE PASS` with Task 3's count. Restore `docs/chebyshev_lambda`. Then build and serve the web bundle (`bash linkage-sim-rs/scripts/build_web.sh`, `bash linkage-sim-rs/scripts/serve_web.sh 8765`) and run the gui-smoke workflow by its script path (`.claude/workflows/gui-smoke.js`): zero console errors. Then open the press link from Step 5 on the local server and screenshot the canvas: the wheel is a circle and "F (contact)" sits at its bottom. Kill the server by PID.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add docs/reference/FORCE_ELEMENTS.md docs/FEATURES.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md
git commit -m "docs(linkage): round shapes and the contact point; BL-043, BL-044; the plan" -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

After Task 4: the whole-branch review (session model), one fix wave if needed, merge into local `main`, and ask the user before pushing. After the push, give the user the press link with the circle and the contact mode.

---

## Part B: animated HTML export (Tasks 5 and 6; code written against `main` at `5690cb1`, decision H-8)

**What it makes:** File -> "Export Animation (HTML)..." (Title Case like its neighbours; decision H-1) saves `mechanism_animation.html`: one self-contained page (decision H-6) that plays the current mechanism through its sweep, like the user's press animation, for any mechanism with an angle or stroke sweep.

**Design ruling (plan writer, 2026-10-06): poses are rebuilt, never solved again.** The sweep already records every moving body's angle (`SweepData::body_angles`) and the world trace of every attachment point (`coupler_traces`). Each body's origin is its first attachment point's trace (points sorted by name) minus that point rotated by the body's angle, so the generalized coordinates of every sample come back exactly, with no change to `SweepData` and no second solve that could land on another branch. The hidden cylinder and rod bodies of a mount-point actuator (`forces/compound.rs`) are bodies of the built mechanism, so they are rebuilt the same way (the reactions need them); the drawing shows only the blueprint's bodies and draws the actuator pin to pin. Reactions are solved per frame with `solve_reactions_with_actuator` at the sample's driver time (`sweep_time`, extracted from the sweep so both use one formula) and equal the sweep's `joint_reaction_magnitudes` to 1e-6.

**Other rulings made while dry-running (each pinned by a test or the browser check):**
- The chart (H-2): the actuator element's required force; in a stroke sweep without one, the linear driver's own force (the sweep keeps it in `driver_torques`; the plots call it "Actuator Force"), labelled as a force with lbf; otherwise the driver torque. A linear driver draws as an actuator.
- The axis wording follows the plots: "Driver angle (deg)" (display frame, BL-041) or "Actuator stroke (mm)".
- Ground hatch marks sit at the joints' ground pivots only, so the view fits the linkage and a long actuator runs off to its anchor, as the hand-built press page did ("the actuator runs off-screen to its anchor C").
- The drawing is 720 units wide whatever the mechanism's size (labels keep one size), fitted by width or a 640-unit height and centred; arrows and labels carry a white halo so they read over links and zones.
- Forces read "21.51 kN / 4835 lbf" from 1 kN up and "12.0 N / 2.7 lbf" below (H-4), reactions included; torques to 0.01 N m; a weight share is lbf when it is a share of a force (an actuator's, or a linear driver's), N m for a crank's torque. A link's own weight is labelled by its size alone (its name is drawn beside it); payloads by name and size. Reactions are listed by joint id (the solver's order follows a hash map).
- With Bounce off the player loops to the start (the first draft stuck on the last frame).

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-anim`, branch `linkage/animation-export`, created by the controller from `main` at `5690cb1` with LF line endings and a worktree-scoped `core.autocrlf=false`; the controller writes this updated plan into it (uncommitted until Task 6), so the main checkout is never modified. Every command uses absolute paths into the worktree. Nothing is pushed without the user's go.

**Process:** Tasks 5 and 6 give the exact code, so implementers transcribe on `sonnet` with a `sonnet` review per task; the whole-branch review after Task 6 runs on the session model; then the controller's browser check, a merge into local `main`, and the user's go before any push.

**Part B global constraints** (the Part A constraints apply, with these changes): base `main` at `5690cb1` ("merge: round shapes and the contact point ..."); worktree `C:/Users/Cole/source/repos/lsim-anim`; the gate command uses `C:/Users/Cole/source/repos/lsim-anim/linkage-sim-rs/scripts/gate.sh`; commits `feat(linkage): ...` (Task 6: `feat(linkage): ...` for the menu item with its docs in the same commit); the user's press model and anything generated from it are never committed (the repo is public). The gate's linkage count at `5690cb1` is 1,079; Task 5 adds 19 tests (1,098) and Task 6 none.

**Part B review focus** (pinned by Task 5's tests):
1. **A body whose first pin is not its origin** (DXF imports, plates): its pose must come from the trace, not assume the pin sits at the origin. Every built-in sample but Strandbeest puts each body's first pin at its origin, which hides a wrong origin; `a_body_whose_first_pin_is_off_its_origin_is_placed_by_its_trace`.
2. **A mount-point actuator** (hidden cylinder and rod bodies): every sample must still rebuild (the first draft found none and exported nothing) and the actuator draws pin to pin; `frames_are_the_sweeps_solved_samples_at_their_traced_positions`, `the_actuator_draws_pin_to_pin_and_its_hidden_bodies_are_not_links`.
3. **A stroke (linear driver) sweep:** millimetres of stroke, no driver arc, the linear driver as the actuator with its force in lbf; `a_stroke_sweep_runs_in_millimetres_with_the_linear_driver_as_the_actuator`.
4. **A name containing `</script>`, or `<!--` then `<script`:** it must not end the page's script or turn its end into script text; every `<` in the data is written `\u003c`; `a_name_cannot_end_the_script_or_open_a_comment_in_it`. A renamed data field must not break the page silently; `the_data_has_every_field_the_player_reads`.
5. **Samples without a solution and a trajectory sweep:** the first are left out (H-5), the second refused with a message and the menu item disabled; `samples_without_a_solution_are_left_out`, `a_trajectory_sweep_cannot_be_animated`.

---

### Task 5: The animation export module

**Files:**
- Modify: `linkage-sim-rs/src/gui/sweep/mod.rs` (extract `sweep_time`)
- Modify: `linkage-sim-rs/src/gui/export/mod.rs` (register the module)
- Create: `linkage-sim-rs/src/gui/export/animation_template.html`
- Create: `linkage-sim-rs/src/gui/export/animation.rs` (the data builder, `generate_animation_html`, `animation_export_available`, 19 tests)

**Interfaces:**
- Consumes: `SweepData` (`angles_deg`, `body_angles`, `coupler_traces`, `actuator_forces`, `actuator_lengths`, `driver_torques`, `mechanical_advantage`, `weight_breakdown`, `sweep_mode`); `AppState::{mechanism, blueprint, sweep_data, driver_display_offset, driver_joint_id, display_units, driver_omega(), driver_theta_0()}`; `force_zone_application` and `BodyGeometry::{centre_world, outline_world}` (Part A); `weight_sources`, `gravity_vector`; `solve_reactions_with_actuator`; `test_support::swept_lift`.
- Produces: `pub(crate) fn sweep_time(x_value: f64, is_stroke: bool, omega: f64, theta_0: f64) -> f64` in `gui::sweep`; `pub fn generate_animation_html(state: &AppState) -> Result<String, String>` and `pub fn animation_export_available(state: &AppState) -> bool`, re-exported from `gui::export` (Task 6's menu item calls both).

- [ ] **Step 1: Check the worktree and run the baseline gate**

The controller created the worktree (`git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b linkage/animation-export C:/Users/Cole/source/repos/lsim-anim 5690cb1`, then `git -C C:/Users/Cole/source/repos/lsim-anim config --worktree core.autocrlf false`) and wrote this plan into it.

```bash
git -C C:/Users/Cole/source/repos/lsim-anim log --oneline -1
git -C C:/Users/Cole/source/repos/lsim-anim status --short
```

Expected: `5690cb1 merge: round shapes and the contact point ...`, and only ` M docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md`. Read `C:/Users/Cole/source/repos/lsim-anim/docs/ai/02-system.yaml` (the export entries) and `03-structure.yaml` (the `export` module). Run the gate (Part B constraints). Expected: `GATE PASS`, linkage count 1,079. Then `git -C C:/Users/Cole/source/repos/lsim-anim checkout -- docs/chebyshev_lambda`.

- [ ] **Step 2: Extract the driver time of a sweep sample**

In `linkage-sim-rs/src/gui/sweep/mod.rs`, replace:

```rust
    let ts: Vec<f64> = x_values
        .iter()
        .map(|&x_value| {
            if is_stroke {
                // f(t) = length_0 + velocity * t  =>  t = (x - length_0) / velocity
                // (omega = velocity, theta_0 = length_0 in linear mode).
                if omega.abs() > f64::EPSILON {
                    (x_value - theta_0) / omega
                } else {
                    0.0
                }
            } else {
                (x_value.to_radians() - theta_0) / omega
            }
        })
        .collect();
```

with:

```rust
    let ts: Vec<f64> = x_values.iter().map(|&x_value| sweep_time(x_value, is_stroke, omega, theta_0)).collect();
```

In the same file, replace:

```rust
pub(crate) fn compute_sweep_data(
```

with:

```rust
/// The driver time of sweep sample `x_value` (degrees in angle mode, metres
/// in stroke mode): the inverse of the driver's `f(t) = theta_0 + omega * t`.
/// In linear mode `omega` is the velocity and `theta_0` the start length; a
/// stopped linear driver gives 0.
pub(crate) fn sweep_time(x_value: f64, is_stroke: bool, omega: f64, theta_0: f64) -> f64 {
    if is_stroke {
        if omega.abs() > f64::EPSILON {
            (x_value - theta_0) / omega
        } else {
            0.0
        }
    } else {
        (x_value.to_radians() - theta_0) / omega
    }
}

pub(crate) fn compute_sweep_data(
```

Run: `cd C:/Users/Cole/source/repos/lsim-anim/linkage-sim-rs && cargo test --lib sweep`. Expected: every test passes (a pure extraction; the sweep's own tests cover it).

- [ ] **Step 3: Create the page template**

Create `linkage-sim-rs/src/gui/export/animation_template.html`:

````html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Linkage animation</title>
<style>
  :root{--slate:#22303C;--steel:#3D5A73;--ink:#1F2A36;--mute:#6B7B8A;--line:#DCE3EA;}
  *{box-sizing:border-box}
  body{margin:0;font-family:'Inter','Segoe UI',Arial,sans-serif;color:var(--ink);background:#EEF2F6;padding:18px;}
  .wrap{max-width:1180px;margin:0 auto;}
  header{margin-bottom:12px;}
  h1{font-size:20px;margin:0 0 2px;color:var(--slate);}
  .sub{font-size:12.5px;color:var(--mute);font-style:italic;margin:0;}
  .grid{display:flex;gap:14px;align-items:stretch;}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 4px rgba(31,42,54,.06);}
  .stage{flex:1.7;min-width:0;padding:6px 8px 2px;display:flex;flex-direction:column;}
  .panel{flex:1;min-width:240px;padding:14px 16px;display:flex;flex-direction:column;gap:10px;}
  .ptitle{font-size:11px;font-weight:700;letter-spacing:.06em;color:var(--steel);text-transform:uppercase;margin:0 0 2px;}
  .big{font-size:30px;font-weight:700;line-height:1;}
  .row{display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:4px 0;border-bottom:1px solid #EEF2F5;}
  .row:last-child{border-bottom:none;}
  .row .k{color:var(--mute);} .row .v{font-weight:700;font-variant-numeric:tabular-nums;text-align:right;}
  .controls{margin-top:14px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;}
  button{font-family:inherit;font-size:13px;font-weight:600;border:none;border-radius:7px;padding:8px 16px;cursor:pointer;background:var(--slate);color:#fff;}
  button.sec{background:#fff;color:var(--ink);border:1px solid var(--line);}
  .ctl{display:flex;align-items:center;gap:7px;font-size:12.5px;}
  input[type=range]{width:230px;accent-color:#C9821A;}
  select{font-family:inherit;font-size:12.5px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;}
  label.chk{display:flex;align-items:center;gap:5px;font-size:12.5px;cursor:pointer;}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--mute);margin-top:4px;}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:3px;vertical-align:-1px;}
  .foot{font-size:11px;color:var(--mute);margin-top:10px;text-align:center;}
  @media(max-width:820px){.grid{flex-direction:column;}.stage,.panel{flex:none;}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title">Linkage animation</h1>
    <p class="sub" id="sub"></p>
  </header>
  <div class="grid">
    <div class="card stage">
      <div id="svgbox"></div>
      <div class="legend">
        <span><i style="background:#3D5A73"></i>links</span>
        <span><i style="background:#C2410C"></i>shapes</span>
        <span><i style="background:#0E7C66"></i>actuator</span>
        <span><i style="background:#C0392B"></i>zone force</span>
        <span><i style="background:#6D28D9"></i>weights</span>
        <span><i style="background:#1F6FB2"></i>joint reactions</span>
        <span><i style="background:#C9821A"></i>driver angle</span>
      </div>
    </div>
    <div class="card panel">
      <div><p class="ptitle" id="xtitle"></p><div class="big" id="xval"></div></div>
      <div><p class="ptitle">Readouts</p><div id="rows"></div></div>
      <div><p class="ptitle" id="charttitle"></p><div id="chartbox"></div></div>
    </div>
  </div>
  <div class="controls">
    <button id="play">Pause</button>
    <button class="sec" id="bounce">Bounce: on</button>
    <div class="ctl"><span id="slabel">Sample</span><input type="range" id="slider" min="0" step="1" value="0"></div>
    <div class="ctl">Speed <select id="speed"><option value="20">slow</option><option value="8" selected>normal</option><option value="3">fast</option></select></div>
    <label class="chk"><input type="checkbox" id="cForces" checked> forces</label>
    <label class="chk"><input type="checkbox" id="cWeights" checked> weights</label>
    <label class="chk"><input type="checkbox" id="cArc" checked> driver angle</label>
  </div>
  <p class="foot">Exported from the linkage simulator. Statics at each solved sample of the sweep; forces in N with lbf beside them; arrow lengths scale with each kind of force.</p>
</div>
<script>
const DATA = /*__ANIMATION_DATA__*/null;
const PAL={ink:"#1F2A36",steel:"#3D5A73",shape:"#C2410C",ground:"#5B6B7A",act:"#0E7C66",force:"#C0392B",violet:"#6D28D9",blue:"#1F6FB2",amber:"#C9821A",mute:"#6B7B8A",grid:"#E7ECF1"};
const F=DATA.frames, N=F.length;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const e1=v=>Math.round(v*10)/10;
const hyp=v=>Math.hypot(v[0],v[1]);
const lbf=n=>Math.round(Math.abs(n)/DATA.n_per_lbf).toLocaleString()+" lbf";

// The view fits every frame's links, shapes, joints, weights and the joints'
// ground pivots; zones and actuator anchors outside it are clipped.
const VIEW=(()=>{let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  const add=p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);};
  DATA.ground.forEach(add);
  F.forEach(f=>{f.links.forEach(l=>l.points.forEach(add));f.joints.forEach(add);f.weights.forEach(w=>add(w.point));
    f.shapes.forEach(s=>{if(s.kind==="circle"){add([s.centre[0]-s.r,s.centre[1]-s.r]);add([s.centre[0]+s.r,s.centre[1]+s.r]);}else s.points.forEach(add);});});
  if(!isFinite(x0)){x0=-100;x1=100;y0=-100;y1=100;}
  const m=0.18*Math.max(x1-x0,y1-y0,1);
  return {x0:x0-m,x1:x1+m,y0:y0-m,y1:y1+m};})();
// A fixed 720-unit-wide drawing, so labels keep one size whatever the
// mechanism's size; the view fits its width or a 640-unit height, centred.
const PAD=20,WID=720,VW=VIEW.x1-VIEW.x0,VH=VIEW.y1-VIEW.y0,S=Math.min((WID-2*PAD)/VW,640/VH);
const HEI=Math.round(VH*S+2*PAD),OX=PAD+(WID-2*PAD-VW*S)/2;
const tx=x=>OX+(x-VIEW.x0)*S, ty=y=>HEI-PAD-(y-VIEW.y0)*S;
const P=p=>`${e1(tx(p[0]))},${e1(ty(p[1]))}`;

// Arrow lengths: 15 to 86 px, scaled by the largest force of each kind.
const maxOf=get=>Math.max(1e-9,...F.flatMap(get));
const FMAX={react:maxOf(f=>f.reactions.map(r=>hyp(r.force))),act:maxOf(f=>f.actuators.map(a=>Math.abs(a.force||0))),
  zone:maxOf(f=>f.zone_points.map(z=>hyp(z.force))),weight:maxOf(f=>f.weights.map(w=>w.newtons))};
const alen=(v,max)=>15+71*Math.min(1,Math.abs(v)/max);

function niceStep(v){const p=Math.pow(10,Math.floor(Math.log10(v)));for(const m of [1,2,5,10])if(m*p>=v)return m*p;return 10*p;}
function arrow(p,d,len,c,lab,w){const n=hyp(d);if(!(n>0))return "";
  const ux=d[0]/n,uy=-d[1]/n,x1=tx(p[0]),y1=ty(p[1]),x2=x1+ux*len,y2=y1+uy*len,ag=Math.atan2(y2-y1,x2-x1),ah=w?6:9;
  const a1=`${e1(x2-ah*Math.cos(ag-0.45))},${e1(y2-ah*Math.sin(ag-0.45))}`,a2=`${e1(x2-ah*Math.cos(ag+0.45))},${e1(y2-ah*Math.sin(ag+0.45))}`;
  // A white halo under each arrow and label keeps it readable over links and zones.
  let s=`<line x1="${e1(x1)}" y1="${e1(y1)}" x2="${e1(x2)}" y2="${e1(y2)}" stroke="#fff" stroke-width="${(w||2.6)+3}" stroke-opacity="0.9"/>`
    +`<line x1="${e1(x1)}" y1="${e1(y1)}" x2="${e1(x2)}" y2="${e1(y2)}" stroke="${c}" stroke-width="${w||2.6}"/><polygon points="${e1(x2)},${e1(y2)} ${a1} ${a2}" fill="${c}"/>`;
  if(lab){const lx=x2+(ux<0?-6:6),ly=y2+(uy>0?14:-6);
    s+=`<text x="${e1(lx)}" y="${e1(ly)}" font-size="11.5" font-weight="700" fill="${c}" stroke="#fff" stroke-width="3" paint-order="stroke" text-anchor="${ux<0?"end":"start"}">${esc(lab)}</text>`;}
  return s;}

function draw(f,o){
  let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${WID} ${HEI}" width="100%" style="display:block">`;
  s+=`<rect width="${WID}" height="${HEI}" fill="#FBFCFD"/>`;
  // The grid spans the whole drawing (the view is centred in it).
  const step=niceStep(WID/S/12),gx0=VIEW.x0-OX/S,gx1=VIEW.x0+(WID-OX)/S,gy0=VIEW.y0-PAD/S,gy1=VIEW.y1+PAD/S;
  for(let gx=Math.ceil(gx0/step)*step;gx<=gx1;gx+=step)s+=`<line x1="${e1(tx(gx))}" y1="0" x2="${e1(tx(gx))}" y2="${HEI}" stroke="${PAL.grid}"/>`;
  for(let gy=Math.ceil(gy0/step)*step;gy<=gy1;gy+=step)s+=`<line x1="0" y1="${e1(ty(gy))}" x2="${WID}" y2="${e1(ty(gy))}" stroke="${PAL.grid}"/>`;
  DATA.zones.forEach(z=>{s+=`<rect x="${e1(tx(z.min[0]))}" y="${e1(ty(z.max[1]))}" width="${e1((z.max[0]-z.min[0])*S)}" height="${e1((z.max[1]-z.min[1])*S)}" fill="${PAL.force}" fill-opacity="0.06" stroke="${PAL.force}" stroke-width="1.2" stroke-dasharray="5 4"/>`;});
  f.shapes.forEach(sh=>{s+=sh.kind==="circle"
    ?`<circle cx="${e1(tx(sh.centre[0]))}" cy="${e1(ty(sh.centre[1]))}" r="${e1(sh.r*S)}" fill="${PAL.shape}" fill-opacity="0.08" stroke="${PAL.shape}" stroke-width="1.8"/>`
    :`<polygon points="${sh.points.map(P).join(" ")}" fill="${PAL.shape}" fill-opacity="0.08" stroke="${PAL.shape}" stroke-width="1.8"/>`;});
  f.actuators.forEach(a=>{s+=`<line x1="${e1(tx(a.a[0]))}" y1="${e1(ty(a.a[1]))}" x2="${e1(tx(a.b[0]))}" y2="${e1(ty(a.b[1]))}" stroke="${PAL.act}" stroke-width="5" stroke-linecap="round" opacity="0.85"/>`;
    [a.a,a.b].forEach(p=>{s+=`<circle cx="${e1(tx(p[0]))}" cy="${e1(ty(p[1]))}" r="3.5" fill="#fff" stroke="${PAL.act}" stroke-width="2"/>`;});});
  f.links.forEach(l=>{const pts=l.points.map(P).join(" ");
    s+=l.closed?`<polygon points="${pts}" fill="${PAL.steel}" fill-opacity="0.12" stroke="${PAL.steel}" stroke-width="4" stroke-linejoin="round"/>`
      :`<polyline points="${pts}" fill="none" stroke="${PAL.steel}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;
    if(l.points.length){const n=l.points.length,c=l.points.reduce((a,p)=>[a[0]+p[0]/n,a[1]+p[1]/n],[0,0]);
      s+=`<text x="${e1(tx(c[0]))}" y="${e1(ty(c[1])-9)}" font-size="11" fill="${PAL.steel}" stroke="#fff" stroke-width="3" paint-order="stroke" text-anchor="middle">${esc(l.name)}</text>`;}});
  if(o.arc&&DATA.driver_pivot&&f.driver_angle!=null){const R=34,x0=tx(DATA.driver_pivot[0]),y0=ty(DATA.driver_pivot[1]);
    const d=((f.driver_angle%360)+360)%360,a=d*Math.PI/180;
    s+=`<path d="M ${e1(x0+R)} ${e1(y0)} A ${R} ${R} 0 ${d>180?1:0} 0 ${e1(x0+R*Math.cos(a))} ${e1(y0-R*Math.sin(a))}" fill="none" stroke="${PAL.amber}" stroke-width="2.2"/>`;
    s+=`<line x1="${e1(x0)}" y1="${e1(y0)}" x2="${e1(x0+R+10)}" y2="${e1(y0)}" stroke="${PAL.amber}" stroke-dasharray="2 3"/>`;}
  DATA.ground.forEach(p=>{const x=tx(p[0]),y=ty(p[1]);
    s+=`<line x1="${e1(x-13)}" y1="${e1(y+8)}" x2="${e1(x+13)}" y2="${e1(y+8)}" stroke="${PAL.ground}" stroke-width="2"/>`;
    for(let i=-10;i<=10;i+=5)s+=`<line x1="${e1(x+i)}" y1="${e1(y+8)}" x2="${e1(x+i-4)}" y2="${e1(y+14)}" stroke="${PAL.ground}" stroke-width="1.4"/>`;});
  f.joints.forEach(p=>{s+=`<circle cx="${e1(tx(p[0]))}" cy="${e1(ty(p[1]))}" r="5.5" fill="#fff" stroke="${PAL.ink}" stroke-width="2"/>`;});
  // A link's own weight is labelled by its size alone: the link's name is beside it.
  if(o.weights)f.weights.forEach(w=>{s+=arrow(w.point,DATA.gravity,alen(w.newtons,FMAX.weight),PAL.violet,w.link_self_weight?lbf(w.newtons):`${w.name} ${lbf(w.newtons)}`,1.8)
    +`<circle cx="${e1(tx(w.point[0]))}" cy="${e1(ty(w.point[1]))}" r="4" fill="${PAL.violet}"/>`;});
  if(o.forces){
    f.zone_points.forEach(z=>{s+=`<circle cx="${e1(tx(z.point[0]))}" cy="${e1(ty(z.point[1]))}" r="5" fill="#fff" stroke="${PAL.force}" stroke-width="2.2"/>`;
      if(z.active)s+=arrow(z.point,z.force,alen(hyp(z.force),FMAX.zone),PAL.force,`F ${lbf(hyp(z.force))}`);});
    f.actuators.forEach(a=>{if(a.force==null)return;const sg=a.force<0?-1:1;
      s+=arrow(a.b,[sg*(a.b[0]-a.a[0]),sg*(a.b[1]-a.a[1])],alen(a.force,FMAX.act),PAL.act,`${lbf(a.force)} ${a.force<0?"pull":"push"}`);});
    f.reactions.forEach(r=>{s+=arrow(r.point,r.force,alen(hyp(r.force),FMAX.react),PAL.blue,r.name);});}
  return s+`</svg>`;}

// The chart: the actuator force (or driver torque) across the sweep, broken
// where samples are missing (no solution).
const CW=320,CH=150,CP=44;
const XS=F.map(f=>f.x),XMIN=Math.min(...XS),XMAX=Math.max(...XS);
const CV=F.map(f=>f.chart).filter(v=>v!=null),CMIN=Math.min(0,...CV),CMAX=Math.max(0,...CV);
const STEP=(()=>{let m=Infinity;for(let i=1;i<N;i++){const d=Math.abs(XS[i]-XS[i-1]);if(d>0)m=Math.min(m,d);}return m;})();
const cx=x=>CP+(XMAX>XMIN?(x-XMIN)/(XMAX-XMIN):0.5)*(CW-CP-8);
const cy=v=>CH-22-((v-CMIN)/((CMAX-CMIN)||1))*(CH-32);
const short=v=>Math.abs(v)>=1000?(v/1000).toFixed(1)+"k":v.toFixed(Math.abs(v)<10?2:0);
function chart(k){
  let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CW} ${CH}" width="100%" style="display:block"><rect width="${CW}" height="${CH}" fill="#fff"/>`;
  s+=`<line x1="${CP}" y1="${e1(cy(0))}" x2="${CW-8}" y2="${e1(cy(0))}" stroke="#DCE3EA"/>`;
  [CMIN,CMAX].forEach(v=>{s+=`<text x="${CP-4}" y="${e1(cy(v)+3)}" font-size="9" fill="${PAL.mute}" text-anchor="end">${short(v)}</text>`;});
  [[XMIN,"start"],[XMAX,"end"]].forEach(([x,a])=>{s+=`<text x="${e1(cx(x))}" y="${CH-6}" font-size="9" fill="${PAL.mute}" text-anchor="${a}">${x.toFixed(1)}</text>`;});
  let seg=[];const segs=[];
  F.forEach((f,i)=>{const gap=i>0&&Math.abs(XS[i]-XS[i-1])>1.5*STEP;
    if(f.chart==null||gap){if(seg.length)segs.push(seg);seg=[];}
    if(f.chart!=null)seg.push(`${e1(cx(f.x))},${e1(cy(f.chart))}`);});
  if(seg.length)segs.push(seg);
  segs.forEach(g=>{s+=`<polyline points="${g.join(" ")}" fill="none" stroke="${PAL.steel}" stroke-width="2"/>`;});
  const f=F[k],x=cx(f.x);
  s+=`<line x1="${e1(x)}" y1="8" x2="${e1(x)}" y2="${CH-22}" stroke="${PAL.mute}" stroke-dasharray="2 3"/>`;
  if(f.chart!=null)s+=`<circle cx="${e1(x)}" cy="${e1(cy(f.chart))}" r="4.5" fill="${PAL.force}" stroke="#fff" stroke-width="1.5"/>`;
  return s+`</svg>`;}

const $=id=>document.getElementById(id);
const svgbox=$("svgbox"),chartbox=$("chartbox"),rows=$("rows"),slider=$("slider"),speed=$("speed");
const cF=$("cForces"),cW=$("cWeights"),cA=$("cArc"),playBtn=$("play"),bounceBtn=$("bounce");
document.title=DATA.title;$("title").textContent=DATA.title;
$("sub").textContent=`${N} samples, ${DATA.x_label} ${XMIN.toFixed(1)} to ${XMAX.toFixed(1)}`;
$("xtitle").textContent=DATA.x_label;$("charttitle").textContent=DATA.chart_label;$("slabel").textContent=DATA.x_label;
slider.max=Math.max(0,N-1);
let pos=0,dir=1,playing=N>1,bounce=true,last=null;
playBtn.textContent=playing?"Pause":"Play";
function render(){const k=Math.round(pos),f=F[k];
  svgbox.innerHTML=draw(f,{forces:cF.checked,weights:cW.checked,arc:cA.checked});
  chartbox.innerHTML=chart(k);
  $("xval").textContent=`${f.x.toFixed(1)} ${DATA.x_unit}`;
  rows.innerHTML=f.readouts.map(r=>`<div class="row"><span class="k">${esc(r[0])}</span><span class="v">${esc(r[1])}</span></div>`).join("");
  slider.value=k;}
function tick(ts){
  if(playing){
    if(last!==null){if(!bounce)dir=1;
      pos+=dir*(N-1)*((ts-last)/1000)/parseFloat(speed.value);
      if(bounce){if(pos>=N-1){pos=N-1;dir=-1;}else if(pos<=0){pos=0;dir=1;}}
      else if(pos>N-1)pos=0;}
    last=ts;render();
  } else last=null;
  requestAnimationFrame(tick);}
playBtn.onclick=()=>{playing=!playing&&N>1;playBtn.textContent=playing?"Pause":"Play";};
bounceBtn.onclick=()=>{bounce=!bounce;bounceBtn.textContent="Bounce: "+(bounce?"on":"off");};
slider.oninput=()=>{playing=false;playBtn.textContent="Play";pos=parseInt(slider.value,10);render();};
[cF,cW,cA].forEach(c=>c.onchange=render);
render();
requestAnimationFrame(tick);
</script>
</body>
</html>
````

- [ ] **Step 4: Create the module with its tests**

Create `linkage-sim-rs/src/gui/export/animation.rs`:

````rust
//! Animated HTML export (plan 2026-10-06 Part B, decisions H-1 to H-8): the
//! current mechanism through every solved sample of its sweep, drawn by a
//! small JS player in one self-contained page (`animation_template.html`).
//!
//! Poses are never solved again: each moving body's pose (the hidden cylinder
//! and rod of a mount-point actuator too) is rebuilt from its angle and the
//! world trace of its first attachment point, both of which the sweep records
//! (`SweepData::body_angles`, `coupler_traces`), so the page shows exactly the
//! samples the plots show. Samples with no solution are
//! left out (decision H-5). Forces come from the helpers the canvas and the
//! sweep use; lengths are millimetres, forces newtons (lbf in the labels,
//! decision H-4).

use nalgebra::{DVector, Vector2};
use serde::Serialize;

use crate::analysis::gravity_breakdown::{gravity_vector, weight_sources};
use crate::core::body::GeometryShape;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::{force_zone_application, ForceElement};
use crate::gui::state::{AppState, LengthUnit};
use crate::gui::sweep::{sweep_time, ShareBasis, SweepData};
use crate::io::{JointJson, MechanismJson};
use crate::solver::reactions::solve_reactions_with_actuator;

/// Newtons per pound-force (decision H-4: lbf beside every force).
const N_PER_LBF: f64 = 4.4482216152605;

/// The page, with `DATA_SLOT` where the JSON goes.
const TEMPLATE: &str = include_str!("animation_template.html");
/// The template's placeholder for the data (valid JS before the swap).
const DATA_SLOT: &str = "/*__ANIMATION_DATA__*/null";

/// Everything the page draws: millimetres in the world frame, newtons.
#[derive(Debug, Serialize)]
pub(crate) struct AnimationData {
    pub title: String,
    /// The driver axis, as the app's plots name it: "Driver angle (deg)" or "Actuator stroke (mm)".
    pub x_label: String,
    /// "deg" or "mm".
    pub x_unit: String,
    /// "Actuator force (N)" (an actuator's, or a linear driver's own), else
    /// "Driver torque (N m)" (decision H-2).
    pub chart_label: String,
    pub n_per_lbf: f64,
    /// The unit vector of gravity; weights hang along it.
    pub gravity: [f64; 2],
    /// The ground pivots of the joints (the hatch marks). An actuator's
    /// anchor is not one, so the view fits the linkage and the actuator runs
    /// off to its anchor, as the hand-built press page did.
    pub ground: Vec<[f64; 2]>,
    /// The driver's fixed pivot, for the driver-angle arc (angle sweeps only).
    pub driver_pivot: Option<[f64; 2]>,
    pub zones: Vec<ZoneBox>,
    /// One frame per solved sweep sample.
    pub frames: Vec<Frame>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ZoneBox {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

#[derive(Debug, Serialize)]
pub(crate) struct Frame {
    /// The driver value as the app shows it: degrees (display frame, BL-041)
    /// or millimetres of stroke.
    pub x: f64,
    /// The chart's value here; `None` where the sweep has none.
    pub chart: Option<f64>,
    /// The driver link's angle for the arc (angle sweeps only), degrees.
    pub driver_angle: Option<f64>,
    pub links: Vec<Link>,
    pub shapes: Vec<Shape>,
    pub joints: Vec<[f64; 2]>,
    pub actuators: Vec<Actuator>,
    pub zone_points: Vec<ZonePoint>,
    pub weights: Vec<Weight>,
    pub reactions: Vec<Reaction>,
    /// The readout panel's rows: (label, value) in the app's units, lbf beside forces.
    pub readouts: Vec<[String; 2]>,
}

/// A moving body: its attachment points in name order.
#[derive(Debug, Serialize)]
pub(crate) struct Link {
    pub name: String,
    pub points: Vec<[f64; 2]>,
    /// Three or more points draw as a plate.
    pub closed: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Shape {
    Polygon { points: Vec<[f64; 2]> },
    Circle { centre: [f64; 2], r: f64 },
}

#[derive(Debug, Serialize)]
pub(crate) struct Actuator {
    pub a: [f64; 2],
    pub b: [f64; 2],
    /// The sweep's force (N, positive = push): the first actuator element's
    /// required force, or in a stroke sweep the first linear driver's.
    pub force: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ZonePoint {
    pub point: [f64; 2],
    pub force: [f64; 2],
    /// The geometry overlaps the zone, so the force applies.
    pub active: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct Weight {
    pub name: String,
    pub point: [f64; 2],
    pub newtons: f64,
    /// A link's own weight (at its centre of gravity), not a payload.
    pub link_self_weight: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct Reaction {
    /// The joint's id; `name` is its label when it has one.
    pub id: String,
    pub name: String,
    pub point: [f64; 2],
    /// The reaction force in the world frame (N), as the canvas draws it.
    pub force: [f64; 2],
}

/// Whether the animation export can run: a mechanism and an angle or stroke
/// sweep. The File menu enables its item by this.
pub fn animation_export_available(state: &AppState) -> bool {
    state.mechanism.is_some() && state.sweep_data.as_ref().is_some_and(|s| !s.sweep_mode.is_trajectory())
}

/// The self-contained animated page of the current mechanism (decision H-6:
/// no network). Every `<` in the JSON is written `\u003c`, the same string to
/// JSON and to JS, so no name can end the script (`</script>`) or switch the
/// HTML parser into a script comment (`<!--` then `<script`).
pub fn generate_animation_html(state: &AppState) -> Result<String, String> {
    let data = animation_data(state)?;
    let json = serde_json::to_string(&data).map_err(|e| e.to_string())?;
    Ok(TEMPLATE.replacen(DATA_SLOT, &json.replace('<', "\\u003c"), 1))
}

/// The page's data: one frame per solved sample of the current sweep.
pub(crate) fn animation_data(state: &AppState) -> Result<AnimationData, String> {
    let (Some(mech), Some(bp)) = (state.mechanism.as_ref(), state.blueprint.as_ref()) else {
        return Err("No mechanism loaded".to_string());
    };
    let Some(sweep) = state.sweep_data.as_ref() else {
        return Err("No sweep computed".to_string());
    };
    if sweep.sweep_mode.is_trajectory() {
        return Err("The animation export needs an angle or stroke sweep".to_string());
    }
    let is_stroke = sweep.sweep_mode.is_stroke();
    // The chart (decision H-2): an actuator element's required force; in a
    // stroke sweep without one, the linear driver's own force (the sweep keeps
    // it in `driver_torques`, which the plots then call the actuator force);
    // else the driver torque.
    let (chart, chart_is_force) = match (&sweep.actuator_forces, is_stroke) {
        (Some(_), _) => (&sweep.actuator_forces, true),
        (None, true) => (&sweep.driver_torques, true),
        (None, false) => (&sweep.driver_torques, false),
    };
    let g = gravity_vector(mech);
    let g_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
    let gravity = if g_norm > 0.0 { [g[0] / g_norm, g[1] / g_norm] } else { [0.0, -1.0] };
    let driver_pivot =
        if is_stroke { None } else { state.driver_joint_id.as_ref().and_then(|id| ground_pivot(bp, id)).map(mm) };
    let ctx = FrameContext { state, mech, bp, sweep, is_stroke, chart, chart_is_force, g_norm, driver_pivot };

    let frames: Vec<Frame> = (0..sweep.angles_deg.len())
        .filter_map(|i| sample_q(mech, sweep, i).map(|q| frame(&ctx, i, &q)))
        .collect();
    if frames.is_empty() {
        return Err("The sweep has no solved sample to animate".to_string());
    }

    let mut joint_ids: Vec<&String> = bp.joints.keys().collect();
    joint_ids.sort();
    let mut ground: Vec<[f64; 2]> = Vec::new();
    for p in joint_ids.iter().filter_map(|id| ground_pivot(bp, id)).map(mm) {
        if !ground.contains(&p) {
            ground.push(p);
        }
    }
    let zones = mech
        .forces()
        .iter()
        .filter_map(|f| match f {
            ForceElement::ForceZone(z) => Some(ZoneBox {
                min: mm(xy(z.zone_min)),
                max: mm(xy(z.zone_max)),
            }),
            _ => None,
        })
        .collect();
    let (x_label, x_unit) =
        if is_stroke { ("Actuator stroke (mm)", "mm") } else { ("Driver angle (deg)", "deg") };
    let chart_label = if chart_is_force { "Actuator force (N)" } else { "Driver torque (N m)" };
    Ok(AnimationData {
        title: format!("Linkage animation: {} links, {} samples", frames[0].links.len(), frames.len()),
        x_label: x_label.to_string(),
        x_unit: x_unit.to_string(),
        chart_label: chart_label.to_string(),
        n_per_lbf: N_PER_LBF,
        gravity,
        ground,
        driver_pivot,
        zones,
        frames,
    })
}

/// What every frame reads.
struct FrameContext<'a> {
    state: &'a AppState,
    mech: &'a Mechanism,
    bp: &'a MechanismJson,
    sweep: &'a SweepData,
    is_stroke: bool,
    /// The chart's series, and whether it is a force (N) or a torque (N m).
    chart: &'a Option<Vec<f64>>,
    chart_is_force: bool,
    g_norm: f64,
    driver_pivot: Option<[f64; 2]>,
}

/// The generalized coordinates of sweep sample `i`, rebuilt from the sweep's
/// own records: each moving body's angle and the world trace of its first
/// attachment point (by name) give its origin. Every body of the built
/// mechanism counts, the blueprint's and the hidden actuator bodies alike.
/// `None` for a sample with no solution (its traces are NaN).
fn sample_q(mech: &Mechanism, sweep: &SweepData, i: usize) -> Option<DVector<f64>> {
    let state = mech.state();
    let mut q = state.make_q();
    for body_id in mech.body_order() {
        let theta = sweep.body_angles.get(body_id)?.get(i)?.to_radians();
        let (name, local) = mech.bodies().get(body_id)?.attachment_points.iter().min_by(|a, b| a.0.cmp(b.0))?;
        let traced = sweep.coupler_traces.get(&format!("{body_id}.{name}"))?.get(i)?;
        if !(theta.is_finite() && traced[0].is_finite() && traced[1].is_finite()) {
            return None;
        }
        let (sin_t, cos_t) = theta.sin_cos();
        let idx = state.get_index(body_id).ok()?;
        q[idx.x_idx()] = traced[0] - (cos_t * local.x - sin_t * local.y);
        q[idx.y_idx()] = traced[1] - (sin_t * local.x + cos_t * local.y);
        q[idx.theta_idx()] = theta;
    }
    Some(q)
}

/// One frame of sample `i` at its rebuilt coordinates `q`.
fn frame(ctx: &FrameContext, i: usize, q: &DVector<f64>) -> Frame {
    let FrameContext { state, mech, bp, sweep, .. } = *ctx;
    let raw_x = sweep.angles_deg[i];
    let x = if ctx.is_stroke { raw_x * 1e3 } else { raw_x + state.driver_display_offset.to_degrees() };
    let world = |body: &str, local: [f64; 2]| mech.state().body_point_global(body, &xy(local), q);

    // The blueprint's bodies only: an actuator's hidden bodies draw as the actuator.
    let links: Vec<Link> = mech
        .body_order()
        .iter()
        .filter_map(|body_id| {
            let body = bp.bodies.get(body_id)?;
            let mut names: Vec<&String> = body.attachment_points.keys().collect();
            names.sort();
            let points: Vec<[f64; 2]> =
                names.iter().map(|n| mm(world(body_id, body.attachment_points[*n]))).collect();
            Some(Link {
                name: body.label.clone().unwrap_or_else(|| body_id.clone()),
                closed: points.len() >= 3,
                points,
            })
        })
        .collect();

    let shapes: Vec<Shape> = mech
        .body_order()
        .iter()
        .filter_map(|body_id| {
            let geo = mech.bodies().get(body_id)?.geometry.as_ref()?;
            let (bx, by, th) = mech.state().get_pose(body_id, q);
            Some(match geo.shape {
                GeometryShape::Circle => Shape::Circle {
                    centre: mm(geo.centre_world(bx, by, th)),
                    r: geo.width / 2.0 * 1e3,
                },
                GeometryShape::Rectangle => Shape::Polygon {
                    points: geo.outline_world(bx, by, th).into_iter().map(mm).collect(),
                },
            })
        })
        .collect();

    let mut joint_ids: Vec<&String> = bp.joints.keys().collect();
    joint_ids.sort();
    let joint_point = |id: &str| match bp.joints.get(id)? {
        JointJson::Revolute { body_i, point_i, .. } | JointJson::Fixed { body_i, point_i, .. } => {
            let local = *bp.bodies.get(body_i)?.attachment_points.get(point_i)?;
            Some(world(body_i, local))
        }
        _ => None,
    };
    let joints: Vec<[f64; 2]> = joint_ids.iter().filter_map(|id| joint_point(id)).map(mm).collect();

    let mut actuators: Vec<Actuator> = Vec::new();
    for la in mech.forces().iter().filter_map(|f| match f {
        ForceElement::LinearActuator(la) => Some(la),
        _ => None,
    }) {
        let force = if actuators.is_empty() { at(&sweep.actuator_forces, i) } else { None };
        actuators.push(Actuator { a: mm(world(&la.body_a, la.point_a)), b: mm(world(&la.body_b, la.point_b)), force });
    }
    // A linear driver is the actuator the user sees; in a stroke sweep the
    // first one's force is the sweep's driver force.
    for (k, ld) in bp.linear_drivers.iter().enumerate() {
        let force = if k == 0 && ctx.is_stroke { at(&sweep.driver_torques, i) } else { None };
        actuators.push(Actuator { a: mm(world(&ld.body_a, ld.point_a)), b: mm(world(&ld.body_b, ld.point_b)), force });
    }

    let zone_points: Vec<ZonePoint> = mech
        .forces()
        .iter()
        .filter_map(|f| match f {
            ForceElement::ForceZone(fz) => {
                let geo = mech.bodies().get(&fz.body_id)?.geometry.as_ref()?;
                let app = force_zone_application(fz, geo, mech.state().get_pose(&fz.body_id, q));
                Some(ZonePoint { point: mm(app.point?), force: fz.force, active: app.active })
            }
            _ => None,
        })
        .collect();

    let weights: Vec<Weight> = weight_sources(bp)
        .into_iter()
        .map(|w| Weight {
            point: mm(world(&w.body_id, w.local_pos)),
            newtons: w.mass * ctx.g_norm,
            link_self_weight: w.is_link_self_weight,
            name: w.name,
        })
        .collect();

    let t = sweep_time(raw_x, ctx.is_stroke, state.driver_omega(), state.driver_theta_0());
    let mut reactions: Vec<Reaction> = match solve_reactions_with_actuator(mech, q, t, state.driver_omega()) {
        Ok(r) => r
            .reactions
            .iter()
            .filter_map(|jr| {
                let point = joint_point(&jr.joint_id)?;
                let label = match bp.joints.get(&jr.joint_id)? {
                    JointJson::Revolute { label, .. } | JointJson::Fixed { label, .. } => label.clone(),
                    _ => None,
                };
                Some(Reaction {
                    id: jr.joint_id.clone(),
                    name: label.unwrap_or_else(|| jr.joint_id.clone()),
                    point: mm(point),
                    force: jr.force_global,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    // By id: the same order in every export (the solver's follows a hash map).
    reactions.sort_by(|a, b| a.id.cmp(&b.id));

    let chart = at(ctx.chart, i);
    let readouts = readouts(ctx, i, x, &reactions);
    Frame {
        x,
        chart,
        driver_angle: ctx.driver_pivot.map(|_| x),
        links,
        shapes,
        joints,
        actuators,
        zone_points,
        weights,
        reactions,
        readouts,
    }
}

/// The readout rows of sample `i`.
fn readouts(ctx: &FrameContext, i: usize, x: f64, reactions: &[Reaction]) -> Vec<[String; 2]> {
    let sweep = ctx.sweep;
    let mut rows = Vec::new();
    rows.push(if ctx.is_stroke {
        ["Actuator stroke".to_string(), format!("{x:.1} mm")]
    } else {
        ["Driver angle".to_string(), format!("{x:.1} deg")]
    });
    if let Some(v) = at(ctx.chart, i) {
        rows.push(if ctx.chart_is_force {
            let way = if v < 0.0 { "pull" } else { "push" };
            ["Actuator force".to_string(), format!("{} ({way})", force_text(v))]
        } else {
            ["Driver torque".to_string(), format!("{v:.2} N m")]
        });
    }
    if let Some(len) = at(&sweep.actuator_lengths, i) {
        rows.push(["Actuator length".to_string(), length_text(len, ctx.state.display_units.length)]);
    }
    if let Some(ma) = sweep.mechanical_advantage.get(i).copied().filter(|v| v.is_finite()) {
        rows.push(["Mech. advantage".to_string(), format!("{ma:.3}")]);
    }
    if let Some(bd) = sweep.weight_breakdown.as_ref() {
        for (k, source) in bd.sources.iter().enumerate() {
            let Some(share) = bd.force_share.get(k).and_then(|s| s.get(i)).copied().filter(|v| v.is_finite())
            else {
                continue;
            };
            // A share of a force (an actuator's, or a linear driver's in a
            // stroke sweep) is newtons; of a crank's torque, newton metres.
            let value = if matches!(bd.basis, ShareBasis::ActuatorForce) || ctx.is_stroke {
                format!("{:+.0} lbf", share / N_PER_LBF)
            } else {
                format!("{share:+.2} N m")
            };
            rows.push([format!("{} share", source.name), value]);
        }
    }
    for r in reactions {
        let magnitude = (r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt();
        rows.push([format!("Reaction {}", r.name), force_text(magnitude)]);
    }
    rows
}

/// The ground side of joint `id` (metres; ground's frame is the world's),
/// when it is a revolute or fixed joint on ground: the driver's pivot, and
/// the hatch marks.
fn ground_pivot(bp: &MechanismJson, id: &str) -> Option<Vector2<f64>> {
    let (JointJson::Revolute { body_i, body_j, point_i, point_j, .. }
    | JointJson::Fixed { body_i, body_j, point_i, point_j, .. }) = bp.joints.get(id)?
    else {
        return None;
    };
    let point = if body_i == GROUND_ID {
        point_i
    } else if body_j == GROUND_ID {
        point_j
    } else {
        return None;
    };
    Some(xy(*bp.bodies.get(GROUND_ID)?.attachment_points.get(point)?))
}

/// A world point (metres) in millimetres, to the micrometre.
fn mm(p: Vector2<f64>) -> [f64; 2] {
    [(p.x * 1e6).round() / 1e3, (p.y * 1e6).round() / 1e3]
}

fn xy(p: [f64; 2]) -> Vector2<f64> {
    Vector2::new(p[0], p[1])
}

/// Sample `i` of an optional sweep series, if finite.
fn at(series: &Option<Vec<f64>>, i: usize) -> Option<f64> {
    series.as_ref()?.get(i).copied().filter(|v| v.is_finite())
}

/// A force's size with its lbf beside it (decision H-4): kN from 1 kN up,
/// newtons below, so a small mechanism's forces do not read as zero.
fn force_text(newtons: f64) -> String {
    let (n, lbf) = (newtons.abs(), newtons.abs() / N_PER_LBF);
    if n >= 1e3 {
        format!("{:.2} kN / {lbf:.0} lbf", n / 1e3)
    } else {
        format!("{n:.1} N / {lbf:.1} lbf")
    }
}

fn length_text(metres: f64, unit: LengthUnit) -> String {
    match unit {
        LengthUnit::Millimeters => format!("{:.1} mm", metres * 1e3),
        LengthUnit::Meters => format!("{metres:.4} m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::BodyGeometry;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::test_support::swept_lift;

    /// The sweep samples with a solution (every body angle finite).
    fn solved(sweep: &SweepData) -> Vec<usize> {
        (0..sweep.angles_deg.len())
            .filter(|&i| sweep.body_angles.values().all(|a| a[i].is_finite()))
            .collect()
    }

    /// Every frame is a solved sample of `state`'s sweep, with each link's
    /// points (name order) where the sweep traced them.
    fn assert_frames_follow_the_traces(state: &AppState) {
        let data = animation_data(state).unwrap();
        let (sweep, bp, mech) =
            (state.sweep_data.as_ref().unwrap(), state.blueprint.as_ref().unwrap(), state.mechanism.as_ref().unwrap());
        let samples = solved(sweep);
        assert!(samples.len() > 300, "only {} samples solved", samples.len());
        assert_eq!(data.frames.len(), samples.len());
        let drawn: Vec<&String> = mech.body_order().iter().filter(|id| bp.bodies.contains_key(*id)).collect();
        for (frame, &i) in data.frames.iter().zip(&samples) {
            assert_eq!(frame.links.len(), drawn.len());
            for (link, &body_id) in frame.links.iter().zip(&drawn) {
                let mut names: Vec<&String> = bp.bodies[body_id].attachment_points.keys().collect();
                names.sort();
                for (p, name) in link.points.iter().zip(names) {
                    let traced = sweep.coupler_traces[&format!("{body_id}.{name}")][i];
                    assert!(
                        (p[0] - traced[0] * 1e3).abs() < 2e-3 && (p[1] - traced[1] * 1e3).abs() < 2e-3,
                        "sample {i} {body_id}.{name}: {p:?} vs {traced:?} m"
                    );
                }
            }
        }
    }

    #[test]
    fn frames_are_the_sweeps_solved_samples_at_their_traced_positions() {
        assert_frames_follow_the_traces(&swept_lift());
    }

    #[test]
    fn a_body_whose_first_pin_is_off_its_origin_is_placed_by_its_trace() {
        // Every other sample puts each body's first pin (by name) at the body's
        // origin, which would hide a wrong origin in the pose rebuild.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::Strandbeest);
        state.compute_sweep();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(
            mech.body_order().iter().any(|id| {
                let (_, p) = mech.bodies()[id].attachment_points.iter().min_by(|a, b| a.0.cmp(b.0)).unwrap();
                p.norm() > 1e-3
            }),
            "the fixture has a body whose first pin is off its origin"
        );
        assert_frames_follow_the_traces(&state);
    }

    #[test]
    fn the_actuator_draws_pin_to_pin_and_its_hidden_bodies_are_not_links() {
        // The lift's actuator mounts on a mount point, so the built mechanism
        // carries its cylinder and rod as bodies the blueprint does not have.
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        let bp = state.blueprint.as_ref().unwrap();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(mech.body_order().iter().any(|id| !bp.bodies.contains_key(id)), "the fixture has hidden bodies");
        let ForceElement::LinearActuator(la) =
            mech.forces().iter().find(|f| matches!(f, ForceElement::LinearActuator(_))).unwrap()
        else {
            unreachable!()
        };
        // The traced attachment point at each end of the actuator.
        let pin_trace = |body: &str, local: [f64; 2]| {
            let (name, _) = mech.bodies()[body]
                .attachment_points
                .iter()
                .find(|(_, p)| (p.x - local[0]).abs() < 1e-12 && (p.y - local[1]).abs() < 1e-12)
                .unwrap_or_else(|| panic!("{body} has a pin at {local:?}"));
            &sweep.coupler_traces[&format!("{body}.{name}")]
        };
        let ends = [(pin_trace(&la.body_a, la.point_a), "a"), (pin_trace(&la.body_b, la.point_b), "b")];
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            let mut names: Vec<&str> = frame.links.iter().map(|l| l.name.as_str()).collect();
            names.sort();
            let mut want: Vec<String> = mech
                .body_order()
                .iter()
                .filter_map(|id| bp.bodies.get(id).map(|b| b.label.clone().unwrap_or_else(|| id.clone())))
                .collect();
            want.sort();
            assert_eq!(names, want);
            for (trace, end) in ends {
                let (got, traced) = (if end == "a" { frame.actuators[0].a } else { frame.actuators[0].b }, trace[i]);
                assert!(
                    (got[0] - traced[0] * 1e3).abs() < 2e-3 && (got[1] - traced[1] * 1e3).abs() < 2e-3,
                    "sample {i} end {end}: {got:?} vs {traced:?} m"
                );
            }
        }
    }

    #[test]
    fn ground_marks_are_the_joints_pivots_not_the_actuators_anchor() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let bp = state.blueprint.as_ref().unwrap();
        let mut want: Vec<[f64; 2]> = Vec::new();
        for joint in bp.joints.values() {
            let (JointJson::Revolute { body_i, body_j, point_i, point_j, .. }
            | JointJson::Fixed { body_i, body_j, point_i, point_j, .. }) = joint
            else {
                continue;
            };
            for (body, point) in [(body_i, point_i), (body_j, point_j)] {
                if body == GROUND_ID {
                    let p = bp.bodies[GROUND_ID].attachment_points[point];
                    want.push([p[0] * 1e3, p[1] * 1e3]);
                }
            }
        }
        assert!(!want.is_empty());
        assert_eq!(data.ground.len(), want.len(), "{:?} vs {want:?}", data.ground);
        for p in &want {
            assert!(data.ground.iter().any(|g| (g[0] - p[0]).abs() < 1e-6 && (g[1] - p[1]).abs() < 1e-6), "{p:?}");
        }
        // The actuator's ground anchor (its cylinder's base) is not a mark.
        let anchor = data.frames[0].actuators[0].a;
        assert!(data.ground.iter().all(|g| (g[0] - anchor[0]).abs() > 1.0 || (g[1] - anchor[1]).abs() > 1.0));
        // The driver arc sits on the driven joint's ground pivot.
        let pivot = data.driver_pivot.expect("an angle sweep");
        assert!(data.ground.contains(&pivot));
    }

    #[test]
    fn reactions_match_the_sweeps_joint_reaction_magnitudes() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        let mut checked = 0;
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            for r in &frame.reactions {
                let Some(series) = sweep.joint_reaction_magnitudes.get(&r.id) else { continue };
                let (got, want) = ((r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt(), series[i]);
                assert!((got - want).abs() <= 1e-6 * want.max(1.0), "{} at sample {i}: {got} vs {want}", r.id);
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} reactions compared");
    }

    #[test]
    fn the_chart_is_the_actuator_force_with_lbf_in_the_readouts() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert_eq!(data.chart_label, "Actuator force (N)");
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            assert_eq!(frame.chart, at(&sweep.actuator_forces, i));
            assert_eq!(frame.actuators[0].force, frame.chart);
            let row = frame.readouts.iter().find(|r| r[0] == "Actuator force");
            assert!(row.map_or(frame.chart.is_none(), |r| r[1].contains("lbf")), "{:?}", frame.readouts);
        }
    }

    #[test]
    fn without_an_actuator_the_chart_is_the_driver_torque() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert_eq!(data.chart_label, "Driver torque (N m)");
        assert!(data.driver_pivot.is_some(), "an angle sweep has the driver arc");
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            assert_eq!(frame.chart, at(&sweep.driver_torques, i));
            assert!(frame.actuators.is_empty());
        }
    }

    #[test]
    fn a_stroke_sweep_runs_in_millimetres_with_the_linear_driver_as_the_actuator() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        let act = state
            .mechanism
            .as_ref()
            .unwrap()
            .forces()
            .iter()
            .position(|f| matches!(f, ForceElement::LinearActuator(_)))
            .unwrap();
        state.convert_actuator_to_linear_driver(act);
        assert_eq!(state.add_point_mass("rocker", 50.0, [0.0, 0.0]).as_deref(), Some("W1"));
        state.compute_sweep();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert!(sweep.sweep_mode.is_stroke() && sweep.actuator_forces.is_none());
        let data = animation_data(&state).unwrap();
        assert_eq!(
            (data.x_label.as_str(), data.x_unit.as_str(), data.chart_label.as_str()),
            ("Actuator stroke (mm)", "mm", "Actuator force (N)")
        );
        assert!(data.driver_pivot.is_none(), "no driver arc without a crank");
        let samples = solved(sweep);
        assert!(samples.len() > 10, "only {} samples solved", samples.len());
        assert_eq!(data.frames.len(), samples.len());
        for (frame, &i) in data.frames.iter().zip(&samples) {
            assert!((frame.x - sweep.angles_deg[i] * 1e3).abs() < 1e-9, "x is the stroke in mm");
            assert!(frame.driver_angle.is_none());
            assert_eq!(frame.chart, at(&sweep.driver_torques, i));
            // The linear driver draws pin to pin, as long as the stroke says.
            let [act] = frame.actuators.as_slice() else { panic!("{} actuators", frame.actuators.len()) };
            assert_eq!(act.force, frame.chart);
            let length = ((act.b[0] - act.a[0]).powi(2) + (act.b[1] - act.a[1]).powi(2)).sqrt();
            assert!((length - frame.x).abs() < 1e-2, "sample {i}: {length} mm long at stroke {} mm", frame.x);
            if frame.chart.is_some() {
                assert!(frame.readouts.iter().any(|r| r[0] == "Actuator force" && r[1].contains("lbf")));
            }
            let shares: Vec<&[String; 2]> = frame.readouts.iter().filter(|r| r[0].ends_with(" share")).collect();
            assert!(!shares.is_empty() && shares.iter().all(|r| r[1].ends_with("lbf")), "{:?}", frame.readouts);
        }
    }

    #[test]
    fn samples_without_a_solution_are_left_out() {
        let mut state = swept_lift();
        let sweep = state.sweep_data.as_mut().unwrap();
        for angles in sweep.body_angles.values_mut() {
            angles[3] = f64::NAN;
        }
        let gone = sweep.angles_deg[3] + state.driver_display_offset.to_degrees();
        let solved_count = solved(state.sweep_data.as_ref().unwrap()).len();
        let data = animation_data(&state).unwrap();
        assert_eq!(data.frames.len(), solved_count);
        assert!(data.frames.iter().all(|f| (f.x - gone).abs() > 1e-9), "sample 3 is left out");
    }

    /// The Parallelogram Press with a 40 mm wheel for its coupler's rectangle
    /// (same centre), the zone grown over the whole sweep and in contact mode.
    fn press_with_wheel() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);
        let bp = state.blueprint.as_mut().unwrap();
        let coupler = bp.bodies.get_mut("coupler").unwrap();
        let hub = coupler.geometry.as_ref().unwrap().offset;
        coupler.geometry = Some(BodyGeometry::circle(0.04, hub).unwrap());
        for force in &mut bp.forces {
            if let ForceElement::ForceZone(zone) = force {
                zone.zone_min = [-10.0, -10.0];
                zone.zone_max = [10.0, 10.0];
                zone.at_contact_point = true;
            }
        }
        state.rebuild();
        state.compute_sweep();
        state
    }

    #[test]
    fn a_wheel_s_contact_point_is_its_top_against_a_downward_force() {
        let state = press_with_wheel();
        let data = animation_data(&state).unwrap();
        assert!(!data.frames.is_empty());
        for frame in &data.frames {
            let Some(Shape::Circle { centre, r }) = frame.shapes.first() else { panic!("the wheel") };
            assert!((r - 20.0).abs() < 1e-9);
            let zone = &frame.zone_points[0];
            assert!(zone.active);
            assert!((zone.point[0] - centre[0]).abs() < 2e-3 && (zone.point[1] - (centre[1] + 20.0)).abs() < 2e-3);
        }
    }

    #[test]
    fn weights_carry_their_names_and_weight() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let g = gravity_vector(state.mechanism.as_ref().unwrap());
        let g = (g[0] * g[0] + g[1] * g[1]).sqrt();
        let frame = &data.frames[0];
        for (name, kg) in [("W1", 50.0), ("W2", 20.0)] {
            let w = frame.weights.iter().find(|w| w.name == name).unwrap_or_else(|| panic!("{name}"));
            assert!((w.newtons - kg * g).abs() < 1e-9, "{name}: {}", w.newtons);
            assert!(!w.link_self_weight, "{name} is a payload");
        }
        let bp = state.blueprint.as_ref().unwrap();
        let links: Vec<&Weight> = frame.weights.iter().filter(|w| w.link_self_weight).collect();
        let massive = bp.bodies.iter().filter(|(id, b)| id.as_str() != GROUND_ID && b.mass > 0.0).count();
        assert!(massive > 0 && links.len() == massive, "one self-weight per link with mass");
        assert!(frame.readouts.iter().any(|r| r[0] == "W1 share" && r[1].ends_with("lbf")), "{:?}", frame.readouts);
    }

    #[test]
    fn forces_read_in_kn_or_n_with_lbf_beside_them() {
        assert_eq!(force_text(21_506.0), "21.51 kN / 4835 lbf");
        assert_eq!(force_text(-21_506.0), "21.51 kN / 4835 lbf", "the size; push or pull is said apart");
        assert_eq!(force_text(1_000.0), "1.00 kN / 225 lbf");
        assert_eq!(force_text(12.0), "12.0 N / 2.7 lbf");
    }

    #[test]
    fn reactions_read_with_lbf_beside_them() {
        let state = swept_lift();
        let frame = &animation_data(&state).unwrap().frames[0];
        let rows: Vec<&[String; 2]> = frame.readouts.iter().filter(|r| r[0].starts_with("Reaction ")).collect();
        assert_eq!(rows.len(), frame.reactions.len());
        for (row, r) in rows.iter().zip(&frame.reactions) {
            assert_eq!(row[0], format!("Reaction {}", r.name));
            assert_eq!(row[1], force_text((r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt()));
        }
    }

    #[test]
    fn no_sweep_means_no_animation() {
        assert!(animation_data(&AppState::default()).is_err());
        assert!(!animation_export_available(&AppState::default()));
        let mut state = swept_lift();
        assert!(animation_export_available(&state));
        state.sweep_data = None;
        assert!(!animation_export_available(&state));
        assert_eq!(animation_data(&state).unwrap_err(), "No sweep computed");
    }

    #[test]
    fn a_trajectory_sweep_cannot_be_animated() {
        use crate::gui::state::{MotionProfile, Trajectory, TrajectoryProfile};
        use crate::gui::sweep::SweepMode;
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};
        let mut state = swept_lift();
        let profile =
            TrajectoryProfile { shape: MotionProfile::ConstantSpeed, start_value: 0.0, end_value: 1.0, duration: 2.0 };
        state.sweep_data.as_mut().unwrap().sweep_mode = SweepMode::Trajectory {
            target: ControlTarget::Angle { body_id: "crank".to_string() },
            trajectory: Trajectory::Profile(profile),
            severity: Severity::Analysis,
            n_samples: 10,
        };
        assert!(!animation_export_available(&state));
        assert_eq!(animation_data(&state).unwrap_err(), "The animation export needs an angle or stroke sweep");
    }

    #[test]
    fn the_template_has_one_data_slot() {
        assert_eq!(TEMPLATE.matches(DATA_SLOT).count(), 1);
    }

    /// The byte range of the JSON the page embeds after `const DATA = `.
    fn data_span(html: &str) -> (usize, usize) {
        let start = html.find("const DATA = ").unwrap() + "const DATA = ".len();
        (start, start + html[start..].find(";\n").unwrap())
    }

    #[test]
    fn the_page_is_self_contained_and_embeds_the_data() {
        let state = swept_lift();
        let html = generate_animation_html(&state).unwrap();
        assert!(!html.contains("https://") && !html.contains("<script src") && !html.contains("<link"));
        assert_eq!(
            html.matches("http://").count(),
            html.matches("http://www.w3.org/2000/svg").count(),
            "the only http:// is the SVG namespace"
        );
        assert!(!html.contains(DATA_SLOT));
        let (start, end) = data_span(&html);
        let v: serde_json::Value = serde_json::from_str(&html[start..end]).unwrap();
        assert_eq!(v["frames"].as_array().unwrap().len(), animation_data(&state).unwrap().frames.len());
    }

    #[test]
    fn a_name_cannot_end_the_script_or_open_a_comment_in_it() {
        let mut state = swept_lift();
        let bp = state.blueprint.as_mut().unwrap();
        let weight = bp.bodies.values_mut().flat_map(|b| b.point_masses.iter_mut()).find(|w| w.id == "W1").unwrap();
        weight.label = Some("</script><!--<script><b>".to_string());
        let html = generate_animation_html(&state).unwrap();
        // The embedded data holds no `<` at all, so the page has only the template's tags.
        let (start, end) = data_span(&html);
        assert!(!html[start..end].contains('<'), "a raw < in the data");
        // The name still reads back whole.
        let v: serde_json::Value = serde_json::from_str(&html[start..end]).unwrap();
        let names: Vec<&str> =
            v["frames"][0]["weights"].as_array().unwrap().iter().map(|w| w["name"].as_str().unwrap()).collect();
        assert!(names.iter().any(|n| n.starts_with("</script><!--<script><b>")), "{names:?}");
    }

    /// The sorted keys of a JSON object.
    fn keys(v: &serde_json::Value) -> Vec<&str> {
        let mut keys: Vec<&str> = v.as_object().expect("an object").keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }

    #[test]
    fn the_data_has_every_field_the_player_reads() {
        // animation_template.html reads exactly these names; renaming one in
        // Rust would break the page without failing any other test.
        let lift = serde_json::to_value(animation_data(&swept_lift()).unwrap()).unwrap();
        assert_eq!(
            keys(&lift),
            ["chart_label", "driver_pivot", "frames", "gravity", "ground", "n_per_lbf", "title", "x_label", "x_unit", "zones"]
        );
        let frame = &lift["frames"][0];
        assert_eq!(
            keys(frame),
            [
                "actuators", "chart", "driver_angle", "joints", "links", "reactions", "readouts", "shapes", "weights", "x",
                "zone_points"
            ]
        );
        assert_eq!(keys(&frame["links"][0]), ["closed", "name", "points"]);
        assert_eq!(keys(&frame["actuators"][0]), ["a", "b", "force"]);
        assert_eq!(keys(&frame["weights"][0]), ["link_self_weight", "name", "newtons", "point"]);
        assert_eq!(keys(&frame["reactions"][0]), ["force", "id", "name", "point"]);

        let wheel = serde_json::to_value(animation_data(&press_with_wheel()).unwrap()).unwrap();
        assert_eq!(keys(&wheel["zones"][0]), ["max", "min"]);
        let frame = &wheel["frames"][0];
        assert_eq!(keys(&frame["zone_points"][0]), ["active", "force", "point"]);
        assert_eq!(frame["shapes"][0]["kind"], "circle");
        assert_eq!(keys(&frame["shapes"][0]), ["centre", "kind", "r"]);

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);
        state.compute_sweep();
        let press = serde_json::to_value(animation_data(&state).unwrap()).unwrap();
        assert_eq!(press["frames"][0]["shapes"][0]["kind"], "polygon");
        assert_eq!(keys(&press["frames"][0]["shapes"][0]), ["kind", "points"]);
    }
}
````

In `linkage-sim-rs/src/gui/export/mod.rs`, replace:

```rust
mod csv;
```

with:

```rust
mod animation;
mod csv;
```

In the same file, replace:

```rust
pub use csv::{generate_coupler_csv_string, generate_sweep_csv_string};
```

with:

```rust
pub use animation::{animation_export_available, generate_animation_html};
pub use csv::{generate_coupler_csv_string, generate_sweep_csv_string};
```

- [ ] **Step 5: Run the tests**

Run: `cd C:/Users/Cole/source/repos/lsim-anim/linkage-sim-rs && cargo test --lib export::animation`. Expected: `19 passed; 0 failed`. A warning that `animation_export_available` and `generate_animation_html` are unused is expected until Task 6.

- [ ] **Step 6: Watch two tests fail on purpose (the red check; nothing kept)**

The module and its tests arrive together, so prove the tests can catch the two mistakes most likely to slip through:
1. In `sample_q`, change `q[idx.x_idx()] = traced[0] - (cos_t` to `q[idx.x_idx()] = traced[0] + (cos_t`. Run the Step 5 command. Expected: `a_body_whose_first_pin_is_off_its_origin_is_placed_by_its_trace` FAILED (every other sample puts the first pin at the origin, so only that test sees it). Undo the change.
2. In `generate_animation_html`, change `json.replace('<', "\\u003c")` to `json.replace('\0', "\\u003c")` (escapes nothing). Run it again. Expected: `a_name_cannot_end_the_script_or_open_a_comment_in_it` FAILED. Undo the change.

Run Step 5 once more: `19 passed`. `git status --short` shows only this task's four files and the plan.

- [ ] **Step 7: Gate and commit**

Run the gate. Expected: `GATE PASS`, linkage count 1,098. Restore `docs/chebyshev_lambda`.

```bash
cd C:/Users/Cole/source/repos/lsim-anim
git add linkage-sim-rs/src/gui/sweep/mod.rs linkage-sim-rs/src/gui/export/mod.rs linkage-sim-rs/src/gui/export/animation.rs linkage-sim-rs/src/gui/export/animation_template.html
git commit -m "feat(linkage): animated HTML export of the sweep (decisions H-1 to H-8)" -m "Co-Authored-By: <model> <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 6: The File-menu item, docs and the plan

**Files:**
- Modify: `linkage-sim-rs/src/gui/menu_bar.rs` (the item, after "Generate Report (HTML)...")
- Modify: `README.md:180`, `docs/FEATURES.md` (before `## Planned / Future`), `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`
- Commit: `docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md` (the controller's updated copy, already in the worktree)

**Interfaces:**
- Consumes: `export::animation_export_available(&AppState) -> bool`, `export::generate_animation_html(&AppState) -> Result<String, String>` (Task 5); `export::download::download_text`, `apply_download_outcome` (menu_bar.rs).

- [ ] **Step 1: The menu item**

In `linkage-sim-rs/src/gui/menu_bar.rs`, replace:

```rust
                    // ── Raster exports (gated by `raster` feature; native always
```

with:

```rust
                    if ui
                        .add_enabled(
                            export::animation_export_available(state),
                            egui::Button::new("Export Animation (HTML)..."),
                        )
                        .on_hover_text(
                            "Export a self-contained HTML page that animates the mechanism through its \
                             sweep, with its forces, weights, joint reactions and actuator force chart",
                        )
                        .clicked()
                    {
                        match export::generate_animation_html(state) {
                            Ok(html) => {
                                let outcome = export::download::download_text(
                                    "mechanism_animation.html",
                                    "text/html",
                                    &html,
                                    export::download::FileFilter {
                                        label: "HTML",
                                        extensions: &["html"],
                                    },
                                );
                                apply_download_outcome(state, outcome);
                            }
                            Err(e) => {
                                state.error_log.push(format!("Animation export failed: {}", e));
                                state.show_error_panel = true;
                            }
                        }
                        ui.close();
                    }
                    // ── Raster exports (gated by `raster` feature; native always
```

Run: `cd C:/Users/Cole/source/repos/lsim-anim/linkage-sim-rs && cargo build && cargo check --target wasm32-unknown-unknown --no-default-features --lib`. Expected: both finish; no warning names `animation.rs` or the new menu lines (the unused-import warnings of Task 5 are gone).

- [ ] **Step 2: README and features**

In `README.md`, replace:

```markdown
- **Export**: PNG, SVG, GIF (ping-pong loop), DXF, CSV, HTML report with interactive Plotly charts
```

with:

```markdown
- **Export**: PNG, SVG, GIF (ping-pong loop), DXF, CSV, HTML report with interactive Plotly charts, animated HTML page of the sweep (one self-contained file: the mechanism with its forces, weights and joint reactions, the actuator force chart, play and scrub controls; desktop and browser)
```

In `docs/FEATURES.md`, replace:

```markdown
## Planned / Future
```

with:

```markdown
### Animated HTML export

- File -> Export Animation (HTML)... saves `mechanism_animation.html`: one self-contained page (no network) that plays the mechanism through every solved sample of its angle or stroke sweep
- Links, round and rectangular shapes, joints and ground pivots; the actuator with its push or pull; each force zone's point and force; weights; joint reactions; the driver-angle arc
- Readouts per sample in the app's units with lbf beside every force (actuator force or driver torque, actuator length, mechanical advantage, weight shares, reactions); a chart of the actuator force (or driver torque) with a moving marker
- Play, bounce or loop, three speeds, a scrub slider, and switches for forces, weights and the arc; works on the desktop and in the browser

## Planned / Future
```

- [ ] **Step 3: docs/ai and the backlog**

Run `grep -n "Linkage force zones: force_zone_application" docs/ai/02-system.yaml` and add, directly above that line (same indentation, two spaces then `- "`), the entry:

```yaml
  - "Linkage animated HTML export (gui/export/animation.rs, File -> Export Animation (HTML)...): the page's data is built from the current sweep, never by solving again: each body's pose comes back from SweepData::body_angles and the coupler_traces of its first attachment point (by name), for every body of the built mechanism, a mount-point actuator's hidden cylinder and rod included (the reactions need them); links are the blueprint's bodies only. Reactions are solved per frame at sweep_time (gui::sweep, the sweep's own formula) and equal joint_reaction_magnitudes. animation_template.html holds one data slot, /*__ANIMATION_DATA__*/null; every < in the JSON is written \\u003c so no name can end the script or open a script comment. animation_export_available enables the menu item (a mechanism and an angle or stroke sweep)."
```

In `docs/ai/03-structure.yaml`, replace:

```yaml
    files: [mod, svg, dxf, csv, raster, report, schematic, firmware]
```

with:

```yaml
    files: [mod, svg, dxf, csv, raster, report, schematic, firmware, animation, animation_template.html]
```

In the same file, replace:

```yaml
      schematic: Auto-generated labeled mechanism schematic (SVG) — File → "Export labeled schematic (SVG)..."; matches docs/superpowers/specs/2026-04-29-linkage-equations-reference.md figures 2 & 3.
```

with:

```yaml
      schematic: Auto-generated labeled mechanism schematic (SVG) — File → "Export labeled schematic (SVG)..."; matches docs/superpowers/specs/2026-04-29-linkage-equations-reference.md figures 2 & 3.
      animation: File → "Export Animation (HTML)..." — AnimationData (one Frame per solved sweep sample, mm and N) embedded as JSON in animation_template.html (an SVG player, no network); generate_animation_html, animation_export_available.
```

In `docs/ai/04-memory.yaml`, append to the end of the file (run `tail -3 docs/ai/04-memory.yaml` first and match its indentation):

```yaml
  - "DONE 2026-10-06 (Part B of docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md, decisions H-1 to H-8): File -> Export Animation (HTML)..., a self-contained animated page of the sweep (gui/export/animation.rs). Poses are rebuilt from the sweep's traces, never solved again."
```

In `docs/ai/05-update-tracker.md`, replace:

```markdown
---

## 2026-10-06 — Round shapes and the contact point (Part A, branch linkage/round-geometry)
```

with:

```markdown
---

## 2026-10-06 — Animated HTML export (Part B, branch linkage/animation-export)
- File -> Export Animation (HTML)... saves one self-contained page that plays the mechanism through its sweep: links, shapes, the actuator, force-zone points, weights, joint reactions, readouts with lbf, an actuator force (or driver torque) chart, play/scrub controls. Desktop and browser (`export::download::download_text`).
- `gui/export/animation.rs` rebuilds each sample's pose from `SweepData::body_angles` and `coupler_traces` (a mount-point actuator's hidden bodies included) instead of solving again; reactions per frame at `sweep_time`, now shared with the sweep.
- `animation_template.html`: the page, one data slot; every `<` in the embedded JSON written `\u003c`.

## 2026-10-06 — Round shapes and the contact point (Part A, branch linkage/round-geometry)
```

In `docs/ai/backlog.yaml`, append to the end of the file:

```yaml

- id: BL-046
  title: "The canvas's orange zone-force label is hard to read on the yellow overlap highlight"
  dimension: gui
  risk: mechanical
  evidence: "force_render.rs draws \"F (locked)\" and \"F (contact)\" in Color32::from_rgb(255, 165, 80) over FORCE_ZONE_OVERLAP_FILL (255, 200, 0, alpha 50; colors.rs); on the live press (2026-10-06) the label at the wheel's bottom barely shows while the wheel is in the zone"
  acceptance: "the zone-force label reads clearly over the overlap highlight (a darker label colour or a backing plate); a headless test checks the label's colour against the highlight, or the label draws on a plate"
  priority: 4
  status: open
  notes: "Seen in the live check of Part A of the round-geometry plan; the animated HTML export draws its labels with a white halo instead."
```

- [ ] **Step 4: The plan**

The controller's updated plan is already in the worktree (Task 5 Step 1); it is committed in Step 5. Do not edit it.

- [ ] **Step 5: Gate and commit**

Run the gate. Expected: `GATE PASS`, linkage count 1,098. Restore `docs/chebyshev_lambda`.

```bash
cd C:/Users/Cole/source/repos/lsim-anim
git add linkage-sim-rs/src/gui/menu_bar.rs README.md docs/FEATURES.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md
git commit -m "feat(linkage): File -> Export Animation (HTML); docs; BL-046" -m "Co-Authored-By: <model> <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

After Task 6 (controller): the whole-branch review (session model) and one fix wave if needed; then the browser check: a throwaway test in a scratch copy (never committed) loads the user's press (`press_w6.json` in the session scratchpad) through `AppState::load_from_json_str`, writes `generate_animation_html` to the scratchpad, and a local `python -m http.server` serves it to Playwright: no console error but the server's favicon, 38 frames over 28 to 65 deg, the 21,506 N peak at 28 deg, scrubbing changes the readouts, Bounce off loops. Then the web bundle (`bash linkage-sim-rs/scripts/build_web.sh`, `serve_web.sh 8765`): the File menu shows the item, enabled with the press loaded, and the download arrives. Merge into local `main` and ask the user before pushing.

## Self-review record

- Spec coverage: circle shape (Task 1), contact point (Task 2), editor and canvas (Task 3), docs and checks (Task 4); the animated export (Task 5: data and page; Task 6: the menu item and docs), staged after Part A by decision H-8.
- Placeholders: none; Parts A and B give exact code and commands.
- Type consistency: `GeometryShape`, `CIRCLE_SEGMENTS`, `BodyGeometry::{circle, centre_world, outline_world, extreme_point_world}`, `ZoneAppMode`, `ForceZoneElement::{app_mode, with_app_mode}`, `force_zone_application`, `ZoneApplication { overlap, active, point, mode }`, `PendingPropertyEdit::{AddGeometry { body_id, shape }, SetGeometryShape, UpdateGeometryDiameter}` are named the same in every task.
- Review Focus: each of the five lines names its test and owning task.
- Dry run (2026-10-06, by the plan's writer): every Part A replacement block (37) applied in order to `f9df219` copies; with the scripted steps (the 9 test literals, the property panel's import) and the created test file, `cargo test --all` passed 1,058 tests, 0 failed. The two "append to the end of the file" steps (Task 1's 10 geometry tests, Task 2's 7 force-zone tests) were not part of that replay, so those tests were first compiled and run by the Task 1 and Task 2 implementers. Two of the plan's tests were corrected on the way: the marker drag takes an 8 px first step (egui reports the pointer's position when the drag starts, which must still be inside the 12 px hit radius), and the sweep comparison switches one state's zone between the two modes instead of comparing two separately loaded states (each body map iterates in its own order, so their solutions can differ by whole turns and in the last digits). The user's press with the wheel as a circle and the contact mode gives 21,506.0 N at crank 28 deg and 5,776.0 N at 65 deg, equal to the hub-locked model (worst relative difference 3.9e-14); not committed (decision R-7).
- Part B dry run (2026-10-06, by the plan's writer, in a scratch worktree from `5690cb1`): Task 5 and Task 6's code as written here compiled (native and wasm32) with no warning in the new code and no clippy finding in it; `cargo test --all` passed 1,097 tests, 0 failed (1,079 + 18). Three mutations were run against the tests: a wrong origin in the pose rebuild, the script escape removed, every reaction given the first joint's force; each made a test fail (the first only after the Strandbeest fixture was added: every other sample puts each body's first pin at its origin). Part A committed this plan with CRLF line endings (copied from the main checkout); Task 6 commits it with LF like every other plan, so its diff shows every line. The user's press (`press_w6.json`, not committed) exported 38 frames over 28 to 65 deg with the 21,506 N peak at 28 deg; Playwright showed the page with no console error but the server's favicon, and the 4-bar sample's page likewise; Bounce off loops. The first draft's faults found on the way and fixed here: no sample rebuilt for a mount-point actuator (its hidden bodies are not in the blueprint), the view fitted the actuator's far anchor, labels grew on small mechanisms, small forces read "0.00 kN", and Bounce off stuck on the last frame.
- Type consistency (Part B): `sweep_time`, `generate_animation_html`, `animation_export_available`, `animation_data`, `AnimationData`, `Frame`, `Link`, `Shape`, `Actuator`, `ZonePoint`, `Weight { link_self_weight }`, `Reaction`, `ZoneBox`; the template reads exactly the serialized field names (`frames[].links/shapes/joints/actuators/zone_points/weights/reactions/readouts`, `x`, `chart`, `driver_angle`; `title`, `x_label`, `x_unit`, `chart_label`, `n_per_lbf`, `gravity`, `ground`, `driver_pivot`, `zones`).
- Task 5 review (sonnet, approved) and the controller's rulings on its Minor findings: fixed in one round, every `<` in the embedded JSON written `\u003c` (a `<!--` then `<script` in a name could otherwise turn the page's own `</script>` into script text) and a test pinning the field names the page reads (`the_data_has_every_field_the_player_reads`; `press_with_wheel` and `data_span` shared by the tests); parked: a body with no attachment points (no joint can hold it, so no sweep solves it) and the per-frame joint sort (negligible). Task 5 then has 19 tests; linkage count 1,098. The commit commands give both trailer lines in one `-m` (Task 5's first commit has a blank line between them; left as is).
- Final whole-branch review (session model): ready "with fixes". It confirmed the physics (poses rebuilt to 3e-14 m and reactions equal to the sweep's on all 30 built-in samples; the stroke-mode sign; the share units) and found two Important issues, both fixed in one wave written and dry-run by the controller: the page ignored the mounting angle the canvas applies (it now draws every point and vector turned by it, with `driver_zero` for the arc and zones as four corners), and a coupler point named like a body's first pin misplaced the body (the export now takes the coupler point's position when it owns the trace key; `sweep::trace_key` is shared by the sweep and the export). Minor findings fixed in the same wave: labels and readouts written in Rust with the canvas's `format_magnitude` and `SHOWN_AS_ZERO_N` (no "1000.0 N", no "push" at 0 N, N beside lbf on shares, lbf to one decimal below 10 lbf), share rows by `plot_panel::source_line_name`, the item disabled while `sweep_dirty`, `weight_sources` and the joint sort hoisted out of the frame loop, reactions ordered J2 before J10, "1 sample", "-0.000" read as 0, the player drawing only on a new sample and without spreading every value into one call, and tests for the mounting angle, the display offset, angle-mode torque shares, the coupler-point clash, stroke-mode reactions, metres, duplicate payload labels, and both halves of the page's field contract. Backlogged: BL-047 (prismatic and cam joints not drawn), BL-048 (the menu's repeated export block). Task 5's module then has 26 tests; linkage count 1,105. Applying the wave, the implementer found the new stroke-sweep reaction comparison flaky (4 runs in 5): at sample 100 the stroke sweep sits next to a dead point (reactions near 1e13 N), where the sweep's solve and the export's solve of one pose differ in the fifth digit with the hash maps' run-to-run order; the comparison now skips reactions above 1 MN (these fixtures carry under a kilonewton) and passed 8 runs in 8. The scoped re-review (session model) found every finding addressed; its Minor residuals: a doc comment where the `\u003c` escape had been decoded to a bare `<` (fixed), the mounting test checking only some fields (now a test exports one sweep at 0 and 0.7 rad and requires every point and vector to be the turned copy), a negative share's sign on the newtons only (now on both numbers); parked: the template check matches a field name on any object. It also found the canvas itself draws force arrows and the zone box unturned under a mounting angle (pre-existing; BL-049). Task 5's module then has 27 tests; linkage count 1,106.
