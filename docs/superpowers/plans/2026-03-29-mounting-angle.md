# Mounting Angle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-mechanism mounting angle that rotates the mechanism relative to gravity, stored in JSON, with visual canvas rotation and a UI slider.

**Architecture:** A single `mounting_angle: f64` (radians) stored in `MechanismJson` and mirrored in `AppState`. During rebuild, the gravity vector is rotated by the mounting angle: `g = [-g·sin(θ), -g·cos(θ)]`. The canvas applies a rotation transform around the mechanism centroid so the linkage appears tilted while gravity arrows point straight down.

**Tech Stack:** Rust, egui, existing `Mechanism` / `ForceElement` / `AppState` APIs

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/io/schema.rs` | Add `mounting_angle` field to `MechanismJson` |
| Modify | `src/gui/state/mod.rs` | Add `mounting_angle` field to `AppState`, default 0.0 |
| Modify | `src/gui/state/blueprint_ops.rs` | Use mounting angle when injecting gravity force |
| Modify | `src/gui/input_panel.rs` | Add mounting angle slider next to gravity slider |
| Modify | `src/gui/canvas/rendering.rs` | Apply rotation transform to canvas drawing |
| Modify | `src/gui/state/file_io.rs` | Sync mounting_angle between AppState and blueprint |

---

### Task 1: Add mounting_angle to JSON schema and AppState

**Files:**
- Modify: `src/io/schema.rs:41-62` (MechanismJson struct)
- Modify: `src/gui/state/mod.rs:~130` (AppState fields)
- Modify: `src/gui/state/mod.rs:~333` (Default impl)

- [ ] **Step 1: Add field to MechanismJson**

In `schema.rs`, add to `MechanismJson`:

```rust
/// Mechanism mounting angle in radians. Rotates the mechanism relative to
/// gravity (0 = horizontal, positive = counterclockwise). Backward-compatible.
#[serde(default, skip_serializing_if = "is_zero")]
pub mounting_angle: f64,
```

Add helper at module level:
```rust
fn is_zero(v: &f64) -> bool { *v == 0.0 }
```

- [ ] **Step 2: Add field to AppState**

In `state/mod.rs`, add alongside `gravity_magnitude`:
```rust
/// Mechanism mounting angle in radians (0 = horizontal).
pub mounting_angle: f64,
```

Default to `0.0` in the Default impl.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -5`
Expected: clean (new field initialized in Default, serde default for JSON)

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/io/schema.rs linkage-sim-rs/src/gui/state/mod.rs
git commit -m "feat: add mounting_angle field to schema and AppState"
```

---

### Task 2: Rotate gravity vector using mounting angle

**Files:**
- Modify: `src/gui/state/blueprint_ops.rs:~579-583` (gravity sync during rebuild)

- [ ] **Step 1: Write failing test**

In `src/gui/state/mod.rs` tests section, add:

```rust
#[test]
fn mounting_angle_rotates_gravity_vector() {
    use crate::gui::samples::{build_sample, SampleMechanism};

    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    state.gravity_magnitude = 9.81;
    state.mounting_angle = std::f64::consts::FRAC_PI_2; // 90°
    state.rebuild();

    // With 90° mount, gravity should point in -x direction: (-9.81, 0)
    let mech = state.mechanism.as_ref().unwrap();
    let gravity_forces: Vec<_> = mech.forces().iter()
        .filter_map(|f| match f {
            crate::forces::elements::ForceElement::Gravity(g) => Some(g),
            _ => None,
        })
        .collect();

    assert_eq!(gravity_forces.len(), 1);
    let g = gravity_forces[0];
    assert!((g.g_vector[0] - (-9.81)).abs() < 1e-6, "g_x should be -9.81, got {}", g.g_vector[0]);
    assert!(g.g_vector[1].abs() < 1e-6, "g_y should be ~0, got {}", g.g_vector[1]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib mounting_angle_rotates`
Expected: FAIL (gravity still uses `[0, -9.81]`)

- [ ] **Step 3: Modify gravity injection in rebuild**

In `blueprint_ops.rs`, change the gravity sync block (~line 579):

```rust
if self.gravity_magnitude > 0.0 {
    let g = self.gravity_magnitude;
    let theta = self.mounting_angle;
    mech.add_force(ForceElement::Gravity(GravityElement {
        g_vector: [-g * theta.sin(), -g * theta.cos()],
    }));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib mounting_angle_rotates`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/state/
git commit -m "feat: rotate gravity vector by mounting angle during rebuild"
```

---

### Task 3: Add mounting angle UI slider

**Files:**
- Modify: `src/gui/input_panel.rs:~100-119` (after gravity slider)

- [ ] **Step 1: Add slider after gravity section**

In `input_panel.rs`, after the gravity collapsing header (after line ~119), add:

```rust
// ── Mounting Angle ───────────────────────────────────────────
egui::CollapsingHeader::new(
    egui::RichText::new("\u{2922} Mounting Angle").color(gravity_color),
)
    .id_salt("mounting_angle_section")
    .default_open(false)
    .show(ui, |ui| {
        let prev = state.mounting_angle;
        let mut angle_deg = state.mounting_angle.to_degrees();
        ui.add(
            egui::Slider::new(&mut angle_deg, -180.0..=180.0)
                .suffix("\u{00b0}")
                .step_by(0.5)
                .clamping(egui::SliderClamping::Always),
        ).on_hover_text("Mechanism mounting angle relative to horizontal. Rotates mechanism and gravity direction.");
        state.mounting_angle = angle_deg.to_radians();
        if (state.mounting_angle - prev).abs() > 1e-9 {
            state.mark_sweep_dirty();
            state.rebuild();
        }
    });
```

- [ ] **Step 2: Verify compilation and manual check**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -5`
Expected: clean compile

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/input_panel.rs
git commit -m "feat: add mounting angle slider to input panel"
```

---

### Task 4: Rotate canvas rendering

**Files:**
- Modify: `src/gui/canvas/rendering.rs:~22-29` (render_mechanism entry)
- Modify: `src/gui/canvas/mod.rs` (ViewTransform or draw_canvas)

- [ ] **Step 1: Apply rotation in render_mechanism**

The approach: modify `ViewTransform::world_to_screen` to apply the mounting angle rotation before the existing pan/zoom transform. This way ALL world→screen conversions rotate automatically — bodies, joints, forces, labels, everything.

In `src/gui/state/mod.rs`, find `ViewTransform` and its `world_to_screen` method. Add rotation:

```rust
pub fn world_to_screen(&self, wx: f64, wy: f64, mounting_angle: f64) -> [f32; 2] {
    // Rotate world coords by mounting angle around mechanism center
    let (sin_a, cos_a) = mounting_angle.sin_cos();
    let rx = cos_a * wx - sin_a * wy;
    let ry = sin_a * wx + cos_a * wy;
    // Then apply existing pan/zoom
    // ...existing transform using rx, ry instead of wx, wy...
}
```

**Note:** This will require updating ALL call sites of `world_to_screen` to pass the mounting angle. An alternative is to store mounting_angle on ViewTransform itself. Check what's simpler — if ViewTransform already has access to AppState fields, store it there.

Evaluate both approaches when reading the code:
1. Add `mounting_angle` field to `ViewTransform`, set it during canvas setup
2. Pass `mounting_angle` as parameter to world_to_screen

Choose whichever requires fewer call-site changes. Similarly update `screen_to_world` for the inverse.

- [ ] **Step 2: Run full test suite**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/
git commit -m "feat: rotate canvas rendering by mounting angle"
```

---

### Task 5: Persist mounting angle in JSON file I/O

**Files:**
- Modify: `src/gui/state/file_io.rs` (load/save logic)
- Modify: `src/gui/state/blueprint_ops.rs` (blueprint sync)

- [ ] **Step 1: Write round-trip serialization test**

```rust
#[test]
fn mounting_angle_round_trips_through_json() {
    use crate::gui::samples::{build_sample, SampleMechanism};
    use crate::io::serialization::{save_mechanism, load_mechanism_unbuilt};

    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    state.mounting_angle = 0.5; // ~28.6°

    // Save — mounting_angle should be in the blueprint
    let bp = state.blueprint.as_ref().unwrap();
    assert!((bp.mounting_angle - 0.5).abs() < 1e-10);

    let json = serde_json::to_string(bp).unwrap();
    assert!(json.contains("mounting_angle"));

    let reloaded: crate::io::schema::MechanismJson = serde_json::from_str(&json).unwrap();
    assert!((reloaded.mounting_angle - 0.5).abs() < 1e-10);
}
```

- [ ] **Step 2: Sync mounting_angle into blueprint on change**

In `blueprint_ops.rs`, find where other AppState fields are synced to the blueprint (likely in `rebuild()` or a sync helper). Add:

```rust
if let Some(ref mut bp) = self.blueprint {
    bp.mounting_angle = self.mounting_angle;
}
```

Also in file_io.rs load path, sync blueprint → AppState:
```rust
self.mounting_angle = bp.mounting_angle;
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib mounting_angle`
Expected: all mounting angle tests pass

- [ ] **Step 4: Run full test suite**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/state/ linkage-sim-rs/src/io/
git commit -m "feat: persist mounting angle in JSON schema"
```

---

### Task 6: Update SYSTEM.md documentation

- [ ] **Step 1: Update SYSTEM.md**

Add mounting angle to the relevant sections describing mechanism properties and force elements.

- [ ] **Step 2: Commit**

```bash
git add docs/SYSTEM.md
git commit -m "docs: document mounting angle feature"
```
