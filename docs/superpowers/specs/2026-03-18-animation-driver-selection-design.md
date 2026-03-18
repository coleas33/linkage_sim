# Sub-project 1: Animation Playback + Revolute Driver Reassignment

Phase 5 GUI enhancement — adds continuous animation and the ability to reassign the driven joint via right-click on the canvas.

---

## Dependencies and Ordering

**Depends on:** Phase 5 MVP (complete). All code changes are in `linkage-sim-rs/src/gui/` and `linkage-sim-rs/src/core/mechanism.rs`.

**Does not depend on:** Workstream 1 (crank selection), JSON save/load, or any unported solver modules. Uses only the existing kinematic solver API.

**Blocks:** Nothing directly. Sub-project 2 (JSON Save/Load) and Sub-project 4 (Core Editor) are independent.

---

## Non-goals

- This sub-project does not add general mechanism editing (no body/joint creation or deletion).
- Driver reassignment is scoped to **grounded revolute joints only** — not prismatic drivers, not arbitrary body pairs.
- No `DrivenRangeResult` integration (depends on Workstream 1 crank selection port). Range defaults to 0..360; limited-range mechanisms will hit solver failure at range limits and the animation will ping-pong at the last converged angle.
- No expression-based driver definitions — only constant-speed revolute drivers.
- The slider-crank sample has only one grounded revolute joint (J1), so driver reassignment is vacuously satisfied for it — there are no alternatives. This is expected and not a bug.

---

## Feature 1: Animation Playback

### Behavior

Replace the static angle slider with a combined slider + playback controls:

- **Play/Pause button** — toggles continuous animation. While playing, the driver angle auto-increments each frame.
- **Speed slider** — controls animation speed in degrees/second (default: 90 deg/s, range: 10..720 deg/s).
- **Loop/Once toggle** — default is continuous loop. In loop mode:
  - Full-rotation mechanisms: angle wraps from 360 back to 0 seamlessly.
  - At solver failure: animation reverses direction (ping-pong) from the last converged angle.
  - In "once" mode: always sweeps **forward** from the current angle to 360 (or range limit), then stops. Pressing Play again after completion does nothing — the user must drag the slider back to restart. Direction is always +1.0 in "once" mode regardless of prior ping-pong state.
- **Manual slider** — still works. Dragging the slider pauses animation and resets `animation_direction` to +1.0.
- **Direction resets:** `animation_direction` resets to +1.0 when: (a) the user switches loop/once mode, (b) `reassign_driver` is called, (c) the user drags the slider manually.

### Solver integration

- During animation, `solve_at_angle()` is called each frame with `angle += speed * dt`.
- `dt` comes from `ui.input(|i| i.stable_dt)` (egui's frame delta time — use `ui.input` since the function receives `&mut egui::Ui`, not `ctx` directly).
- On solver failure: do not advance angle. If looping, reverse direction. If once, stop.
- Cache last successful `q` as initial guess for next frame (continuation — already implemented).
- Call `ui.ctx().request_repaint()` while playing to keep the animation loop running.

### UI layout

The input panel (`input_panel.rs`) expands to include:
```
[▶ Play] [Speed: ====90°/s====] [🔁 Loop ▼]
[Crank angle: ========120.0°========        ]
● Converged in 4 iters (‖Φ‖ = 3.2e-12)
```

The play button shows ▶ when paused, ⏸ when playing (text, not emoji — egui renders text reliably). Speed slider is compact. Loop/Once is a small dropdown or toggle button.

---

## Feature 2: Revolute Driver Reassignment

### Scope

Allows the user to change which grounded revolute joint is the driven input. Scoped to:
- **Grounded revolute joints only** — joints where one body is `"ground"`.
- **Constant-speed driver** — omega and theta_0 are preserved from the current driver.
- **Mechanism rebuild required** — changing the driver requires rebuilding the mechanism (the driver is a constraint equation, not a force element).

### Interaction model

**Right-click a revolute joint on the canvas → context menu:**

```
Joint: J1 (Revolute)
──────────────────
  ✓ Set as Driver    ← only for grounded revolute joints
    Properties...    ← future: opens property panel (stub for now)
```

- "Set as Driver" is **only shown** for revolute joints where `body_i_id == "ground" || body_j_id == "ground"`.
- If the joint is already the driver, the menu item is grayed/checked.
- Clicking "Set as Driver" triggers a mechanism rebuild with the new driver joint.

### Joint-to-body-pair mapping

The driver API is body-pair-based (`add_constant_speed_driver(id, body_i, body_j, omega, theta_0)`), not joint-based. When the user right-clicks joint J4 (which connects "rocker" and "ground"), the code must:

1. Read `joint.body_i_id()` and `joint.body_j_id()` from the `Constraint` trait.
2. Determine which body is ground. The non-ground body becomes the driven body.
3. Call `add_constant_speed_driver("D1", "ground", driven_body_id, omega, theta_0)`.

This mapping is always unambiguous for grounded revolute joints: exactly one body is ground, and the other is the driven body. The `build_sample_with_driver` function encodes this logic.

### Mechanism rebuild flow

When the user selects a new driver joint:

1. Identify the current sample mechanism type (from `AppState::current_sample`).
2. Rebuild the mechanism from scratch using the sample builder, but with the driver attached to the selected joint's body pair instead of the default.
3. This requires a new builder API: `build_sample_with_driver(sample, driver_joint_id) -> (Mechanism, DVector<f64>)`.
4. The builder constructs bodies and joints without a driver, then uses the joint-to-body-pair mapping to attach a constant-speed driver to the correct body pair.
5. Solve at angle 0 to get initial configuration.
6. Reset animation state (pause, angle = 0, direction = +1.0).

**Future note:** This rebuild-from-sample approach only works while the GUI operates on hardcoded samples. When the editor (Sub-project 4) arrives, a `Mechanism::replace_driver()` or unbuild/rebuild API will be needed for arbitrary user-created mechanisms.

### API changes required

**`Mechanism` needs a method to identify grounded revolute joints:**

```rust
impl Mechanism {
    /// Return IDs of revolute joints where one body is ground.
    pub fn grounded_revolute_joint_ids(&self) -> Vec<String>

    /// Return the body pair (body_i, body_j) of the first driver, if any.
    /// Used by the GUI to check if a joint is already the driven joint.
    pub fn driver_body_pair(&self) -> Option<(&str, &str)>
}
```

**`samples.rs` needs a flexible builder:**

```rust
/// Build a sample mechanism with the driver on a specific joint.
/// If driver_joint_id is None, uses the default driver joint.
pub fn build_sample_with_driver(
    sample: SampleMechanism,
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String>
```

This builder constructs the mechanism without a driver, then attaches the constant-speed driver to the specified joint. The joint must be a grounded revolute joint. Returns an error if the joint is not valid for driving.

**`AppState` gets a `reassign_driver` method:**

```rust
impl AppState {
    /// Rebuild the current mechanism with a different driver joint.
    pub fn reassign_driver(&mut self, joint_id: &str)
}
```

This calls `build_sample_with_driver`, replaces the mechanism, resets the angle and animation state, and solves at t=0.

### Canvas context menu

**`canvas.rs` adds right-click handling using a pending-action pattern:**

The canvas borrows `state.mechanism` immutably during the rendering pass. The context menu closure cannot call `state.reassign_driver()` directly because that requires `&mut self`. Instead:

1. During the immutable rendering pass, collect a `right_clicked_joint: Option<(String, bool)>` — the joint ID and whether it is a grounded revolute — using the same hit-test logic as left-click selection.
2. After the rendering scope ends (dropping the immutable borrow), check if a context menu action is pending.
3. Use `response.context_menu(|ui| { ... })` on the canvas response. Inside the closure, use the collected joint info to render the menu items. Set a `pending_driver_reassignment: Option<String>` field on AppState if the user clicks "Set as Driver".
4. After the context menu closure, check `pending_driver_reassignment` and call `state.reassign_driver()` if set.

This matches the existing canvas pattern where hit-test data is collected during the immutable pass and mutations happen afterward.

---

## State changes to AppState

Add these fields:

```rust
pub struct AppState {
    // ... existing fields ...

    // Animation state
    pub playing: bool,
    pub animation_speed_deg_per_sec: f64,  // default: 90.0
    pub loop_mode: bool,                    // true = continuous loop, false = once
    pub animation_direction: f64,           // +1.0 or -1.0 (for ping-pong)

    // Driver state
    pub driver_joint_id: Option<String>,    // cache: which joint is currently driven
    // Note: driver_joint_id is a cache of information derivable from
    // Mechanism::driver_body_pair(). It exists because matching a joint's
    // body pair against the driver's body pair on every frame is wasteful.
    // Reset on load_sample() and reassign_driver().

    // Pending action (for context menu borrow pattern)
    pub pending_driver_reassignment: Option<String>,
}
```

---

## Definition of Done

- [ ] Play/Pause button toggles continuous animation
- [ ] Speed slider controls animation rate (10..720 deg/s)
- [ ] Loop/Once toggle: loop wraps or ping-pongs, once sweeps and stops
- [ ] Manual slider pauses animation on drag
- [ ] Right-click grounded revolute joint → "Set as Driver" context menu
- [ ] "Set as Driver" rebuilds mechanism with new driver, resets angle
- [ ] Non-grounded and non-revolute joints do not show "Set as Driver"
- [ ] Already-driven joint shows the option as grayed/checked
- [ ] Animation and driver selection work for both sample mechanisms (4-bar has 2 grounded revolute candidates; slider-crank has 1 — no reassignment possible)
- [ ] All existing tests pass + new tests for:
  - `grounded_revolute_joint_ids()` — returns correct IDs, empty Vec for mechanisms with none
  - `driver_body_pair()` — returns correct body pair
  - `build_sample_with_driver()` — builds with non-default driver, errors on invalid joint
  - Animation frame stepping logic (angle increment, wrap, ping-pong direction reversal)
