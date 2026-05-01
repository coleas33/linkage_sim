# Mobile-Friendly Viewing Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the web (WASM) version usable on mobile phones for viewing and demonstrating mechanisms — load samples, animate, scrub crank angle, pinch-to-zoom.

**Architecture:** Detect small screens at startup via egui's available screen size. When width < 768px, switch to a mobile layout: collapse all side panels, increase touch targets, add pinch-to-zoom, show a floating control bar. No code-path changes to the solver or mechanism — purely GUI/layout.

**Tech Stack:** Rust, egui, eframe (WASM), HTML/CSS viewport meta

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `web/index.html` | Add viewport meta tag |
| Create | `src/gui/mobile.rs` | Mobile detection + style helpers |
| Modify | `src/gui/mod.rs` | Conditional panel layout, mobile module, floating controls |
| Modify | `src/gui/canvas/interaction.rs` | Pinch-to-zoom support |
| Modify | `src/gui/canvas/colors.rs` | Scale hit radii for mobile |

---

### Task 1: Viewport meta tag

**Files:**
- Modify: `web/index.html`

- [ ] **Step 1: Add viewport meta tag**

In `web/index.html`, inside `<head>`, add:
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

The `user-scalable=no` prevents the browser's default pinch-zoom (which zooms the whole page). We'll handle pinch-zoom ourselves on the canvas.

- [ ] **Step 2: Commit**

```bash
git add linkage-sim-rs/web/index.html
git commit -m "feat: add viewport meta tag for mobile rendering"
```

---

### Task 2: Mobile detection and style helpers

**Files:**
- Create: `src/gui/mobile.rs`
- Modify: `src/gui/mod.rs` (add `mod mobile;`)

- [ ] **Step 1: Create mobile.rs**

```rust
//! Mobile layout detection and style adjustments.

use eframe::egui;

/// Returns true if the screen is small enough to use mobile layout.
/// Called once per frame with the current screen rect.
pub fn is_mobile(ctx: &egui::Context) -> bool {
    let screen = ctx.screen_rect();
    screen.width() < 768.0
}

/// Apply mobile-friendly style overrides: larger buttons, more padding,
/// bigger fonts. Call once at the start of each frame when mobile is detected.
pub fn apply_mobile_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // Larger touch targets (Apple HIG: minimum 44x44px)
    style.spacing.button_padding = egui::vec2(16.0, 12.0);
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.interact_size.y = 44.0; // minimum interactive height

    // Larger slider grab
    style.spacing.slider_width = 200.0;

    // Slightly larger text
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::proportional(16.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(16.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::proportional(13.0),
    );

    ctx.set_style(style);
}

/// Restore default desktop style. Call when switching back to desktop.
pub fn apply_desktop_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.interact_size.y = 18.0;
    style.spacing.slider_width = 100.0;

    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::proportional(14.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(14.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::proportional(10.0),
    );

    ctx.set_style(style);
}
```

- [ ] **Step 2: Register module in mod.rs**

Add `pub mod mobile;` near the other module declarations in `src/gui/mod.rs`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p linkage-sim-rs`

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/mobile.rs linkage-sim-rs/src/gui/mod.rs
git commit -m "feat: add mobile detection and style helpers"
```

---

### Task 3: Conditional panel layout for mobile

**Files:**
- Modify: `src/gui/mod.rs` (the `update()` method, panel layout section ~lines 836-886)

- [ ] **Step 1: Add mobile state to LinkageApp**

Add a field to `LinkageApp`:
```rust
is_mobile: bool,
```
Initialize to `false` in `new()`.

- [ ] **Step 2: Detect mobile at start of each frame**

At the top of `update()`, before any panel drawing:
```rust
let was_mobile = self.is_mobile;
self.is_mobile = mobile::is_mobile(ctx);
if self.is_mobile != was_mobile {
    if self.is_mobile {
        mobile::apply_mobile_style(ctx);
    } else {
        mobile::apply_desktop_style(ctx);
    }
}
```

- [ ] **Step 3: Wrap panel layout in mobile check**

For the left panel (~line 836):
```rust
if !self.is_mobile {
    egui::SidePanel::left("left_panel")
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| { ... });
}
```

Same for the right parametric panel, bottom plot panel, and the toolbar ribbon. On mobile, these are all hidden by default.

Keep the menu bar (top) but simplify it: only show the Samples dropdown and Play/Pause.

- [ ] **Step 4: Add floating mobile controls**

When `self.is_mobile`, draw a floating bottom bar using `egui::Area`:
```rust
if self.is_mobile {
    egui::Area::new(egui::Id::new("mobile_controls"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -10.0))
        .show(ctx, |ui| {
            egui::Frame::popup(&ctx.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Play/Pause button
                    let play_label = if self.state.playing { "Pause" } else { "Play" };
                    if ui.button(play_label).clicked() {
                        self.state.playing = !self.state.playing;
                    }

                    // Crank angle slider (compact)
                    let mut angle_deg = self.state.driver_angle.to_degrees();
                    if ui.add(
                        egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                            .show_value(false)
                    ).changed() {
                        self.state.solve_at_angle(angle_deg.to_radians());
                    }

                    // Sample picker button
                    ui.menu_button("Samples", |ui| {
                        for sample in crate::gui::samples::SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close();
                            }
                        }
                    });
                });
            });
        });
}
```

- [ ] **Step 5: Verify compilation and test on desktop**

Run: `cargo check -p linkage-sim-rs`

On desktop (window > 768px), behavior should be unchanged. Resize window below 768px to see mobile layout.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/mod.rs
git commit -m "feat: mobile layout with collapsed panels and floating controls"
```

---

### Task 4: Pinch-to-zoom on canvas

**Files:**
- Modify: `src/gui/canvas/interaction.rs`

- [ ] **Step 1: Add pinch-to-zoom support**

In `handle_interaction()`, find the scroll-wheel zoom handler (search for `scroll_delta` or `zoom`). Add pinch-to-zoom using egui's built-in `zoom_delta`:

```rust
// Pinch-to-zoom (touch devices) and scroll-wheel zoom
let zoom_delta = ui.input(|i| i.zoom_delta());
if zoom_delta != 1.0 {
    // zoom_delta > 1.0 means zoom in, < 1.0 means zoom out
    let zoom_center = response.hover_pos().unwrap_or(canvas_rect.center());
    let [wx, wy] = state.view.screen_to_world(zoom_center.x, zoom_center.y);
    state.view.scale *= zoom_delta;
    state.view.scale = state.view.scale.clamp(1.0, 100000.0);
    // Adjust offset so zoom centers on the pointer
    let [new_sx, new_sy] = state.view.world_to_screen(wx, wy);
    state.view.offset_x += zoom_center.x - new_sx;
    state.view.offset_y += zoom_center.y - new_sy;
}
```

Check if there's already a scroll-wheel zoom handler — if so, add the `zoom_delta` check alongside it (or replace if scroll is already handled via zoom_delta).

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p linkage-sim-rs`

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas/interaction.rs
git commit -m "feat: pinch-to-zoom support for touch devices"
```

---

### Task 5: Scale canvas hit radii for mobile

**Files:**
- Modify: `src/gui/canvas/colors.rs`
- Modify: `src/gui/canvas/mod.rs` or `rendering.rs`

- [ ] **Step 1: Make hit radii scale-aware**

Currently in `colors.rs`:
```rust
pub const HIT_RADIUS: f32 = 12.0;
pub const JOINT_RADIUS: f32 = 7.0;
```

Add mobile-scaled versions. The simplest approach: pass a `mobile: bool` flag to the canvas draw functions and use larger radii:

```rust
pub const HIT_RADIUS: f32 = 12.0;
pub const HIT_RADIUS_MOBILE: f32 = 24.0;
pub const JOINT_RADIUS: f32 = 7.0;
pub const JOINT_RADIUS_MOBILE: f32 = 12.0;
```

In `draw_canvas()`, select the appropriate radius based on `state.is_mobile` (add an `is_mobile: bool` field to AppState, set from the LinkageApp each frame).

Alternatively, keep it simpler: add `pub is_mobile: bool` to AppState, and in the canvas code, compute `let hit_radius = if state.is_mobile { 24.0 } else { 12.0 };`

- [ ] **Step 2: Add is_mobile to AppState**

In `src/gui/state/mod.rs`, add:
```rust
pub is_mobile: bool,
```
Default to `false`.

In `src/gui/mod.rs` update(), sync it:
```rust
self.state.is_mobile = self.is_mobile;
```

- [ ] **Step 3: Use scaled hit radius in canvas interaction**

In `interaction.rs`, replace `HIT_RADIUS` usages with:
```rust
let hit_radius = if state.is_mobile { HIT_RADIUS_MOBILE } else { HIT_RADIUS };
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p linkage-sim-rs`

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas/ linkage-sim-rs/src/gui/state/mod.rs linkage-sim-rs/src/gui/mod.rs
git commit -m "feat: larger touch targets on mobile for canvas interaction"
```

---

### Task 6: Auto-load sample on mobile

**Files:**
- Modify: `src/gui/mod.rs`

- [ ] **Step 1: On mobile first load, show sample picker or auto-load FourBar**

In `update()`, after the mobile detection, if the mechanism is None and we're on mobile, auto-load a sample:

```rust
if self.is_mobile && self.state.mechanism.is_none() && self.state.current_sample.is_none() {
    self.state.load_sample(samples::SampleMechanism::FourBar);
    self.state.playing = true; // auto-play animation
}
```

This gives mobile users an immediate interactive experience instead of a blank canvas.

- [ ] **Step 2: Commit**

```bash
git add linkage-sim-rs/src/gui/mod.rs
git commit -m "feat: auto-load sample and play on mobile first visit"
```
