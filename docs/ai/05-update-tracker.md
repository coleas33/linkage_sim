# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-04-10 — Modularize rendering and plot_panel

**What:** Split two of the largest GUI files:

- `src/gui/canvas/rendering.rs` (2233 lines) → `src/gui/canvas/rendering/` directory
  - `mod.rs` (1236 lines): render_mechanism, render_overlays, tooltips, grid, background image
  - `primitives.rs` (508 lines): spring/damper/arrow/arc/marker/alignment-guide drawing helpers
  - `force_render.rs` (526 lines): force element visualization + load path heat map

- `src/gui/plot_panel.rs` (1770 lines) → `src/gui/plot_panel/` directory
  - `mod.rs` (539 lines): PlotTab enum, dispatcher, shared helpers (colors, outlier filter, toggle markers, series-with-range)
  - `mechanics.rs` (324 lines): body angles, transmission, mechanical advantage, joint reactions
  - `dynamics.rs` (275 lines): driver torque, inverse dynamics, energy
  - `actuator.rs` (390 lines): force, speed, power
  - `coupler.rs` (289 lines): trace, velocity, acceleration, output force

**Why:** Both files mixed many distinct responsibilities. `rendering.rs` conflated
drawing primitives, force visualization, and mechanism render pass into one file.
`plot_panel.rs` had 13 independent plot functions that shared nothing except
helpers. The splits preserve behavior exactly (zero logic changes) but make the
files easier to navigate and reason about.

**Test results:** 644 tests pass, WASM + native builds clean, pre-existing
linear_driver doctest still fails (not related).

**Breaking changes:** None. All public APIs re-exported from the new `mod.rs`
files so external callers are unchanged.

---

## 2026-04-10 — Non-Grashof sweep support + two-pass statics

**What:**
- Sweep no longer breaks at unreachable angles. Pushes NaN across all data
  channels and continues, so non-Grashof mechanisms that only oscillate
  produce usable plots for the reachable range.
- Two-pass statics solve in sweep for actuator-driven mechanisms:
  pass 1 with force=0 → read driver torque → compute F_actuator via power
  balance → pass 2 with F_actuator injected into Q → reactions reflect the
  actuator load path.

**Why:** User's Custom 6-Bar press has a non-Grashof linkage (only oscillates)
and is actuator-driven. The previous sweep stopped at the first unreachable
angle, and joint reactions didn't show the actuator's load path.

---

## 2026-04-10 — DXF import overhaul

**What:**
- Drag-and-drop DXF import (works on web + native)
- Interactive DXF entity selection
- Conversion buttons: → Links, → Multi-joint Body, → Add Geometry to Selected
  Link, → Ground Pivots, → Linear Actuator, → Delete Selected
- Auto-create ground pivots when LinearActuator endpoint has no nearby body point
- All conversions are additive (preserve existing mechanism, push undo, rebuild)
- "Add Geometry to Selected Link" uses the Link Editor's pattern: sets body.geometry
  directly instead of creating a new body

**Why:** Imports SolidWorks sketches into the simulator with fast selection-based
workflow. Rigid geometry attachment matches the standard Link Editor behavior.

---

## 2026-04-10 — UX polish

**What:**
- `+Ground` tool snaps to unconnected link endpoints and creates revolute joint
- Joint creation handles coincident pivots correctly (excludes first-clicked point)
- Canvas joint labels show actual joint IDs (J1, J2, ...) instead of R1/R2 auto
- Driver speed editor added to input panel (RPM display, rad/s readout)
- Plot legends moved from RightTop to LeftTop to avoid blocking data
- Computed actuator force displayed on canvas when stored force=0 (sizing mode)
- Force zone creation no longer auto-creates BodyGeometry
- Sweep range editor defers recompute until drag_stopped/lost_focus to avoid
  crashes from transient inverted ranges
- `(full)` legend tooltip explains faded dashed vs solid curve when sweep range
  is enabled

**Why:** Various discoverability, correctness, and robustness fixes from user
feedback during the DXF press-mechanism workflow.

---

## Earlier history

See `docs/FEATURES.md` and `docs/history/` for the full project history including
the Python→Rust port (411→644 tests), Phase 5 GUI, sample mechanisms, export
formats, and Phase 6.8 UX overhaul.
