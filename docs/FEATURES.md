# Feature Tracker

Comprehensive list of implemented features, in-progress work, and planned improvements.

---

## Implemented

### Core Solver

- **Kinematics**: position, velocity, and acceleration of every body and coupler point across the full range of motion
- **Static forces**: required input torque and all joint reaction forces at each configuration
- **Inverse dynamics**: required actuator effort for a prescribed motion profile, including inertial loads
- **Forward dynamics**: time-domain simulation of mechanism response to applied forces
- **Crank selection analysis**: Grashof-based classification, driver ranking, and numerical range estimation

### Force Elements (12 types, Python + Rust parity)

| Force Type | Python | Rust |
|---|---|---|
| Gravity | Yes | Yes (GUI slider, 0-100g) |
| Linear springs | Yes | Yes |
| Torsion springs | Yes | Yes |
| Viscous dampers (translational) | Yes | Yes |
| Rotary dampers | Yes | Yes |
| External point forces | Yes | Yes |
| External torques | Yes | Yes |
| Gas springs | Yes | Yes |
| Bearing friction | Yes | Yes |
| Joint limits | Yes | Yes |
| Motors (force-based) | Yes | Yes |
| Linear actuators | Yes | Yes |

All force elements are editable in the GUI property panel and rendered on the canvas.

### Rust Solver Kernel

- Full solver port (Phases 1-4: kinematics, statics, inverse dynamics, forward dynamics) validated against Python golden fixtures (411 tests)
- `ForceElement` enum with 12 variants covering all Python force elements
- Expression evaluator: user-defined driver expressions (e.g., `"pi/2 * sin(3*t)"`) with GUI editor, serializable to JSON

### GUI (Phase 5 -- complete)

- 10 plot tabs: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration
- Forward dynamics simulation with timeline scrubbing, playback speed control, and constraint drift display
- PNG + SVG export (resvg-based rasterization, 1920x1080 default)
- GIF animation export (renders sweep frames via resvg, encodes with gif crate, 800x600 @ 20fps)
- CSV export (sweep data + coupler traces with comprehensive columns)
- DXF export (mechanism geometry)
- HTML report generation (plots and analysis summary)
- Diagnostics panel: Grashof classification, Jacobian conditioning, crank selection, motor sizing, torque envelopes
- 28 sample mechanisms (11 four-bar + 7 six-bar + 10 specialty), JSON save/load, undo/redo
- Animation playback with seamless 360-degree wrap (solver initial guess resets to cached angle-0 solution)
- Right-click driver reassignment on any grounded revolute joint
- Gravity slider (0-100g / 0-981 m/s^2) with real-time g-value display
- Gravity-loaded reaction force arrows at every joint; gravity direction indicator on canvas; both toggleable via View menu
- Auto-sweep: plots update automatically after mechanism changes (200ms debounce)
- Error panel: simulation failures surface in a collapsible bottom panel
- Link Editor panel: dropdown body selector with length sliders and mass/inertia controls
- Force element toolbar ribbon: categorized "Joint Torques" and "Link Forces" dropdown menus
- Professional dark theme with color-coded toolbar buttons (blue for tools, green for play, orange for forces)
- Links rendered as rectangular bars for visual clarity; dashed coupler traces
- All sidebar sections collapsible (Playback, Gravity, Driver, Simulation, Force Elements, Diagnostics)
- WebAssembly build (feature flags, WASM entry point, zero-warning compilation)

### Interactive Editor

- Create bodies, joints, and ground pivots via right-click context menu on the canvas
- Edit link lengths via logarithmic sliders in the Link Editor panel (deferred rebuild on slider release)
- Edit mass and inertia properties via logarithmic sliders with type-to-enter exact values
- Delete bodies and joints with cascading cleanup of dependent elements
- Two-click joint creation workflow with visual feedback (green ring highlight, hint text)
- Live validation warnings in status bar: DOF mismatch, disconnected bodies, missing driver
- MechanismBlueprint (MechanismJson) as editable source of truth; rebuild() on every edit
- All edit operations push to the undo/redo stack; blueprint stays in sync with snapshots
- Multi-point body creation via + Body tool
- Add Pivot Here context menu: right-click a body edge to add attachment points
- Body-aware Draw Link: snapping to body segments auto-creates pivots for branching connections
- Create Joint two-click flow: right-click attachment point -> Create Joint -> click second point
- Toolbar: [Select] [Draw Link] [+ Body] [+ Ground] | [Play/Pause] | Force dropdowns
- Closed polygon rendering for 3+ point bodies (ternary plates render as triangles)
- Prismatic + Fixed joint creation (Create Joint submenu: Revolute / Prismatic / Fixed)
- Link dimension annotations (toggleable via View > Link Dimensions)
- Point mass GUI (add/remove point masses on bodies with parallel axis theorem recomputation)
- Keyboard shortcuts help (Help > Keyboard Shortcuts dialog)
- Autosave (periodic save every 30s to sibling `.autosave.json` file)
- Autosave recovery (prompt on startup when autosave file found)
- Recent files menu (tracks last 5 opened/saved files, persisted across sessions)
- Canvas hover tooltips (body/joint info on hover without clicking)
- Mechanism summary (body count, joint count, total mass always visible in property panel)
- Ctrl+S / Ctrl+Shift+S save shortcuts (quick save / Save As)
- Ctrl+N new mechanism
- Link lengths in property panel (segment lengths shown with attachment point list)

### Phase 6.8 GUI/UX Overhaul

*Theme & Layout:*
- Professional dark theme (custom egui Visuals -- dark panels, blue accent, subtle widget fills)
- All sidebar sections collapsible with color-coded unicode icons
- Compact status bar with dim/bright contrast, Greek symbols, warning icons
- Resizable left panel

*Toolbar:*
- Color-coded tool buttons: blue editor tools, green play/pause, purple sample selector
- Unicode icons for every tool (cursor, pencil, plus, anchor, play, gear)
- Speed slider and sample mechanism dropdown integrated into toolbar
- Force toolbar ribbon: green "Joint Torques", orange "Link Forces" dropdowns

*Canvas:*
- Links rendered as rectangular bars (16px wide) for visibility
- Major/minor grid hierarchy with origin crosshair (red X / green Y, CAD convention)
- Joint glow effects: green for driver, orange for selected
- Fixed joints use X-marker (distinct from revolute circles)
- Dashed coupler traces for visual distinction
- Hover tooltips on body segments
- Zoom-adaptive grid (spacing adjusts to zoom level, down to 0.1mm)
- Alignment guides (snap guides appear during drag operations)

*Sidebar Panels:*
- Link Editor: dropdown body selector, logarithmic length/mass/inertia sliders
- Crank Angle section replaces redundant playback controls
- Gravity slider (0-100g / 0-981 m/s^2) with g-value readout
- Simulation section collapsed by default
- Force element toolbar (not sidebar) for adding forces
- Error panel for simulation failures (collapsible, status bar indicator)

### Schema v1.1.0 Features

- Body geometry: optional rectangular visual shapes on links (width, height, offset)
- Force zones: spatial regions applying distributed forces to bodies (wind, magnetic fields)
- Element labels and hover tooltips: user-defined labels on bodies, joints, and force elements
- Crank angle limits: configurable sweep range (min/max angle) for partial-rotation analysis
- Parallelogram Press sample mechanism (demonstrates body geometry, force zones, and labels)

### Post-Phase 6.8 Additions

*New features:*
- Mounting angle: per-mechanism mounting angle (radians) rotates the mechanism relative to gravity. Stored in JSON (`mounting_angle` field), UI slider in input panel, canvas rotates visually, gravity vector rotates physically
- Ground pivot editing: drag ground pivots on canvas to reposition. Property panel shows editable X/Y coordinates plus ground link distance/angle controls
- Chebyshev Lambda mechanism: rebuilt as the lambda straight-line cognate with endpoint trace
- Chebyshev Lambda + Actuator sample: custom proportions with linear actuator force element
- Linear driver constraint: `LinearDriver` type implementing the Constraint trait. Prescribes distance between two body points as d(t). Includes constant-velocity and cosine-oscillation factories
- Force element equations reference (`docs/reference/FORCE_ELEMENTS.md`)
- Parametric study user guide (`docs/guides/PARAMETRIC_STUDIES.md`)

*Bug fixes:*
- Unicode rendering: replaced all broken Unicode subscript labels and emoji codepoints with ASCII equivalents
- Scotch Yoke and Inverted Slider Crank: rebuilt with correct 3-body constraint topology
- Force zone deletion: sweep data now recomputes immediately on force add/remove/update

*Code quality:*
- DRY: `solve_and_update` -- extracted 7 copies of solve-position-then-update pattern into single helper
- DRY: `project_velocity` -- extracted duplicated velocity projection block
- DRY: `fourbar_initial_guess` -- consolidated 4 diverging versions into single canonical function
- Condition number: extracted rank-aware computation into shared `solver/condition.rs`
- Doc reorg: reorganized `docs/` into `architecture/`, `guides/`, `reference/`, `history/` subdirectories

### Plot & Analysis Improvements

- Plot zoom/pan: all plots support drag-to-pan and pinch/scroll-to-zoom
- Parametric failure feedback: orange warning when sweep evaluations fail to converge
- Thicker toggle angle markers: 2.5px red dashed lines at dead points
- Transmission angle ideal zone annotation: "Poor output zone" labels at 40/140 degrees
- Parametric sweep comparison: save named results, overlay as faded lines for side-by-side comparison
- Driver torque sign convention: labels on torque and inverse dynamics plots explaining positive/negative meaning

### Sharing & Collaboration

- Share via URL: File > Share via URL compresses mechanism JSON (deflate + base64url), copies a shareable link. Web deployment reads `?m=` URL parameter on startup
- Image trace overlay: File > Import Background Image loads a PNG/JPEG as a translucent canvas background. Adjustable opacity, scale (px/m), and X/Y offset. Native only

### Canvas Enhancements

- Load path visualization: View > Load Path color-codes links by joint reaction force magnitude (blue-cyan-green-yellow-red gradient)
- Zoom-adaptive grid: grid spacing automatically adjusts to zoom level
- Alignment guides: snap guides appear during drag operations

### Sample Gallery

- Visual sample gallery: Samples dropdown groups mechanisms by category (4-Bar, 6-Bar, Specialty) with separator headers and tooltip descriptions
- 28 total samples: 11 four-bar, 7 six-bar, 10 specialty mechanisms
- Interactive tutorial: Help > Tutorial: Build a 4-Bar

### Multi-select

- Shift+click to toggle items in a multi-selection set
- Normal click clears multi-selection and selects a single item
- Multi-selected items highlighted on canvas
- Delete and arrow-key nudge apply to all multi-selected items

### Welcome Screen

- Centered welcome panel when no mechanism is loaded
- Quick-start buttons: Load a Sample Mechanism, Start Tutorial, New Empty Mechanism, Watch Demo
- Drag & drop hint for JSON file import

### Demo Mode

- Auto-cycles through all 29 sample mechanisms with animation
- 5-second dwell time per sample
- Banner overlay: "Demo Mode -- press Escape to stop"
- Click or Escape to exit demo mode

### Actuator Sizing

- **Actuator force plot** -- required actuator force (N) vs crank angle via power balance (F = T × omega / dL_dt)
- **Statics + inverse dynamics curves** -- solid line (statics, quasi-static) and dashed line (with inertia, includes acceleration effects)
- **Actuator stroke display** -- Health Report shows stroke (max-min length), min/max actuator length, peak forces from both methods
- **Force zone overlap feedback** -- Diagnostics shows per-zone overlap percentage and applied force magnitude at current configuration
- **CSV export** includes `actuator_force_N`, `actuator_force_id_N`, `actuator_length_m` columns
- **HTML report** includes Actuator Force Envelope section with interactive Plotly chart

### Mechanism Building

- **Add Joint Point tool** -- click anywhere to add attachment points to existing bodies, creating ternary/quaternary shapes
- **Place Mass tool** -- two-phase workflow: select body, click to place. Move to Link and Reposition buttons. Preview circle at cursor.
- **Scale Mechanism** -- uniform scale with 50%/75%/150%/200% presets and custom percentage input
- **Arrow key nudge** -- select joint/body, arrow keys move by grid step, Shift+arrow = 10x
- **Angle/length constraints** -- live readout during Draw Link drag, editable angle slider per segment in property panel
- **Wider body segment hit** -- 20px radius for easier link attachment to existing bodies

### Image Overlay

- **Dedicated Image menu** in the menu bar
- **Drag-and-drop import** on web (WASM) + native file picker on desktop
- **Image Settings window** -- persistent floating panel with opacity, width (mm), shrink/grow buttons, X/Y position
- **Ctrl+drag to move** image on canvas
- **Auto-geometry for force zones** -- body geometry created automatically when force zones are added

### Additional Fixes & Polish

- **Active tool highlight** -- blue filled background with white text on selected tool button
- **Force arrows yellow** -- visible against red force zone boundaries
- **Reaction forces rounded** to nearest Newton in all displays
- **Sweep range min/max** no longer swap when editing
- **Plot double-click to reset** zoom, with hint text
- **Scrollable sample dropdown** for 29 samples
- **Floating link rejection** -- link not created if endpoint doesn't snap, with guidance message
- **Robust URL loading** -- tries multiple starting angles with continuation when zero guess fails
- **Share URL preserves crank angle** -- mechanism loads at the exact configuration the sharer was viewing
- **GIF export** plays forward then reverse for smooth ping-pong loop
- **Strandbeest (Jansen Walking)** -- 8-bar sample, 361/361 convergence, 29th sample
- **1kg mass on all sample bodies** with computed CG and Izz
- **SolidWorks import guide** (`docs/guides/SOLIDWORKS_IMPORT.md`)
- **Mechanism Health Report** -- green/yellow/red indicators for Grashof, toggles, transmission angle, peak torque, peak reactions, conditioning, convergence
- **Undo History panel** -- visual timeline with undo/redo buttons
- **Nathan Mode** -- grayscale toggle in View menu

---

## In Progress

(Nothing currently in progress)

---

## Planned / Future

### Actuator Sizing & Power

- **Required power curve** -- P = F x v at each crank angle. Shows peak power for motor/pump sizing. Critical for selecting electric motors or hydraulic power units.
- **Actuator speed plot** -- dL/dt (extension/retraction rate) at each angle. Actuators have speed limits that must not be exceeded. Already computed internally, just needs display.
- **Force margin visualization** -- User enters actuator rated force (e.g., 5000 N). Plot shows margin (rated - required) at each angle. Red zones where actuator is undersized. Simple go/no-go for actuator selection.
- **Hydraulic cylinder calculator** -- Given required force + system pressure, compute bore diameter. Given bore + pressure, overlay available force line on the actuator force plot. Most industrial actuators are hydraulic.
- **Output force at a specific point** -- "What force does my mechanism produce at THIS point in THIS direction?" Direct readout instead of indirect computation via force zones.
- **Motion profile editor** -- Replace constant-speed driver with trapezoidal/S-curve acceleration profiles. Required actuator force is much higher during acceleration phases. Standard in industrial automation.
- **Duty cycle / RMS analysis** -- For cyclic mechanisms, show RMS force over one complete cycle. Critical for electric actuator thermal sizing and fatigue life.
- **Spring counterbalance for actuators** -- "What spring parameters minimize peak actuator force?" Extend existing counterbalance assistant to work with linear actuator mechanisms.
- **Safety factor overlay** -- Color-code the force plot by ratio of required/rated: green (<50%), yellow (50-80%), red (>80%). Visual pass/fail across the full stroke.
- **Actuator datasheet overlay** -- Import a CSV of force-vs-stroke for a specific commercial actuator. Overlay on the required force plot to verify capability at every position.

### Engineering Tools

- **Measurement tool** -- on-canvas distance/angle measurement between arbitrary points
- **Auto mass from geometry/density** -- compute mass and inertia from link dimensions and material density
- **Mirror mechanism** -- reflect a mechanism across an axis to create symmetric designs
- **Tolerance analysis** -- Monte Carlo analysis with dimensional tolerances, showing coupler curve envelopes and torque spread
- **Coupler curve synthesis** -- draw desired output path, find 4-bar proportions that approximate it

### Canvas Interaction

- **Copy/paste bodies** -- duplicate selected bodies with offset placement
- **Annotation tool** -- user-placed text labels and callouts on the canvas
- **Spring/damper visualization** -- render springs as coils and dampers as dashpots instead of plain lines
- **Snap to midpoints/perpendiculars** -- additional snap targets beyond grid and alignment guides
- **Dimension constraints (parametric locks)** -- lock link lengths, angles, or distances as parametric constraints
- **Freehand sketch on canvas** -- draw freehand shapes for concept exploration

### Workflow

- **Better onboarding (first-use tooltips)** -- contextual hints that appear on first launch and dismiss after use
- **Keyboard shortcut reference improvements** -- searchable shortcut list, printable cheat sheet
- **Bill of materials export** -- table of all links, joints, forces exportable as CSV/PDF

### Analysis

- **Interference/collision detection** -- detect when links overlap or collide during motion
- **Workspace/reachability map** -- plot the full workspace envelope of a coupler point or end-effector
- **Velocity/acceleration arrows on canvas** -- visualize coupler point kinematics as animated arrows
- **Force balance indicator** -- show net force/moment balance status for static equilibrium verification
- **Animation keyframe markers** -- mark specific crank angles on the timeline for reference

### Export / Sharing

- **QR code sharing** -- encode mechanism URL as a QR code for easy sharing on mobile
- **Export to STEP/IGES (CAD import)** -- generate 3D CAD interchange files from 2D mechanism geometry
- **Include health report in HTML export** -- embed the mechanism health report in the HTML export
- **Ctrl+V paste images from clipboard** -- platform-specific clipboard image access

### Polish

- **Mechanism complexity badge** -- visual indicator of mechanism complexity (number of bodies, joints, DOF)
- **Natural language builder** -- describe a mechanism in words, auto-generate link lengths
- **Mobile support** -- responsive layout for phones/tablets (3 commits saved in git, ready to cherry-pick)
- **Mobile support** -- responsive layout and touch interactions for tablets and phones
