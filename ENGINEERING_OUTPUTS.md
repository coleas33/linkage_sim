# Engineering Outputs

What the tool actually produces for the engineer. Every output is defined precisely: what it is, what coordinates it uses, and what caveats apply.

---

## Joint Reaction Forces

Joint reactions are the primary structural output. They size bearings, pins, and supports.

### Output Formats

Every joint reaction is available in all of these formats simultaneously:

| Format | Definition | Use case |
|---|---|---|
| **Global Fx, Fy** | Lagrange multiplier components, directly from solver | General-purpose, for combining with external loads |
| **Resultant magnitude** | `F_res = √(Fx² + Fy²)` | Bearing catalog selection (static load rating) |
| **Direction angle** | `α = atan2(Fy, Fx)` | Load direction analysis |
| **Body-local components** | `F_local = Aᵢᵀ * [Fx, Fy]` — reaction in body `i`'s frame | Force analysis relative to link geometry |
| **Radial / tangential** | Radial = along line from body CG to joint. Tangential = perpendicular | Bearing load decomposition, radial vs. thrust loading |

For **fixed joints**, the reaction also includes a torque component `T` (N·m), reported alongside the force components.

For **prismatic joints**, the reaction includes: perpendicular force (N) and anti-rotation torque (N·m).

### Driver Reaction

The driven joint's Lagrange multiplier is reported separately as:

- **Required input torque** (N·m) for revolute drivers
- **Required input force** (N) for prismatic drivers

This is the primary actuator sizing output.

### Indeterminate Reactions Warning

If the mechanism has redundant constraints (detected by `constraint_rank < n_constraint_equations`), passive joint reactions are not unique. The solver reports the minimum-norm solution and warns: "Reactions in redundant constraint directions are indeterminate. Reported values are one valid solution (minimum norm). Actual load distribution depends on structural compliance, which this rigid-body model does not capture."

---

## Result Envelopes

For bearing sizing, actuator sizing, and fatigue estimation, raw force-vs-angle traces are not enough. The tool computes summary statistics over a full cycle or time range:

| Envelope | Definition | Use case |
|---|---|---|
| **Peak load** | `max(F_res)` over the sweep/time range | Bearing static load rating, structural proof load |
| **RMS load** | `√(mean(F_res²))` over the sweep/time range | Bearing equivalent dynamic load (approximate) |
| **Min / Max components** | `min(Fx)`, `max(Fx)`, `min(Fy)`, `max(Fy)` | Load range for fatigue (ΔF = max - min) |
| **Peak input torque** | `max(|T_driver|)` | Motor peak torque requirement |
| **RMS input torque** | `√(mean(T_driver²))` | Motor continuous torque rating |
| **Peak input speed** | `max(|ω_driver|)` | Motor speed rating |
| **Mean input power** | `mean(T_driver * ω_driver)` | Motor thermal sizing |
| **Worst-case transmission angle** | `min(μ)` or `max(|μ - 90°|)` over the sweep | Linkage geometry quality metric |

Envelopes are computed per joint, per force element, and for the driver. They are available as a summary table alongside the full traces.

---

## Mechanical Advantage

Mechanical advantage requires the user to define:

- **Input coordinate**: which driven joint (and its generalized coordinate)
- **Output coordinate**: a specific point on a specific body, and a direction of interest

The instantaneous mechanical advantage is then:

```
MA = ∂(output displacement) / ∂(input displacement)
```

For force transmission:
```
Force MA = output force capacity / input effort = 1 / (velocity ratio)
```

This is computed from the Jacobian. For a revolute input driving a point moving in direction `û`:

```
velocity ratio = v_output · û / ω_input
MA_force = ω_input / (v_output · û)
```

**Caveat:** mechanical advantage is meaningful only when the input and output coordinates are defined. For a generic mechanism with no obvious "output," the user must specify what to measure.

---

## Transmission Angle

The transmission angle `μ` measures how effectively force is transmitted through the mechanism. It is defined for a **specific pair of connected bodies** at a specific joint.

**Classical 4-bar definition:** the angle between the coupler link and the output (follower) link at their common joint. Ideal: 90°. Acceptable range: ~40°–140°. Approaching 0° or 180° means poor force transmission or lockup.

**General mechanism:** the user selects which body pair and joint to evaluate. The tool computes the angle between the two bodies at that joint and reports it vs. input position.

**Transmission angle is not universally defined for arbitrary mechanisms.** It is most meaningful for specific transmission paths through the mechanism. The tool does not automatically determine which body pairs matter — the user chooses based on engineering judgment.

### Pressure Angle

Pressure angle is related to transmission angle but defined at a joint relative to a specific force transmission direction. For a simple revolute joint connecting a driver body to a driven body:

```
pressure angle = 90° - transmission angle
```

But this simple relationship holds only when the force direction aligns with the link-to-link geometry. For general cases (off-axis loads, prismatic joints, complex topology), pressure angle must be computed from the actual force direction at the joint relative to the allowed motion direction.

The tool computes pressure angle when the user specifies the force direction of interest at a joint.

---

## Toggle Positions and Dead Points

**Toggle position:** a configuration where the mechanism's Jacobian becomes singular (or nearly so). The transmission angle passes through 0° or 180°. The mechanism locks or loses controllability.

**Dead point:** a specific type of toggle where the input cannot drive the output. The input link can rotate but the output link is momentarily stationary (velocity ratio → 0 or ∞).

**Detection:** the solver monitors:
- Condition number of `Φ_q` — spikes indicate proximity to singularity
- Singular values of `Φ_q` — the smallest singular value approaching zero indicates a toggle
- Transmission angle (for user-selected body pairs) — crossing 0° or 180°

Toggle positions are reported as specific input angles/displacements, with the associated body pair and joint identified.

---

## Coupler Curves

The full path of any named coupler point on any body, traced over the input sweep. Output as:

- `(x, y)` coordinates in global frame vs. input angle
- Parametric curve data suitable for plotting
- Velocity magnitude and direction at each point
- Acceleration magnitude and direction at each point

Coupler curves are the primary output for path-generation analysis: verifying that a mechanism guides a tool, end-effector, or workpiece along the desired trajectory.

---

## Load-Path Decomposition

For engineering decision-making, knowing the total reaction at a joint is often not enough. The engineer needs to know *what is driving the load* — is it gravity, spring preload, inertia, or the external payload? This decomposition turns the simulator from a "solver" into an "engineering tool."

### Source Decomposition

At each configuration, the generalized force vector `Q` is assembled from individual force element contributions. The Lagrange multipliers (reactions) can be decomposed by source by solving the equilibrium equation separately for each contribution:

```
Φ_q^T * λ_total = -(Q_gravity + Q_springs + Q_dampers + Q_external + Q_inertia)
```

Decompose into:
```
Φ_q^T * λ_gravity  = -Q_gravity
Φ_q^T * λ_springs  = -Q_springs
Φ_q^T * λ_dampers  = -Q_dampers
Φ_q^T * λ_external = -Q_external
Φ_q^T * λ_inertia  = -Q_inertia     (inverse dynamics only: Q_inertia = -M * q̈)
```

Each `λ_source` gives the joint reactions and driver effort attributable to that source alone. By linearity of the equilibrium equation, `λ_total = Σ λ_source`.

**Output: stacked bar chart or area plot** showing the contribution of each source to the driver torque (or any joint reaction) vs. input angle. This immediately answers: "Is peak torque driven by gravity, inertia, or the spring?"

### Decomposition Axes

| Axis | What it answers |
|------|----------------|
| **By source** (gravity / springs / dampers / external / inertia) | What is driving the load? |
| **By body** | Which link is the heaviest contributor? Useful for mass reduction |
| **By joint** | Which joint sees the worst load? Where should I use a larger bearing? |
| **By load case** | How do loads change across operating scenarios? |
| **By input-angle region** | Where in the cycle is the mechanism working hardest? |

### Driver Effort Breakdown

The driver's Lagrange multiplier `λ_driver` is the single most important output for actuator sizing. The source decomposition applied to `λ_driver` gives:

```
T_driver_total = T_gravity + T_springs + T_dampers + T_external + T_inertia
```

Plotted as a stacked chart vs. input angle, this shows the engineer exactly where the motor effort comes from and which sources can be reduced (e.g., adding a counterbalance spring to cancel gravity contribution).

---

## Mechanism Health Panel

A summary dashboard computed at the end of every analysis run. Provides at-a-glance assessment of mechanism quality without requiring the engineer to interpret raw plots.

### Health Metrics

| Metric | Definition | Warning threshold | Interpretation |
|--------|-----------|------------------|----------------|
| Min transmission angle | `min(μ)` over input sweep | < 40° | Poor force transmission in part of the cycle |
| Min singular value | `min(σ_min(Φ_q))` over input sweep | < 1e-6 | Mechanism passes near singularity |
| Peak condition number | `max(κ(Φ_q))` over input sweep | > 1e8 | Reactions unreliable somewhere in the cycle |
| Peak driver torque | `max(|T_driver|)` | (user-defined motor limit) | Motor may be undersized |
| RMS driver torque | `√(mean(T_driver²))` | (user-defined continuous rating) | Thermal sizing check |
| Peak bearing load | `max(F_res)` at each joint | (user-defined per joint) | Bearing may be undersized |
| Branch continuity | Any branch jumps detected during sweep? | Any detected | Results may be on wrong assembly mode |
| Constraint drift (fwd. dynamics) | `max(‖Φ(q)‖)` over time | > 1e-7 | Integration accuracy degraded |
| Energy balance error (fwd. dynamics) | `|KE + PE + dissipated - work_in| / max(|KE|)` | > 0.01 | Integrator losing or creating energy |

### Health Summary Format

```
MECHANISM HEALTH — 4-bar crank-rocker
────────────────────────────────────────
  Min transmission angle:   38.2°    ⚠ below 40° at input = 172°
  Peak condition number:    2.4e5    ✓
  Peak driver torque:       1.83 N·m ✓ (limit: 3.0 N·m)
  RMS driver torque:        0.91 N·m ✓ (limit: 1.5 N·m)
  Peak bearing load (J2):   47.3 N   ✓
  Peak bearing load (J3):   62.1 N   ⚠ approaching limit (70 N)
  Branch continuity:        OK       ✓
  Dominant load source:     Gravity (68% of peak driver torque)
────────────────────────────────────────
```

This panel is computed automatically — no configuration needed beyond the mechanism definition and optional user-specified limits. It should be the first thing an engineer sees after a run.

---

## Force Element State

Each force element reports its internal state vs. input angle or time:

| Element type | Reported state |
|---|---|
| Linear spring | Current length, elongation from free length, force magnitude |
| Torsion spring | Current relative angle, deflection from free angle, torque |
| Viscous damper | Relative velocity, damping force |
| Rotary damper | Relative angular velocity, damping torque |
| Coulomb friction | Joint speed, friction torque, stick/slip indicator |
| Motor | Speed, torque, operating point on T-ω curve, power |
| Linear actuator | Stroke, speed, force |
| Gas spring | Stroke, force, pressure (computed from volume) |

---

## Load Cases and Study Management

Real engineering analysis is rarely a single run. Users need to compare across scenarios:

### Load Case Structure

```
LoadCase {
    id: string
    label: string                          // e.g., "Nominal", "Max payload", "Cold start"
    mechanism_overrides: {
        gravity_angle: float | null        // override mechanism default
        gravity_magnitude: float | null
    }
    point_mass_overrides: {                // add/modify/remove point masses
        add: [PointMass]
        modify: {id: PointMass_partial}
        remove: [PointMass.id]
    }
    force_element_overrides: {             // modify force element parameters
        {element_id: {param_name: value}}
    }
    driver_override: Driver | null         // different motion profile
}
```

A load case modifies the base mechanism without duplicating it. Multiple load cases can be run in batch, and results are tagged by load case ID for comparison.

### Study

```
Study {
    mechanism_file: string
    load_cases: [LoadCase]
    analysis_modes: ["kinematic", "static", "inverse_dynamics"]
    sweep_range: {start, end, steps}
    outputs_requested: [string]           // which outputs to compute and save
}
```

A study runs multiple load cases through specified analysis modes and collects results. The output includes comparison tables (envelope values across load cases) and overlay plots.

---

## Output Coordinate Frames

All outputs use consistent coordinate conventions:

| Context | Convention |
|---|---|
| Global frame | Right-handed: X right, Y up. Origin at (0,0). All body positions and global forces use this frame |
| Body-local frame | Origin at body's local (0,0). Rotates with the body. Attachment points and CG are defined in this frame |
| Joint-local frame | For revolute: no preferred orientation (just a point). For prismatic: axis along slide direction, perpendicular direction defined by 90° rotation |
| Display frame | Same as global, but scaled to mm for length and degrees for angles |

When reaction forces are reported in "local" coordinates, the frame is always identified (which body's frame, or which joint's axis system).
