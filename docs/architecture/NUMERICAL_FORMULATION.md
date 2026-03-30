# Numerical Formulation

This document defines the exact mathematical formulation used by all solvers. It is the implementation reference — every equation here maps directly to code.

---

## Generalized Coordinate Vector

For a mechanism with `n` moving bodies (excluding ground), the generalized coordinate vector is:

```
q = [x₁, y₁, θ₁, x₂, y₂, θ₂, ..., xₙ, yₙ, θₙ]ᵀ
```

Each moving body contributes 3 coordinates:
- `xᵢ, yᵢ` — global position of the body's local frame origin (meters)
- `θᵢ` — orientation angle of the body's local frame relative to global (radians)

**Total unconstrained coordinates:** `n_coords = 3 * n_moving_bodies`

The body's local frame origin is the point `(0, 0)` in its local coordinate system. All attachment points, CG location, and coupler points are defined relative to this origin in the body's local frame.

**Ground body** is excluded from `q`. Its pose is fixed at `(0, 0, 0)` in the global frame. Ground attachment points are at fixed global positions.

### Coordinate Bookkeeping

Each body is assigned a contiguous block of 3 indices in `q`:

```
body_i.q_start = 3 * i        // where i is the body's index (0-based, among moving bodies)
body_i.x_idx = 3 * i
body_i.y_idx = 3 * i + 1
body_i.θ_idx = 3 * i + 2
```

This mapping is stored in `state.py` and used by all constraint and force element assembly code.

---

## Transformation Utilities

Given body `i` with pose `(xᵢ, yᵢ, θᵢ)` and a point `sᵢ = (sₓ, sᵧ)` in body-local coordinates, the global position of that point is:

```
r_point = rᵢ + Aᵢ * sᵢ
```

Where:
```
rᵢ = [xᵢ, yᵢ]ᵀ

Aᵢ = [cos θᵢ  -sin θᵢ]
     [sin θᵢ   cos θᵢ]
```

The partial derivatives needed for Jacobians:

```
∂(Aᵢ * sᵢ)/∂θᵢ = Bᵢ * sᵢ

where Bᵢ = [-sin θᵢ  -cos θᵢ] = ∂Aᵢ/∂θᵢ
            [ cos θᵢ  -sin θᵢ]
```

These appear in every constraint Jacobian and every force-to-Q conversion.

---

## Constraint Equations by Joint Type

### Revolute Joint

A revolute joint between body `i` at local point `sᵢ` and body `j` at local point `sⱼ` constrains the two points to be coincident in the global frame.

**Constraint (2 equations):**

```
Φ_rev = rᵢ + Aᵢ * sᵢ - rⱼ - Aⱼ * sⱼ = 0
```

This is a 2×1 vector equation (x and y components).

**Jacobian rows** (2 rows × n_coords columns):

For body `i` columns (if body `i` is not ground):
```
∂Φ/∂xᵢ = [1, 0]ᵀ
∂Φ/∂yᵢ = [0, 1]ᵀ
∂Φ/∂θᵢ = Bᵢ * sᵢ
```

For body `j` columns (if body `j` is not ground):
```
∂Φ/∂xⱼ = [-1, 0]ᵀ
∂Φ/∂yⱼ = [0, -1]ᵀ
∂Φ/∂θⱼ = -Bⱼ * sⱼ
```

If either body is ground, its columns do not exist in `Φ_q` (ground has no entries in `q`), and the corresponding `r + A*s` term becomes a constant in the constraint equation.

### Prismatic Joint

A prismatic joint between body `i` and body `j` allows relative translation along one axis and constrains relative rotation to zero.

Define:
- `êᵢ` = unit vector along the slide axis in body `i`'s local frame (given as `axis_local_i`)
- `n̂ᵢ` = unit vector perpendicular to `êᵢ` in body `i`'s local frame (rotate `êᵢ` by 90°)
- `sᵢ`, `sⱼ` = attachment points in local frames
- `d = rⱼ + Aⱼ * sⱼ - rᵢ - Aᵢ * sᵢ` = global vector from point_i to point_j

**Constraint (2 equations):**

```
Φ_prism[0] = n̂ᵢ_global · d = 0        // no displacement perpendicular to slide axis
Φ_prism[1] = θⱼ - θᵢ - Δθ₀ = 0        // no relative rotation (Δθ₀ is initial relative angle)
```

Where `n̂ᵢ_global = Aᵢ * n̂ᵢ` rotates with body `i`.

The displacement along the slide axis is:
```
s_slide = êᵢ_global · d
```

This is not constrained — it is the free coordinate. When a driver is attached to a prismatic joint, the driver constraint prescribes `s_slide` as a function of time.

**Jacobian rows:** Derived by differentiating both constraint equations with respect to all entries of `q`. The perpendicular-displacement constraint involves products of rotation matrices and attachment point vectors. The no-rotation constraint is simpler: `∂Φ[1]/∂θⱼ = 1`, `∂Φ[1]/∂θᵢ = -1`, all position partials are zero.

The full Jacobian expressions for the prismatic joint are lengthy. See implementation in `core/constraints.py` with inline derivations.

### Prismatic Joint Conventions — Worked Examples

Prismatic joints are the most convention-sensitive joint type. Sign errors, axis ownership confusion, and offset misinterpretation are the primary source of debugging time. These four canonical cases resolve all ambiguities.

#### Convention Summary

| Decision | Convention | Rationale |
|----------|-----------|-----------|
| Which body owns the slide axis? | **Body `i`** always owns the axis. `axis_local_i` is defined in body `i`'s local frame. The axis rotates with body `i`. | Consistent with the constraint equation: `n̂ᵢ_global = Aᵢ * n̂ᵢ` uses body `i`'s rotation matrix. |
| Positive stroke direction | Positive `s_slide` means point `j` has moved in the **positive `êᵢ` direction** relative to point `i`. | `s_slide = êᵢ_global · d` where `d` points from `i` to `j`. Positive = extension. |
| Initial offset `Δθ₀` | Computed from the initial assembly configuration: `Δθ₀ = θⱼ_init - θᵢ_init`. | Not user-specified. The solver captures it at first assembly. |
| Zero-stroke reference | `s_slide = 0` when point `j` coincides with point `i` projected onto the slide axis. The initial stroke is computed from the initial assembly. | Stroke is measured from geometric coincidence, not from an arbitrary datum. |

#### Case 1 — Slider Block on Horizontal Ground Rail

```
Ground body (body_i):
    attachment_points: {"rail": (0.2, 0.0)}    # rail reference point
    axis_local_i: (1, 0)                        # horizontal slide axis (global X)

Slider block (body_j):
    attachment_points: {"slide_pt": (0.0, 0.0)} # block center
```

- Body `i` = ground, so `θᵢ = 0` always. The slide axis is fixed in the global X direction.
- `n̂ᵢ = (0, 1)` → perpendicular constraint enforces: block stays on the horizontal rail (no vertical displacement).
- `Δθ₀ = θ_block_init - 0`. If the block starts at `θ = 0`, then `Δθ₀ = 0` and the block remains horizontal.
- Positive stroke = block moves to the right (+X).
- `s_slide = x_block - 0.2` (offset from ground rail reference point along the axis).

**Reaction forces:**
- `λ[0]` = perpendicular force (normal force pushing block against rail, in global Y direction)
- `λ[1]` = anti-rotation torque (moment preventing block from spinning)

#### Case 2 — Slider Block on a Moving Body's Axis

```
Rocker (body_i):
    attachment_points: {"pivot": (0.0, 0.0), "slide_start": (0.05, 0.0)}
    axis_local_i: (1, 0)    # slide axis along the rocker's length

Slider block (body_j):
    attachment_points: {"pin": (0.0, 0.0)}
```

- Body `i` = rocker (moving). The slide axis **rotates with the rocker**.
- When rocker is at `θᵢ = 30°`, the slide axis points at 30° from horizontal in the global frame.
- `n̂ᵢ_global = Aᵢ * (0, 1) = (-sin 30°, cos 30°)` → perpendicular constraint enforces: block stays on the rocker's centerline.
- Positive stroke = block moves away from the rocker pivot (toward the rocker tip).
- `Δθ₀ = θ_block_init - θ_rocker_init`. The block maintains its initial angular relationship with the rocker.

**This is the case that causes the most bugs.** The axis direction in the constraint equation changes every timestep because body `i` rotates. Verify by checking that `n̂ᵢ_global · d = 0` remains satisfied after solving.

#### Case 3 — Linear Actuator (Stroke as Driven Coordinate)

```
Body_i (base bracket):
    attachment_points: {"mount": (0.0, 0.0)}
    axis_local_i: (1, 0)    # actuator extends along body_i's local X

Body_j (actuator rod tip):
    attachment_points: {"tip": (0.0, 0.0)}
```

When a driver is attached to this prismatic joint:
- `s_slide` is the measured stroke (actuator extension length)
- Positive stroke = rod extends (tip moves away from base in the `êᵢ` direction)
- The driver constraint: `s_slide(q) - f(t) = 0`
- `λ_driver` = required actuator force (positive = pushing in the extension direction)
- If `λ_driver > 0`, the actuator must push to maintain the prescribed extension
- If `λ_driver < 0`, the actuator must pull (or the load is assisting extension)

**Sign convention for actuator force:** `λ_driver` positive means the actuator must exert force in the positive stroke direction (extension). This matches the physical intuition: positive lambda = actuator pushes the rod out.

#### Case 4 — Reaction Force Decomposition

For any prismatic joint, the two Lagrange multipliers map to physical forces as follows:

| Multiplier | Physical meaning | Coordinate frame |
|-----------|-----------------|------------------|
| `λ[0]` (perpendicular constraint) | Force perpendicular to slide axis, pushing body `j` back onto the rail | In the direction of `n̂ᵢ_global` at the current configuration |
| `λ[1]` (rotation constraint) | Anti-rotation torque preventing relative rotation between body `i` and body `j` | Pure torque (N·m), sign follows right-hand rule |

**To convert to global force components:**
```
F_perp_global = λ[0] * n̂ᵢ_global    # force vector perpendicular to rail
T_anti_rot = λ[1]                     # torque on body_j (equal and opposite on body_i)
```

**To convert to body-local components:**
```
F_perp_local_i = λ[0] * n̂ᵢ          # always along the local perpendicular (by construction)
```

Note: there is no "along-axis" reaction force from the prismatic joint itself — the slide axis is the free direction. Any force along the axis comes from force elements (springs, dampers, actuators) or from the driver constraint (if driven).

### Fixed Joint

A fixed joint constrains all relative motion to zero.

**Constraint (3 equations):**

```
Φ_fixed[0:2] = rᵢ + Aᵢ * sᵢ - rⱼ - Aⱼ * sⱼ = 0    // coincident points (2 eqs)
Φ_fixed[2]   = θⱼ - θᵢ - Δθ₀ = 0                      // locked relative angle (1 eq)
```

Jacobian: combination of revolute Jacobian (first 2 rows) and the rotation constraint from the prismatic joint (third row).

---

## Driver Constraints

A driver on a joint adds one constraint equation to the system.

### Revolute Driver

```
Φ_driver = θ_relative(q) - f(t) = 0
```

Where `θ_relative = θⱼ - θᵢ` is the relative angle between the two bodies connected by the joint, and `f(t)` is the prescribed motion.

**Jacobian:** `∂Φ_driver/∂θᵢ = -1`, `∂Φ_driver/∂θⱼ = 1`, all other partials zero. (Adjusted if either body is ground.)

**Time derivative:** `Φ_t = -ḟ(t)` (used in velocity equation).

### Prismatic Driver

```
Φ_driver = s_slide(q) - f(t) = 0
```

Where `s_slide = êᵢ_global · d` is the displacement along the slide axis.

**Jacobian:** derived from the dot product expression. Involves both body positions, angles, and the attachment point vectors.

### Driver as Lagrange Multiplier Source

The Lagrange multiplier `λ_driver` associated with the driver constraint row is the generalized force required to enforce the prescribed motion:

- For a revolute driver: `λ_driver` is the required input **torque** (N·m)
- For a prismatic driver: `λ_driver` is the required input **force** (N)

This is the primary output for static analysis and inverse dynamics. No special extraction is needed — it comes directly from solving the constrained equilibrium equations.

---

## Global System Assembly

### Constraint Assembly

Given `m` total constraint equations (from all joints + all drivers):

```
Φ(q, t) = [Φ_joint1, Φ_joint2, ..., Φ_jointk, Φ_driver1, ...]ᵀ    (m × 1)
```

The global Jacobian:

```
Φ_q = ∂Φ/∂q    (m × n_coords)
```

is assembled by placing each joint's Jacobian rows into the correct columns (determined by the body index mapping from coordinate bookkeeping).

### Mass Matrix Assembly

The mass matrix `M` is block-diagonal:

```
M = diag([M₁, M₂, ..., Mₙ])    (n_coords × n_coords)
```

Where each body's block is:

```
Mᵢ = [mᵢ   0    0  ]
     [ 0   mᵢ   0  ]
     [ 0    0   Izzᵢ]
```

`mᵢ` and `Izzᵢ` are the composite values (body + all attached point masses, with parallel axis theorem applied for Izz).

### Generalized Force Assembly

The generalized force vector `Q` (n_coords × 1) is assembled by iterating over all force elements:

```
Q = Σ Q_element_k
```

Each force element computes its contribution using the helper utilities:

**Point force at location `s` on body `i`, force vector `F` in global frame:**
```
Q[xᵢ] += Fₓ
Q[yᵢ] += Fᵧ
Q[θᵢ] += (Bᵢ * sᵢ)ᵀ · F    // which equals (rₚ - rᵢ) × F in 2D
```

Where `rₚ` is the global position of the application point. The θ term is the moment of `F` about the body's coordinate origin.

**Pure torque `T` on body `i`:**
```
Q[θᵢ] += T
```

**Gravity on body `i` with composite mass `m` and composite CG at local `s_cg`:**
```
F_gravity = m * g_vector    // g_vector is [gₓ, gᵧ] in global frame
Q[xᵢ] += m * gₓ
Q[yᵢ] += m * gᵧ
Q[θᵢ] += (Bᵢ * s_cg)ᵀ · F_gravity
```

**Spring between point `sᵢ` on body `i` and point `sⱼ` on body `j`:**

Compute global positions of both attachment points, compute the spring force vector `F` along the line between them, then apply `F` to body `i` at `sᵢ` and `-F` to body `j` at `sⱼ` using the point-force formula above.

---

## Solver Equations by Analysis Mode

### Kinematic Analysis

**Position:** solve the nonlinear system:
```
Φ(q, t) = 0
```

Using Newton-Raphson:
```
q_{k+1} = q_k - Φ_q⁻¹ * Φ(q_k, t)
```

(In practice, solve the linear system `Φ_q * Δq = -Φ` at each iteration.)

Initial guess: previous converged position (for sweeps) or user-supplied (for first solve). The solver tracks the solution branch by continuation — it does not automatically detect or switch assembly modes. The user selects the initial configuration (e.g., "open" vs. "crossed" 4-bar) via the initial guess.

**Velocity:** linear solve (no iteration):
```
Φ_q * q̇ = -Φ_t
```

`Φ_t` is nonzero only for driven joints (the time derivative of the driver function).

**Acceleration:** linear solve (no iteration):
```
Φ_q * q̈ = γ
```

Where:
```
γ = -(Φ_q * q̇)_q * q̇ - 2 * Φ_qt * q̇ - Φ_tt
```

The `γ` vector (sometimes called the "right-hand side acceleration" or "constraint acceleration") accounts for centripetal and Coriolis-like terms from the constraints. Each joint type computes its contribution to `γ` alongside its `Φ` and `Φ_q`.

### Static Force Analysis

At each configuration from kinematic analysis, solve:

```
Φ_q^T * λ = -Q
```

This is a linear system for the Lagrange multipliers `λ` (m × 1). The system has `m` equations (one per constraint) and `m` unknowns.

**Partitioning:** The Lagrange multipliers include:
- `λ_joints` — passive reaction forces/torques at joints (what the bearings carry)
- `λ_driver` — the required input effort (what the actuator must provide)

The driver's `λ` is the primary output: required input torque (revolute) or force (prismatic) at this configuration.

**Note:** If the mechanism has redundant constraints (`constraint_rank < m`), the system `Φ_q^T * λ = -Q` is underdetermined. The passive reactions in redundant directions are indeterminate — there are infinitely many valid reaction sets. The solver should use a minimum-norm solution (pseudoinverse) and warn the user. The driver reaction is still uniquely determined if the driver constraint is independent.

### Singularity Analysis and Conditioning

The equilibrium solve `Φ_q^T * λ = -Q` (statics) and `Φ_q^T * λ = Q - M*q̈` (inverse dynamics) depend critically on the conditioning of `Φ_q`. Near singular configurations, multipliers blow up and reaction recovery becomes unreliable. Three distinct questions must be separated:

#### 1. Existence of Equilibrium

Does a static equilibrium exist at this configuration? This fails only if the applied loads `Q` have a component in the null space of `Φ_q^T` — that is, the applied loads cannot be balanced by any combination of constraint forces. For a well-posed mechanism with a driver, this is rare, but it can happen at exact singularities where `Φ_q` loses rank.

**Detection:** Compute the SVD of `Φ_q`. If `rank(Φ_q) < m` and the projection of the right-hand side onto the null space of `Φ_q^T` is nonzero, no equilibrium exists. Report: "No static equilibrium — applied loads cannot be balanced at this configuration."

#### 2. Uniqueness of Multiplier Solution

Are the Lagrange multipliers uniquely determined? This depends on whether the system is over-determined, exactly-determined, or under-determined:

| Condition | `rank(Φ_q)` vs. `m` | Multiplier status |
|-----------|---------------------|-------------------|
| Full rank, square | `rank = m = n_coords` | Unique (fully constrained) |
| Full rank, rectangular | `rank = m < n_coords` | Unique (typical for well-posed mechanisms) |
| Rank-deficient | `rank < m` | **Not unique** — redundant constraints create indeterminate passive reactions |

When multipliers are not unique, the driver reaction `λ_driver` is still uniquely determined *if the driver constraint is independent* (i.e., removing the driver row does not change the rank deficiency). Passive reactions in redundant directions are physically indeterminate — the minimum-norm pseudoinverse gives one valid solution, but it is not "the" answer. Report which constraint directions are redundant.

#### 3. Conditioning of Reaction Recovery

Even when multipliers are technically unique, ill-conditioning of `Φ_q` amplifies small errors in `Q` (or `Q - M*q̈`) into large errors in `λ`. This happens near toggle points, dead centers, and singular configurations.

**Monitoring (computed at every configuration):**

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Smallest singular value `σ_min` | From SVD of `Φ_q` | Approaches zero at singularity. When `σ_min < ε_warn`, reactions are unreliable |
| Condition number `κ(Φ_q)` | `σ_max / σ_min` | When `κ > 1e8`, flag: "reactions may be unreliable at this configuration" |
| Driver mechanical advantage | `∂(output displacement) / ∂(input displacement)`, or equivalently the ratio of driver λ to output force | Approaches infinity at toggle (force amplification), approaches zero at dead center (motion impossible) |

**Warning thresholds (configurable):**

```
σ_min < 1e-8:    "Near-singular configuration — reactions unreliable"
κ(Φ_q) > 1e10:   "Ill-conditioned Jacobian — reaction magnitudes may be inaccurate"
|MA| > 1e4:       "Near toggle — mechanical advantage approaching infinity"
|MA| < 1e-4:      "Near dead center — mechanism approaching lock-up"
```

**Null-space reporting:** When `Φ_q` is rank-deficient or nearly so, report the null-space direction(s) of `Φ_q^T`. These indicate *which physical directions* have indeterminate reactions. For example, in a parallelogram linkage, the null-space direction corresponds to the internal tension/compression that the links can trade between each other without affecting equilibrium.

#### Interaction with Transmission Angle

Singularity of `Φ_q` is directly related to transmission angle: `σ_min → 0` corresponds to transmission angle → 0° or 180°. The transmission angle metric (see `docs/ENGINEERING_OUTPUTS.md`) and the `σ_min` metric carry the same information in different forms. Both should be reported together so the engineer sees the geometric interpretation (transmission angle) alongside the numerical reliability indicator (condition number).

### Inverse Dynamics

Given `q`, `q̇`, `q̈` from kinematic analysis at each timestep, solve:

```
M * q̈ + Φ_q^T * λ = Q

⟹  Φ_q^T * λ = Q - M * q̈
```

This is the same linear system as statics, but with `Q` replaced by `Q - M * q̈`. The inertial terms `M * q̈` are treated as fictitious forces (D'Alembert principle).

Again, `λ_driver` gives the required actuator effort including inertial loads.

### Forward Dynamics

Solve the coupled DAE:

```
M * q̈ + Φ_q^T * λ = Q       (equations of motion)
Φ(q, t) = 0                   (constraint equations)
```

This is an index-3 DAE. Direct numerical integration is difficult. Options:

**Option A — Index reduction + Baumgarte stabilization:**

Differentiate the constraints twice to get acceleration-level equations:
```
Φ_q * q̈ = γ
```

Combine with the EOM:
```
[M    Φ_q^T] [q̈] = [Q]
[Φ_q    0  ] [λ ] = [γ]
```

Solve this linear system at each timestep for `q̈` and `λ`, then integrate `q̈` → `q̇` → `q`.

**Problem:** constraint drift. After many timesteps, `Φ(q)` drifts away from zero because we only enforce `Φ̈ = 0`, not `Φ = 0`.

**Fix:** Baumgarte stabilization — replace `γ` with:
```
γ_stabilized = γ - 2α * (Φ_q * q̇ + Φ_t) - β² * Φ
```

Where `α` and `β` are stabilization parameters. Alternatively, project `q` back onto the constraint manifold periodically.

**Option B — Coordinate partitioning:**

Partition `q` into independent coordinates `qᵢ` (size = DOF) and dependent coordinates `q_d`:
```
q = [qᵢ, q_d]
```

Use the constraint equations to express `q_d` as implicit functions of `qᵢ`. Derive the EOM purely in terms of `qᵢ` (a true ODE), and integrate with standard methods.

**Problem:** the partition may become singular at certain configurations, requiring re-partitioning.

**Option C — Direct DAE solver:**

Use a DAE-capable integrator (implicit BDF methods, Radau IIA). These handle the algebraic constraints directly without manual index reduction. Most robust for stiff systems.

**Implementation note:** Start with Option A (simplest), monitor constraint drift, add projection if needed. Consider Option C if stiffness or drift becomes problematic.

---

## Reaction Force Post-Processing

Joint reaction forces are extracted from the Lagrange multipliers `λ`. Each joint's multipliers correspond to the forces/torques the joint exerts on the connected bodies.

### Revolute Joint Reactions

A revolute joint has 2 constraint equations → 2 Lagrange multipliers `[λₓ, λᵧ]`. These are the **global X and Y components** of the reaction force at the joint.

**Output formats:**

| Format | Definition |
|---|---|
| Global components | `Fx = λₓ`, `Fy = λᵧ` |
| Resultant magnitude | `F_res = √(λₓ² + λᵧ²)` |
| Direction angle | `α = atan2(λᵧ, λₓ)` |
| Body-local components | `F_local = Aᵢᵀ * [λₓ, λᵧ]` — reaction in body `i`'s frame |
| Radial/tangential (at joint) | Radial = along line from body CG to joint. Tangential = perpendicular. Useful for bearing load decomposition |

### Prismatic Joint Reactions

A prismatic joint has 2 constraint equations → 2 Lagrange multipliers. The first corresponds to the perpendicular-to-axis force. The second corresponds to the anti-rotation torque.

### Fixed Joint Reactions

A fixed joint has 3 multipliers → `[Fx, Fy, T]`: reaction force (2 components) and reaction torque.

### Driver Reactions

A driver has 1 constraint equation → 1 Lagrange multiplier:
- Revolute driver: `λ` = required input torque (N·m)
- Prismatic driver: `λ` = required input force (N)

This is always extracted and reported as a primary output.

---

## Velocity and Acceleration of Arbitrary Points

Given a solved configuration `(q, q̇, q̈)`, the velocity and acceleration of any point `s` on body `i` (in body-local coordinates) are:

**Global position:**
```
r_p = rᵢ + Aᵢ * s
```

**Velocity:**
```
ṙ_p = ṙᵢ + Bᵢ * s * θ̇ᵢ
```

Which is:
```
ṙ_p = [ẋᵢ - sₓ sin θᵢ * θ̇ᵢ - sᵧ cos θᵢ * θ̇ᵢ]
      [ẏᵢ + sₓ cos θᵢ * θ̇ᵢ - sᵧ sin θᵢ * θ̇ᵢ]
```

**Acceleration:**
```
r̈_p = r̈ᵢ + Bᵢ * s * θ̈ᵢ + ∂(Bᵢ)/∂θᵢ * s * θ̇ᵢ²
```

The last term is the centripetal acceleration:
```
∂Bᵢ/∂θᵢ * s * θ̇ᵢ² = -Aᵢ * s * θ̇ᵢ²
```

So:
```
r̈_p = r̈ᵢ + Bᵢ * s * θ̈ᵢ - Aᵢ * s * θ̇ᵢ²
```

This is used for coupler point analysis and for computing force element velocities.

---

## Numerical Considerations

### Dimensionless Scaling and Tolerance Strategy

The generalized coordinate vector `q` mixes translational coordinates (meters) and rotational coordinates (radians). For a typical desk-scale mechanism with links of 10–100 mm, translational coordinates are O(0.01–0.1) while angular coordinates are O(1). This creates conditioning issues:

- A convergence tolerance of `1e-10` applied uniformly to `‖Φ‖` means different things for the translational constraint residuals (1e-10 m ≈ 0.1 nm, well below machine precision for mechanism-scale problems) and rotational residuals (1e-10 rad ≈ 6e-9°, similarly tight but in a different physical domain).
- The Jacobian `Φ_q` has columns with different physical dimensions (∂Φ/∂x has units of 1, ∂Φ/∂θ has units of meters), which affects SVD-based conditioning metrics.

**Scaling approach:** apply characteristic length scaling to normalize the system:

```
L_char = characteristic length of the mechanism (e.g., longest link length or diagonal of bounding box)

Scaled convergence test:
  translational residuals: |Φ_trans| < ε * L_char
  rotational residuals:    |Φ_rot| < ε
  where ε ≈ 1e-10 (dimensionless)
```

For Jacobian conditioning assessment, scale columns by `[L_char, L_char, 1]` per body to make the SVD dimensionally consistent. The reported condition number should be of the scaled Jacobian.

This does not affect the equations or the solver — it only affects convergence testing and conditioning metrics. The solver works in SI throughout.

### Newton-Raphson Convergence

The kinematic position solver uses Newton-Raphson on `Φ(q, t) = 0`. Convergence depends on:

- **Initial guess quality:** for a sweep over input angle, the predictor-corrector continuation provides the initial guess (see "Branch Management" below). For the first solve, the user provides an initial guess (or the assembly routine uses a heuristic). Step sizes of 1–2° for the input angle typically give good convergence.
- **Jacobian conditioning:** near singular configurations (toggle points), the Jacobian becomes ill-conditioned. The solver monitors `σ_min` and condition number (of the scaled Jacobian) and warns or reduces step size.
- **Convergence tolerance:** position residual `‖Φ‖ < ε` with dimensionless scaling (see above). Equivalent to ~0.1 nm positional accuracy for mechanism-scale lengths.

### Jacobian Computation

All Jacobians are computed analytically (not by finite differences). Each constraint type implements its own Jacobian function. This is essential for:
- Reliable Newton-Raphson convergence
- Accurate velocity and acceleration solutions (which are exact given exact Jacobians)
- Correct Lagrange multiplier extraction
- Proper conditioning assessment

### Branch Management and Assembly Mode Tracking

A mechanism with `n` DOF may have multiple valid closed-loop configurations (assembly modes) for the same input coordinate value. For example, a 4-bar linkage has two assembly modes: "open" and "crossed." The solver must stay on the correct branch throughout a sweep and detect when it has failed to do so.

**Raw previous-state seeding is not sufficient.** When the step size is too large, the Jacobian is ill-conditioned, or the mechanism passes near a toggle point, Newton-Raphson can converge to a root on a different branch. This produces silently wrong results — the solver reports convergence, but the mechanism has jumped configurations.

#### Predictor-Corrector Continuation

Instead of using only the previous converged solution as the initial guess, use a predictor based on the two most recent solutions:

```
q_predict = q_{k} + (q_{k} - q_{k-1}) * (t_{k+1} - t_k) / (t_k - t_{k-1})
```

This linear extrapolation follows the solution curve and provides a much better initial guess than raw seeding, especially on curved coupler paths and near inflection points.

For the first two solves (where history is insufficient), fall back to user-provided initial guess and then single-step seeding.

#### Orientation Invariant Tracking

To detect branch jumps after convergence, track orientation invariants at each step:

- **Joint-angle sign convention:** for each revolute joint, compute the signed relative angle `θ_rel = θ_j - θ_i`. A branch flip typically reverses the sign of one or more relative angles.
- **Cross-product test at loops:** for a closed kinematic loop, compute the signed area (cross product) of the triangle formed by three consecutive joint positions. A sign change indicates the loop has "flipped."
- **Body-pose continuity:** compute `‖q_{k+1} - q_predict‖`. If this exceeds a threshold proportional to step size (e.g., `C * Δt` where `C` is calibrated from recent steps), flag a potential branch jump.

When a branch jump is detected:
1. Halve the step size
2. Re-solve from the last good state with the smaller step
3. If the jump persists at minimum step size, report it as a genuine branch change (the mechanism has passed through a bifurcation)

#### Automatic Step-Size Adaptation

Adjust the sweep step size based on Newton-Raphson behavior:

| Condition | Action |
|-----------|--------|
| NR converges in ≤ 3 iterations | Step size is adequate. May increase by 1.5× (up to user max) |
| NR converges in 4–6 iterations | Step size is marginal. Hold current |
| NR converges in 7+ iterations | Reduce step size by 0.5× |
| NR fails to converge | Halve step size and retry. If minimum step size is reached, report failure location |
| Branch jump detected | Halve step size and retry from last good state |

Minimum step size: `1e-4` radians (≈ 0.006°). If convergence fails at minimum step, the solver has hit a genuine singularity or bifurcation.

#### Arc-Length Continuation (Future Enhancement)

For mechanisms that pass through turning points (where the input-output relationship folds back), standard parameter-stepping fails. Arc-length continuation parameterizes the solution curve by its arc length rather than the input variable, allowing the solver to follow folds and turning points.

This is deferred to a later phase but the solver architecture should not preclude it: the continuation interface should accept a general parameterization, not hardcode the input joint angle as the independent variable.

#### What Is Not Included (v1)

- Automatic branch enumeration (finding all assembly modes for a given input)
- Mode switching (intentionally jumping to a different branch)
- Bifurcation classification (fold, pitch-fork, etc.)

These are acknowledged as valuable for synthesis and optimization but are not needed for the primary use case of sweeping a known mechanism on a known branch.

### Constraint Drift (Forward Dynamics)

During forward dynamics integration, the position-level constraints `Φ(q) = 0` will drift due to numerical integration error, even if the acceleration-level constraints are satisfied.

**Monitoring:** compute `‖Φ(q)‖` after each timestep. If it exceeds a tolerance (e.g., 1e-7 m), corrective action is needed.

**Correction options:**
1. Baumgarte stabilization (continuous, built into the acceleration equation)
2. Periodic projection: after each timestep, solve `Φ(q) = 0` via Newton-Raphson using the integrated `q` as the initial guess, then replace `q` with the converged result
3. Both (Baumgarte for stability, projection for insurance)
