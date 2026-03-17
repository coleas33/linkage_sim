# Architecture — Core Data Model

## Body–Constraint Incidence Model

The mechanism topology is a body–joint incidence model, not a simple edge graph.

A simple graph where nodes = joints and edges = links breaks on any body with more than two attachment points. Ternary links, coupler plates, bell-cranks, and slider blocks all have 3+ joint connections on a single rigid body. The edge-graph model cannot represent them without splitting one physical body into multiple artificial "links," which corrupts mass properties and complicates the solver.

Instead:

- **Bodies** are first-class rigid objects. Each body has a local coordinate frame, multiple named attachment points in that frame, and mass properties defined in that frame. A binary bar is a body with two attachment points. A ternary plate is a body with three. A slider block is a body with attachment points for the slide joint and for connected links. They are all the same object type.

- **Joint constraints** connect two bodies by referencing one attachment point on each body. The joint type determines how many DOF are removed and what constraint equations are generated.

- **Force elements** act between attachment points on bodies (or at joints) and contribute to the generalized force vector.

- **Point masses** attach to bodies at arbitrary locations and modify the body's composite mass properties.

This means a 4-bar, a Stephenson 6-bar with a ternary link, a Watt 6-bar, a slider-crank, or a bell-crank plate mechanism are all just different configurations of the same four object types.

---

## Data Model

### Body

```
Body {
    id: string
    attachment_points: {name: (x, y)}  // local coordinates, meters (SI)
    mass: float                         // kg
    cg_local: (x, y)                   // m, in body-local frame
    Izz_cg: float                       // kg·m², about CG, z-axis (out-of-plane)
    coupler_points: {name: (x, y)}     // arbitrary tracked points, body-local, meters
    render_shape: [(x, y)] | null       // optional polygon outline for GUI
    color: string                       // GUI rendering
}
```

**Notes:**

- The inertia is `Izz_cg` (about the z-axis through the CG), not `Ixx`. For planar motion in the x-y plane, the relevant rigid-body inertia is about the out-of-plane axis.
- `attachment_points` defines where joints and force elements can connect. Every joint references an attachment point by name.
- `coupler_points` are tracked for output (path tracing, velocity/acceleration) but are not connection points for joints or force elements.
- `render_shape` is purely visual. It does not affect analysis. If null, the GUI draws a line between attachment points (for bars) or a convex hull.

**Ground body:** A special body with `mass = 0`, `Izz_cg = 0`, fixed at the global origin. Ground attachment points define the locations of fixed pivots and slide bearings. The ground body is excluded from the generalized coordinate vector `q`.

### JointConstraint

```
JointConstraint {
    id: string
    type: "revolute" | "prismatic" | "fixed"
    body_i: Body.id
    point_i: string                    // attachment point name on body_i
    body_j: Body.id
    point_j: string                    // attachment point name on body_j

    // Prismatic-specific fields:
    axis_local_i: (x, y) | null       // unit vector defining slide direction in body_i frame
    ref_point_j: string | null         // reference attachment point on body_j for alignment

    limits: (min, max) | null          // joint travel limits (rad for revolute, m for prismatic)
    driven: bool                       // is this joint driven by a motion profile?
    driver: Driver | null              // motion profile (adds constraint rows to Φ)
}
```

**Revolute joint** — removes 2 translational DOF. Constrains the two attachment points to be coincident in the global frame. Allows relative rotation between the two bodies. See `docs/NUMERICAL_FORMULATION.md` for constraint equations.

**Prismatic joint** — removes 2 DOF (1 translational + 1 rotational). Constrains:
- No relative displacement perpendicular to the slide axis
- No relative rotation between the two bodies
- One translational DOF along `axis_local_i` remains free

The prismatic joint requires:
- `axis_local_i`: a unit vector in body_i's local frame defining the allowed translation direction. When body_i is ground, this is the global slide axis.
- `point_i` on body_i and `point_j` on body_j: these define where the slide constraint acts.
- `ref_point_j`: an optional second reference point on body_j used to define the perpendicular-to-axis constraint more robustly (for slider blocks with two guide points).

The measured displacement along the prismatic axis is the generalized coordinate for this joint when it is driven.

**Fixed joint** — removes 3 DOF (2 translational + 1 rotational). Constrains both relative position and relative angle to zero. Used for ground attachments and welded connections.

### Driver

A motion profile applied to a driven joint. **A driver is a constraint, not a force element.** It adds one row to the constraint equation system `Φ(q, t) = 0`, prescribing the joint coordinate as a function of time. The Lagrange multiplier associated with this constraint row is the required actuator effort (torque for revolute, force for prismatic).

```
Driver {
    type: "constant_speed" | "expression" | "interpolated"
    parameters: dict
    // constant_speed: {rate: float}                      // rad/s or m/s
    // expression: {expr: string, variable: "t"}          // safe expression, not eval()
    // interpolated: {times: [float], values: [float]}    // linearly interpolated
}
```

The driver constraint equation for a revolute joint:

```
Φ_driver: θ_relative(q) - f(t) = 0
```

For a prismatic joint:

```
Φ_driver: d_relative(q) - f(t) = 0
```

Where `θ_relative` or `d_relative` is the relative angle or displacement at the joint, extracted from the generalized coordinates, and `f(t)` is the prescribed motion from the driver.

**Why drivers are constraints, not forces:**
- In kinematic analysis, the driver prescribes motion. The solver finds the configuration that satisfies the driver constraint along with all joint constraints.
- In static analysis and inverse dynamics, the driver's Lagrange multiplier `λ_driver` is the required input torque/force. This falls out naturally from the constrained equilibrium equations without any special handling.
- If the driver were modeled as a force element, you would need a separate control loop to find the force that produces the desired motion — much more complex and numerically fragile.

### ForceElement

```
ForceElement {
    id: string
    type: string
    body_i: Body.id | null           // first body (null = ground)
    point_i: string | null           // attachment point on body_i
    body_j: Body.id | null           // second body (null = single-body loads)
    point_j: string | null           // attachment point on body_j
    joint_ref: JointConstraint.id | null  // for joint-based loads (friction)
    parameters: dict                 // type-specific
    evaluate(state, t) → Q_contribution  // returns contribution to generalized force vector
}
```

**Canonical internal representation:** All force elements return their contribution as a segment of the generalized force vector `Q`. There is only one return type — no choice between "Cartesian wrenches" or "generalized forces." Internally, everything becomes `Q`.

Helper utilities handle the conversion:

- `point_force_to_Q(body, local_point, F_global, q)` — converts a Cartesian force applied at a point on a body into the corresponding entries in `Q`, using the body's current pose and the Jacobian of the application point.
- `body_torque_to_Q(body, T)` — converts a pure torque on a body into `Q` (directly adds to the body's θ entry).
- `gravity_to_Q(body, g_vector)` — applies `m * g` at the body's composite CG.
- `joint_torque_to_Q(joint, T)` — applies equal and opposite torques to the two connected bodies' θ entries.

Force element implementations call these helpers. The solver sees only the assembled `Q` vector.

**Built-in force element types:**

| Type | Attachment | Parameters | Behavior |
|---|---|---|---|
| `linear_spring` | point_i on body_i, point_j on body_j | `k`, `free_length`, `preload`, `mode` | Force along line between points. `mode`: tension-only / compression-only / both |
| `torsion_spring` | joint_ref (revolute) | `k`, `free_angle`, `preload` | Torque proportional to relative angle at joint |
| `viscous_damper` | point_i on body_i, point_j on body_j | `c` | Force proportional to relative velocity along line between points |
| `rotary_damper` | joint_ref (revolute) | `c` | Torque proportional to relative angular velocity |
| `viscous_friction` | joint_ref (revolute or prismatic) | `c` | Torque/force proportional to relative velocity. Smooth, reliable, no approximation. Use as default friction model |
| `regularized_coulomb` | joint_ref | `mu_static`, `mu_kinetic`, `reg_velocity` | **Approximate** Coulomb friction via `tanh` regularization. See friction model note below |
| `gravity` | body_i only | (uses mechanism-level g) | `F = m * g` at composite CG |
| `motor_droop` | joint_ref (revolute) | `T_stall`, `omega_no_load` | `T = T_stall * (1 - ω/ω_no_load)` |
| `linear_actuator` | point_i on body_i, point_j on body_j | `force_table` or `force_const`, `speed_limit` | Force along line, function of stroke/velocity |
| `gas_spring` | point_i on body_i, point_j on body_j | `P_init`, `area`, `stroke_init`, `c_damp` | Pressure-coupled force + damping |
| `external_load` | point_i on body_i, direction | `expression` | Arbitrary `F(pos, vel, t)` via safe expression |
| `custom` | varies | `plugin_name` | Named plugin from registry. Local-only, non-portable |

**Friction model note — model tiers and their limitations:**

Friction in linkage mechanisms is deceptively complex. The built-in models are explicitly tiered by fidelity so engineers know what they're getting:

| Model | Type | Smooth? | Limitations | Use for |
|-------|------|---------|-------------|---------|
| `viscous_friction` | ForceElement | Yes | No static friction, no direction-dependent behavior. Force is zero at zero velocity. | Default friction model. Reliable in all analysis modes. Adequate when you need "some friction" for damping and energy dissipation estimates |
| `regularized_coulomb` | ForceElement | Yes (approximate) | **Not true sticking.** Uses `tanh(v / v_reg)` — force is continuous but nonzero even at zero velocity. At low velocities, the model produces a fictitious viscous-like zone. Can cause false creep near equilibrium, energy artifacts in forward dynamics, and incorrect static holding behavior | Statics and inverse dynamics where approximate Coulomb behavior is sufficient. **Not reliable for:** dead-point holding, precise stick-slip transitions, loaded joints near toggle, direction-reversal accuracy |
| True stick-slip (deferred) | Conditional | No | Requires complementarity or mode-switching solver. Changes the mathematical class of the problem | Phase 4B and beyond. Needed only for precision mechanism analysis and dead-point locking |

**The GUI and output reports must label these models explicitly.** An engineer seeing "friction: μ = 0.15" must also see whether it's viscous, regularized Coulomb, or true stick-slip. The model tier affects the reliability of the results and should not be hidden behind a generic "friction" label.

**User-defined loads note:** Raw Python lambdas are not used. They are not serializable, not portable, not safe to share, and not GUI-friendly. User-defined loads use either: (a) a safe expression language (math expressions referencing named state variables, parsed by `asteval` or similar — not `eval()`), or (b) a named plugin registry where Python callables are registered locally and referenced by name in the mechanism file. The file stores only the name. Plugin mode is marked local-only and non-portable.

### PointMass

```
PointMass {
    id: string
    body: Body.id
    position_local: (x, y)          // m, in body frame
    mass: float                      // kg
    label: string                    // user-facing name
}
```

When point masses are attached, the solver recomputes the body's composite mass properties before assembly:

- `m_composite = m_body + Σ m_point`
- `cg_composite = (m_body * cg_body + Σ m_point * pos_point) / m_composite`
- `Izz_composite = Izz_body + m_body * d_body² + Σ (m_point * d_point²)` (parallel axis theorem, where `d` is distance from new composite CG)

Multiple point masses can attach to the same body. The GUI displays them as markers on the body.

### Mechanism

```
Mechanism {
    schema_version: string              // e.g. "1.0.0" — semver for data format
    bodies: {id: Body}
    joints: {id: JointConstraint}
    force_elements: {id: ForceElement}
    point_masses: {id: PointMass}
    gravity_magnitude: float            // m/s², default 9.81
    gravity_angle: float                // radians from -Y, default 0 (straight down)
    dof_grubler: int                    // informational, computed from Grübler
    constraint_rank: int | null         // rank of Φ_q at current config (computed at solve time)
    instantaneous_mobility: int | null  // n_coords - constraint_rank (authoritative DOF at current config)
}
```

**Schema versioning:** Every saved mechanism file includes `schema_version`. When the data model changes, the version increments. The loader checks the version and applies migration transforms if needed. This prevents every model change from breaking saved files.

**Coordinate representation note:** The v1 solver uses absolute coordinates (3 per moving body) for all mechanisms. This is the right starting point for arbitrary topology. However, the assembly and output APIs should not hardcode the assumption that the state vector is always full absolute coordinates. Keep interfaces agnostic enough to support future alternative representations:

- **Reduced coordinates** for common mechanism classes (e.g., a 1-DOF 4-bar parameterized by a single input angle). Better conditioning, faster solves, simpler output interpretation, essential for optimization loops.
- **Linearized/modal models** around equilibrium configurations. Local stiffness, small-signal natural frequencies, sensitivity to dimension changes.
- **Hybrid formulations** where some bodies use absolute coordinates and others use relative (joint) coordinates.

Concretely, this means: solver functions should accept and return state through the `State` abstraction (see `core/state.py`), not by directly indexing raw `q` vectors with hardcoded `3*i` offsets outside of the State module. Force elements and constraints should query body pose through State methods, not by slicing `q` directly. This costs nothing now and preserves the option later.

**DOF fields:**
- `dof_grubler` is computed from body/joint counts using Grübler's formula. It is an informational sanity check — it can be wrong for mechanisms with redundant constraints or special geometric conditions.
- `constraint_rank` is the numerical rank of the assembled Jacobian `Φ_q` at a specific configuration. This is the real constraint count.
- `instantaneous_mobility` is `3 * n_moving_bodies - constraint_rank`. This is the authoritative mobility at that configuration. It may change at singular configurations (the mechanism may appear to gain or lose mobility at specific poses). It is labeled "instantaneous" to reflect this.

---

## Unit System

### Internal (all solvers, all storage, all JSON)

| Quantity | Unit |
|---|---|
| Length | m |
| Force | N |
| Torque | N·m |
| Mass | kg |
| Moment of inertia | kg·m² |
| Angle | rad |
| Time | s |
| Angular velocity | rad/s |
| Linear velocity | m/s |
| Acceleration | m/s² |

This is standard SI. `F = ma`, `T = Iα`, and `g = 9.81 m/s²` all work without conversion factors.

### GUI Display and User Input

| Quantity | Display unit | Conversion factor (input → internal) |
|---|---|---|
| Length | mm | × 1e-3 |
| Torque | N·mm | × 1e-3 |
| Angle | degrees | × π/180 |
| Force | N | (none) |
| Mass | kg | (none) |
| Inertia | kg·mm² | × 1e-6 |

The conversion layer lives in `util/units.py` and is called only at the GUI boundary. No solver code ever imports it.

### Why Not mm + N + kg?

Because it is not a consistent unit set. `1 N = 1 kg·m/s²`, not `1 kg·mm/s²`. If you compute `F = m * a` with `m` in kg and `a` in mm/s², you get a result that is off by a factor of 1000 unless you manually insert a conversion factor. This silently corrupts every dynamic equation: `F = ma`, `T = Iα`, gravity, inverse dynamics, forward dynamics. Using SI internally eliminates this entire class of bugs.

---

## Serialization

Mechanism files are JSON. The format is:

```json
{
    "schema_version": "1.0.0",
    "gravity_magnitude": 9.81,
    "gravity_angle": 0.0,
    "bodies": {
        "ground": {
            "attachment_points": {"O2": [0.0, 0.0], "O4": [0.038, 0.0]},
            "mass": 0.0, "cg_local": [0.0, 0.0], "Izz_cg": 0.0
        },
        "crank": {
            "attachment_points": {"A": [0.0, 0.0], "B": [0.010, 0.0]},
            "mass": 0.05, "cg_local": [0.005, 0.0], "Izz_cg": 4.2e-7,
            "coupler_points": {}
        }
    },
    "joints": {
        "J1": {
            "type": "revolute",
            "body_i": "ground", "point_i": "O2",
            "body_j": "crank", "point_j": "A",
            "driven": true,
            "driver": {"type": "constant_speed", "parameters": {"rate": 1.0472}}
        }
    },
    "force_elements": {},
    "point_masses": {}
}
```

All values are in SI units (meters, kg, radians, etc.). No unit annotations in the file — the unit system is defined by the schema version.

### Schema Migration

When loading a file:
1. Read `schema_version`
2. If version < current, apply migration functions in order (e.g., `migrate_1_0_to_1_1()`)
3. If version > current, reject with error (forward compatibility not guaranteed)
4. Validate the migrated data against the current schema

Migration functions are registered in a version-ordered list. Each migration is a pure function: `dict → dict`.

---

## Rust Portability

This data model is designed to port cleanly to Rust after Phase 4. Key mapping decisions are documented in `docs/RUST_MIGRATION.md`. During Python development, follow these conventions to keep the port mechanical:

- **Body, Mechanism, JointConstraint, ForceElement, PointMass** → Rust structs with `serde` derive macros
- **ForceElement.evaluate()** → `trait ForceElement { fn evaluate(&self, state: &State, t: f64) -> DVector<f64>; }`
- **Joint type enum** → Rust `enum JointType { Revolute, Prismatic, Fixed }`
- **JSON schema** is shared between Python and Rust — the same mechanism files load in both. This is a hard requirement
- **No Python-specific serialization** (pickle, shelve). JSON only
