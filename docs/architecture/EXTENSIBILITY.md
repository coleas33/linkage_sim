# Extensibility

How to add new components without breaking existing solvers, and an honest assessment of what is easy vs. hard.

---

## Three Extension Points

Every future component maps to one of these categories. The categories differ in how much solver involvement is needed.

### Extension Point 1: New ForceElement Types

**Solver changes required: none.**

Any element whose behavior can be expressed as `evaluate(state, t) → Q_contribution` (a contribution to the generalized force vector) is a new ForceElement subclass. The solver already iterates over all force elements and sums their `Q` contributions. A new type is invisible to the solver.

**To add a new ForceElement type:**

1. Implement a class with `evaluate(state, t)` that uses the helper utilities (`point_force_to_Q`, `body_torque_to_Q`, etc.) to produce its generalized force contribution.
2. Define the parameter schema (name, type, unit, default, bounds) for GUI auto-generation and JSON serialization.
3. Register the type name in the force element registry.
4. Add a benchmark test with a known analytical result.

**What this covers:**
- Nonlinear springs (Belleville disc springs, rubber bushings, constant-force springs)
- Pneumatic cylinders (force = pressure × area, pressure coupled to enclosed volume)
- Bumpers and elastomer stops (steep nonlinear stiffness at joint limits)
- Magnetic springs (inverse-square or custom force-distance)
- Any smooth, state-dependent force or torque

### Extension Point 2: New JointConstraint Types

**Solver changes required: minimal (assembly only).**

Elements that enforce a new kinematic relationship between bodies. Each new type provides:
- `Phi(q, t)` — its constraint equations (residual vector)
- `Phi_q(q, t)` — its Jacobian rows (partial derivatives w.r.t. generalized coordinates)
- `gamma(q, q_dot, t)` — its contribution to the acceleration right-hand side

The solver's assembly code collects these from all joints and stacks them into the global system. The assembly logic is generic — it does not know what the constraint represents. The solution code (`solve()`, `newton_raphson()`, etc.) does not change.

**To add a new JointConstraint type:**

1. Implement the constraint equations `Phi(q)` and Jacobian `Phi_q(q)`.
2. Implement the `gamma` contribution for acceleration analysis.
3. Define the DOF count removed (for Grübler calculation).
4. Define the parameter schema and attachment geometry.
5. Register the type name.
6. Add benchmark tests verifying the constraint is satisfied and the Jacobian is correct (compare analytical Jacobian to finite-difference numerical Jacobian).

**What this covers:**
- Gear pair (fixed angular velocity ratio between two bodies at a shared joint)
- Rack-and-pinion (rotary-to-linear coupling)
- Cam profile (follower displacement as a function of cam angle — a time-varying or configuration-dependent constraint)
- Screw joint (coupled rotation and translation)

### Extension Point 3: Conditional Constraints and Topology-Switching Elements

**Solver changes required: potentially significant.**

Elements that change the active set of constraints based on the current state. These are fundamentally different from smooth force elements or fixed constraints because they change the mathematical structure of the problem at runtime.

This includes:
- **One-way clutch / overrunning bearing** — transmits torque in one direction, freewheels in the other. The constraint is active or inactive depending on relative velocity (or torque sign).
- **Cables and chains** — tension-only elements that go slack when compressed. Effectively a unilateral constraint: the force element is active when the cable is taut and inactive when slack. If modeled as a constraint, it switches between "constraint present" and "constraint absent."
- **Hard stops with restitution** — when a joint reaches its limit, an impulsive reaction is applied and the velocity reverses. This is an event that interrupts the integration.
- **Frictional stick-slip** — static friction is a constraint (zero relative velocity), kinetic friction is a force. The system switches between these modes depending on the applied load vs. the friction capacity.

**Why these are harder:**

These components change the numerical class of the problem:
- Smooth ODE/DAE → nonsmooth system
- Bilateral constraints → unilateral constraints (inequalities)
- Fixed topology → mode-switching topology
- Continuous dynamics → event-driven dynamics

**What may be needed:**
- Event detection (zero-crossing functions in the integrator)
- Complementarity solvers (for unilateral constraints)
- Mode logic (tracking which constraints are active)
- Nonsmooth mechanics methods (Moreau time-stepping, etc.)

**Honest assessment:** implementing these well is a research-level problem in computational mechanics. For a practical engineering tool, the simplest viable approaches are:

- **Cables:** model as a spring with very low compression stiffness (near-zero force when compressed, normal spring force when in tension). This avoids topology switching entirely. It's a hack, but it works for most practical cases.
- **One-way clutch:** model as a rotary damper with direction-dependent coefficient (high damping in freewheel direction to dissipate energy, normal transmission in locked direction). Or use event detection to switch between "joint present" and "joint absent."
- **Hard stops:** penalty method (stiff spring at limit) for forward dynamics. For kinematics/statics, simply clamp the joint coordinate at the limit.
- **Stick-slip friction:** velocity regularization (already in the v1 friction model) is adequate for most engineering purposes. True stick-slip is needed only for precision mechanism analysis.

---

## Design Rules

These rules ensure that extension points remain clean as the codebase grows.

### Rule 1: Solvers Never Switch on Component Type

No `if isinstance(element, Spring)` or `if element.type == "gear"` in any solver code. Force elements go through `evaluate()`. Constraints go through `Phi()`, `Phi_q()`, and `gamma()`. The solver sees only the interface.

**Why:** if you add type-switching to the solver, every new component requires editing the solver, and the solver becomes a sprawling switch statement that's hard to test and easy to break.

### Rule 2: Mechanism Holds Generic Collections

The `Mechanism` class stores `{id: JointConstraint}`, `{id: ForceElement}`, etc. It iterates them generically. It does not have `self.springs`, `self.dampers`, `self.motors` as separate collections. (The GUI may filter by type for display purposes, but the data model does not separate them.)

### Rule 3: GUI Panels Are Auto-Generated from Schemas

Each element type declares a parameter schema:

```python
SCHEMA = {
    "k": {"type": "float", "unit": "N/m", "display_unit": "N/mm", "default": 100.0, "min": 0.0},
    "free_length": {"type": "float", "unit": "m", "display_unit": "mm", "default": 0.05, "min": 0.0},
    "preload": {"type": "float", "unit": "N", "display_unit": "N", "default": 0.0},
    "mode": {"type": "enum", "options": ["both", "tension_only", "compression_only"], "default": "both"}
}
```

The property panel renders input fields from this schema. Adding a new type to the GUI means registering its schema — no new UI code for basic parameter editing.

### Rule 4: Serialization Is Generic

JSON save/load uses `type` + `parameters` dict. A new element type is automatically serializable if it follows the schema convention. No per-type serialization code.

### Rule 5: Constraints Declare Their Own Math

Every JointConstraint type must implement:
- `constraint(q, t) → residual_vector`
- `jacobian(q, t) → jacobian_matrix_rows`
- `gamma(q, q_dot, t) → acceleration_rhs_contribution`
- `n_equations` → number of constraint rows

The solver assembles these into the global system without knowing what the constraint represents. If a new constraint type provides these four things, it works with all existing solvers.

### Rule 6: Jacobian Correctness Is Testable

Every constraint type must pass a finite-difference Jacobian check:

```python
def test_jacobian_correctness(constraint, q0, epsilon=1e-7):
    analytical = constraint.jacobian(q0)
    numerical = finite_difference_jacobian(constraint.constraint, q0, epsilon)
    assert np.allclose(analytical, numerical, atol=1e-5)
```

This test is mandatory for every new constraint type. Analytical Jacobian errors are the most common implementation bug and the hardest to diagnose from solver failures.

---

## Future Components Catalog

Not in scope for any current phase. Documented so the architecture accounts for them.

| Component | Extension Point | Implementation Notes |
|---|---|---|
| **Cable / belt / chain** | Conditional (or ForceElement hack) | Tension-only. Simplest approach: linear spring with near-zero compression stiffness. Proper approach: unilateral constraint with slack detection |
| **Pulley** | Topology modifier + ForceElement | Changes cable force direction. Can be modeled as a constraint that redirects a cable element's force. Complex if cable wraps partially around the pulley |
| **Gear pair** | JointConstraint | Fixed ratio: `θ_j - r * θ_i = const`. One constraint equation, straightforward Jacobian |
| **Rack-and-pinion** | JointConstraint | `x_rack = r * θ_pinion`. One constraint equation coupling translation to rotation |
| **One-way clutch** | Conditional constraint | Transmits torque in one direction. Simplest: direction-dependent damper. Proper: event detection + mode switch |
| **Pneumatic cylinder** | ForceElement | `F = P(V) * A`. Volume `V` is coupled to stroke. Polytropic gas law: `P * V^n = const`. This is a state-dependent force element, no solver changes needed |
| **Hydraulic cylinder (flow-controlled)** | ForceElement or Constraint | If velocity-controlled: may need to be a driver-like constraint rather than a force element. If pressure-controlled: ForceElement |
| **Nonlinear spring (Belleville)** | ForceElement | Lookup table or analytical expression for force-deflection. Straightforward |
| **Constant-force spring** | ForceElement | Nearly flat force over stroke range. Simple special case of lookup-table spring |
| **Bumper / elastomer stop** | ForceElement | Steep nonlinear spring activated at joint limits: `F = k_bump * max(0, penetration)^n`. Smoother than rigid contact for dynamics |
| **Cam profile** | JointConstraint | `displacement = f(cam_angle)` — one configuration-dependent constraint equation. The cam profile `f` is a tabulated or analytical function |
| **Magnetic spring** | ForceElement | Inverse-square or custom force-distance. Straightforward |

---

## Coordinate Representation Extensions

The v1 solver uses absolute coordinates (3 per moving body). Future phases may benefit from alternative representations without changing the data model or the extension point interfaces.

| Representation | When useful | What changes |
|---------------|------------|-------------|
| **Reduced coordinates** | 1-DOF mechanisms in optimization loops. Parameterize by input angle only — eliminate Newton-Raphson entirely for the kinematic solve | New solver path in `solvers/kinematics.py` that uses implicit function theorem to express dependent coordinates as functions of the independent one. Assembly and force element APIs unchanged — they still receive full body poses via State |
| **Linearized model** | Natural frequency estimation, local sensitivity, small-signal stability | New analysis module (`analysis/linearization.py`). Compute tangent stiffness and mass matrices at an equilibrium, extract eigenvalues. Does not replace the nonlinear solver — supplements it |
| **Joint coordinates** | Mechanisms where joint angles are more natural than Cartesian body poses | Alternative State implementation that stores joint coordinates and reconstructs body poses on demand. Constraint and force element interfaces unchanged if they query State, not raw `q` |

**Design protection (v1):** The v1 code should access body poses through `State` methods (e.g., `state.get_pose(body_id)`, `state.get_point_global(body_id, local_point)`) rather than raw index arithmetic outside of the State module. This costs nothing and preserves the option to swap in alternative coordinate representations later. See `docs/ARCHITECTURE.md` § "Coordinate representation note."

---

## Additional Analysis Capabilities Worth Considering

These are not component types but analysis features that extend the tool's usefulness:

- **Sensitivity analysis** — partial derivatives of outputs w.r.t. design parameters. Critical for tolerance analysis and optimization.
- **Backlash and clearance** — joint clearance as a small free-play zone with contact model. Important for precision mechanisms.
- **Thermal effects** — body dimension changes with temperature. Relevant for high-temperature applications.
- **Elastic bodies (compliant mechanisms)** — model bodies as beams with finite stiffness. Opens flexure design.
- **Fatigue life estimation** — from reaction force histories, estimate bearing/pin life.
- **3D visualization** — extrude planar mechanism for presentation and interference checking.
- **Hardware interface** — read encoder positions from a physical prototype, overlay on simulation.
- **Script/macro mode** — expose solver as Python API for batch analyses and Monte Carlo studies.
- **Collision detection** — flag when bodies physically interfere during motion.
- **Counterbalance design** — compute optimal spring or point mass to minimize input torque variation.
- **Vibration isolation** — analyze transmitted forces vs. damping for mechanisms on compliant mounts.
