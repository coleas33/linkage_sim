# Analysis Modes

All analysis modes share the same mathematical backbone: generalized coordinates `q`, constraint equations `Φ(q,t) = 0`, and the constraint Jacobian `Φ_q`. Each mode adds complexity incrementally. See `docs/NUMERICAL_FORMULATION.md` for exact equations.

---

## Mode 1 — Kinematic Analysis

**Purpose:** Given the input joint coordinate (angle or stroke) as a function of time or swept over a range, compute all body poses, velocities, and accelerations. No forces involved.

**Inputs:**
- Mechanism definition (bodies, joints, driver)
- Input sweep range (e.g., 0°–360° at 1° steps) or time-parameterized driver

**Solver steps:**

1. **Position** — at each input value, solve `Φ(q, t) = 0` using Newton-Raphson with the analytical Jacobian `Φ_q`. Initial guess from previous converged position (continuation).

2. **Velocity** — solve the linear system `Φ_q * q̇ = -Φ_t`. Exact, no iteration. `Φ_t` is nonzero only for driven joints.

3. **Acceleration** — solve `Φ_q * q̈ = γ`. Exact, no iteration. The right-hand side `γ` is assembled from velocity-dependent terms (centripetal, Coriolis-like contributions from the constraints).

4. **Coupler point evaluation** — for each named coupler point on each body, compute global position, velocity, and acceleration using the body's solved pose and angular velocity/acceleration.

**Outputs:**
- All body poses `(x, y, θ)` vs. input angle/time
- All body velocities `(ẋ, ẏ, θ̇)` and accelerations `(ẍ, ÿ, θ̈)`
- Coupler point trajectories (curves), velocities, accelerations
- Joint-space quantities: relative angles and displacements at each joint

**Failure modes:**
- Newton-Raphson does not converge → the requested configuration is not achievable (mechanism cannot reach that input angle), or the solver is near a toggle/bifurcation. The branch manager automatically halves step size and retries before reporting failure. See `docs/NUMERICAL_FORMULATION.md` § "Branch Management."
- Jacobian is singular → the mechanism is at a singular configuration (toggle point, dead point). The solver flags the configuration via `σ_min` monitoring, reduces step size, and attempts to approach from both sides.
- Silent branch jump → the solver converges but to a different assembly mode. Detected by orientation invariant tracking (joint-angle sign changes, loop cross-product sign flips, pose discontinuity exceeding step-size-proportional threshold). On detection, the solver halves step size and re-solves from last good state.

---

## Mode 2 — Static Force Analysis

**Purpose:** At each configuration from kinematic analysis, compute the force/torque balance. Determine the required input effort and all joint reaction forces, accounting for all applied loads but not inertia.

**Inputs:**
- Solved kinematics from Mode 1
- All force elements with their parameters
- Gravity settings (magnitude and direction)
- Point masses attached to bodies

**Solver steps:**

1. **Composite mass property update** — for each body, recompute mass, CG, and Izz incorporating all attached point masses (parallel axis theorem).

2. **Generalized force assembly** — evaluate each force element at the current state, accumulate contributions into the generalized force vector `Q`.

3. **Equilibrium solve** — solve `Φ_q^T * λ = -Q` for the Lagrange multipliers `λ`.

4. **Reaction extraction** — map `λ` back to physical forces at each joint (global Fx, Fy, resultant, local components). Extract the driver's `λ` as the required input torque/force.

**Separating passive reactions from driver effort:**

The Lagrange multiplier vector `λ` contains entries for every constraint row. The structure is:

```
λ = [λ_joint1_x, λ_joint1_y, λ_joint2_x, λ_joint2_y, ..., λ_driver]
```

The driver's entry is at a known position (the last constraint rows added during assembly, by convention). This is the required input effort. All other entries are passive joint reactions.

**Redundant constraint handling:** If `rank(Φ_q) < m` (number of constraint equations), the system `Φ_q^T * λ = -Q` is underdetermined. Use minimum-norm pseudoinverse solution `λ = Φ_q^T⁺ * (-Q)`. Warn the user that passive reactions in redundant directions are not unique. The driver reaction is still unique if the driver constraint is independent.

**Singularity and conditioning:** At every configuration, the static solver reports `σ_min`, `κ(Φ_q)`, and driver mechanical advantage. Near singularities, reaction magnitudes blow up — the solver flags these configurations as unreliable rather than silently reporting large numbers as if they were trustworthy. See `docs/NUMERICAL_FORMULATION.md` § "Singularity Analysis and Conditioning" for thresholds and null-space reporting.

**Outputs:**
- Required input torque or force vs. position
- Joint reaction forces at every joint (global, local, radial/tangential — see `docs/ENGINEERING_OUTPUTS.md`)
- Individual force element states (spring force, damper force, etc.) vs. position

---

## Mode 3 — Inverse Dynamics

**Purpose:** Given a prescribed motion profile `q(t)` (and thus `q̇(t)`, `q̈(t)`) from the kinematic solver, compute the required actuator effort and all joint forces including inertial loads.

**Inputs:**
- Solved kinematics from Mode 1 (position, velocity, acceleration at each timestep)
- Mass properties for all bodies (including point masses)
- All force elements
- Gravity

**Solver steps:**

1. **Mass matrix assembly** — build the block-diagonal mass matrix `M` from composite body mass properties.

2. **Generalized force assembly** — evaluate all force elements at the current state → `Q`.

3. **Inverse dynamics solve** — at each timestep:
   ```
   Φ_q^T * λ = Q - M * q̈
   ```
   This is the same linear system as statics, with the right-hand side modified by the inertial term `M * q̈`. The D'Alembert principle: inertial forces are fictitious loads subtracted from the applied loads.

4. **Reaction extraction** — identical to static analysis.

**Key difference from statics:** the inertial terms `M * q̈` can dominate at high speeds or for heavy links. A mechanism that requires almost no input torque in static analysis may require substantial torque when driven dynamically, especially during acceleration/deceleration phases.

**Motor feasibility check:** overlay the required torque-speed operating points `(ω_driver(t), T_required(t))` on the motor's T-ω envelope. If any operating point falls outside the envelope, the motor cannot follow the demanded profile. Report: max required torque, max required speed, and all infeasible timesteps.

**Outputs:**
- Required input torque/force vs. time
- All joint reaction forces vs. time (same formats as statics)
- Motor operating point trajectory on T-ω diagram
- Individual force element contributions vs. time (to understand what dominates: inertia, gravity, springs, friction)

---

## Mode 4 — Forward Dynamics

**Purpose:** Given applied forces/torques (no prescribed motion), integrate the equations of motion to find what the mechanism actually does over time.

**This is fundamentally different from Modes 1–3.** In forward dynamics, the motion is unknown — it is the solution. The mechanism's response depends on its mass, stiffness, damping, and the applied loads.

**Inputs:**
- Mechanism definition with mass properties
- All force elements (springs, dampers, motors, gravity, etc.)
- Applied torques/forces at actuated joints (or as force elements)
- Initial conditions: `q(0)` and `q̇(0)` (starting configuration and velocities)

**Mathematical formulation:**

```
M * q̈ + Φ_q^T * λ = Q(q, q̇, t)     (equations of motion)
Φ(q, t) = 0                            (constraint equations)
```

This is a differential-algebraic equation (DAE), not a plain ODE. The constraints `Φ = 0` are algebraic, not differential. See `docs/NUMERICAL_FORMULATION.md` for solution approaches.

**Critical requirements:**

- **Damping is essential.** Without energy dissipation, the simulation rings indefinitely at natural frequencies. If no damping elements are present, the solver should warn. Even small viscous damping (rotary or translational) stabilizes the response.
- **Constraint drift monitoring.** After each integration step, check `‖Φ(q)‖`. If it exceeds tolerance, apply correction (Baumgarte stabilization or projection).
- **Energy balance.** Track kinetic energy, potential energy (springs, gravity), and dissipated energy (dampers, friction) at every timestep. If `KE + PE + dissipated ≠ constant + work_input` (within integrator tolerance), the simulation has a problem.

**Contact and hard stops at joint limits:**

When a joint coordinate hits a mechanical limit, the system's behavior changes discontinuously. Options:

- **Penalty method:** model the stop as a very stiff spring that activates at the limit. Simple to implement but introduces stiffness that may require implicit integration.
- **Event detection:** detect the exact moment the limit is reached (zero-crossing of `q_joint - limit`), apply an impulsive reaction (coefficient of restitution), and restart integration. More accurate but more complex.

**Integration approach (keep loose for now):**

Start with index reduction + Baumgarte stabilization, which is the simplest to implement. Use `scipy.integrate.solve_ivp` with an implicit method (Radau or BDF) for stiff systems. If constraint drift is unacceptable, add periodic projection. Consider coordinate partitioning or a dedicated DAE solver if the simple approach fails.

Do not lock the implementation to a specific library or method before testing with real mechanisms.

**Outputs:**
- Position, velocity, acceleration of all bodies and coupler points vs. time
- Joint reaction forces vs. time
- Force element states vs. time
- Energy components (KE, PE, dissipated, work input) vs. time
- Phase portraits (velocity vs. position for any coordinate)
- Natural frequency estimation (optional): linearize about equilibrium, extract eigenvalues from the linearized constrained system

---

## Cross-Cutting Concerns

### Assembly Mode and Branch Management

Assembly mode is managed by predictor-corrector continuation with branch-jump detection. See `docs/NUMERICAL_FORMULATION.md` § "Branch Management and Assembly Mode Tracking" for the full specification.

Key behaviors:

- The user provides an initial guess that selects the desired assembly configuration (e.g., "open" vs. "crossed" 4-bar)
- During a sweep, a **linear predictor** from the two most recent solutions provides the initial guess (not just raw previous-state seeding)
- **Orientation invariants** (joint-angle signs, loop cross-products) are tracked to detect silent branch jumps after convergence
- **Step size adapts** based on Newton-Raphson iteration count: increases when convergence is easy, halves when it struggles, halves again on detected branch jumps
- If convergence fails at minimum step size, the solver reports the failure location and the last good configuration

Full automatic branch enumeration and mode switching are deferred.

### Conditioning and Singularity

All modes depend on the Jacobian `Φ_q`. Near singular configurations:

- Kinematic solver may fail to converge or produce large corrections
- Velocity and acceleration solutions may have large magnitudes (poor conditioning)
- Static and inverse dynamic reactions may become very large (force transmission deteriorating)

See `docs/NUMERICAL_FORMULATION.md` § "Singularity Analysis and Conditioning" for the full framework, which distinguishes three separate questions:

1. **Existence** — can equilibrium be achieved at this configuration?
2. **Uniqueness** — are the Lagrange multipliers (reactions) uniquely determined, or do redundant constraints create indeterminate directions?
3. **Conditioning** — even if unique, are the reactions numerically reliable, or does ill-conditioning amplify errors?

The solver reports at every configuration:
- Smallest singular value `σ_min` of `Φ_q`
- Condition number `κ(Φ_q)`
- Driver mechanical advantage (approaching infinity at toggle, zero at dead center)
- Null-space directions when rank-deficient (identifies which reaction directions are indeterminate)

Warning thresholds are configurable. This is closely related to transmission angle — singularity ↔ transmission angle → 0° or 180°. Both metrics are reported together so the engineer sees the geometric interpretation alongside the numerical reliability indicator.

### Analysis Mode Dependencies

```
Mode 1 (Kinematics) ← standalone, no dependencies
Mode 2 (Statics)    ← requires Mode 1 results (positions)
Mode 3 (Inv. Dyn.)  ← requires Mode 1 results (positions, velocities, accelerations)
Mode 4 (Fwd. Dyn.)  ← standalone (computes its own kinematics during integration)
```

Mode 4 does not depend on Modes 1–3. It solves for motion directly. However, Modes 1–3 results are useful for: validating Mode 4 results, providing initial conditions, and understanding the mechanism before running dynamics.
