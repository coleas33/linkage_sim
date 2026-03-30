# Validation

Two separate concerns: (1) validating that a user's mechanism is well-posed before solving, and (2) validating that the solver produces correct results against known benchmarks.

---

## Mechanism Validation Layers

Validation is layered, from cheap topology checks to expensive numerical analysis. Each layer catches different classes of errors.

### Layer 1: Topology Checks

**When:** run on every edit (mechanism definition change). Instant.

**Graph connectivity.** All bodies must be reachable from ground through joint constraints. If the mechanism has disconnected sub-mechanisms (a body not connected to ground through any chain of joints), flag it. The solver cannot handle disconnected subsystems.

**Grübler DOF count.** Compute:
```
M = 3 * (n_moving_bodies) - Σ (DOF removed by each joint)
```

Where revolute removes 2 DOF, prismatic removes 2 DOF, fixed removes 3 DOF.

This is an **informational sanity check**, not an authoritative result. Grübler can be wrong because:
- It does not account for redundant constraints (over-constrained mechanisms)
- It does not detect special geometric conditions that create unexpected mobility
- It does not verify that constraints are independent

Report the Grübler count alongside the target DOF. If `M ≠ expected DOF` (typically 1 for a single-input mechanism), flag it as a warning, not a hard error.

**Grashof condition (4-bar specific).** For mechanisms identified as 4-bars (4 bodies including ground, 4 revolute joints), compute:
```
s + l ≤ p + q
```
Where `s` = shortest link, `l` = longest link, `p` and `q` = other two. Report whether the mechanism is Grashof (shortest link can make full rotations) or non-Grashof (rocker-rocker only). This is informational.

**Attachment point existence.** Every joint references attachment points by name on two bodies. Verify that the named points exist on the referenced bodies. This catches typos and stale references.

### Layer 2: Constraint Analysis

**When:** run before solving, after mechanism definition is complete. Requires assembling the Jacobian at the initial configuration.

**Jacobian rank.** Assemble `Φ_q` at the initial configuration and compute its numerical rank (via SVD, with a reasonable tolerance, e.g., singular values below `1e-10 * max_singular_value` are treated as zero).

```
constraint_rank = rank(Φ_q)
instantaneous_mobility = 3 * n_moving_bodies - constraint_rank
```

Compare `instantaneous_mobility` to Grübler:
- If they agree: good, proceed
- If Grübler says 1 DOF but Jacobian says 0: mechanism may be over-constrained or at a singular configuration. The solver may fail or produce indeterminate reactions. Warn.
- If Grübler says 1 DOF but Jacobian says 2+: mechanism has special geometry creating extra mobility, or the initial configuration is at a bifurcation. Warn.
- If they disagree, Jacobian rank is closer to reality (at this configuration), but neither is the final word.

**Redundant constraint detection.** If `constraint_rank < n_constraint_equations`, some constraints are redundant. Identify which constraints are in the null space of `Φ_q^T` (which multipliers are indeterminate). Report to the user: "This mechanism has redundant constraints. Joint reactions at [affected joints] are not uniquely determined by rigid-body analysis. Actual load distribution depends on structural compliance."

**Driver independence.** Verify that the driver constraint row is not in the null space of `Φ_q`. If it is, the driver is redundant (trying to prescribe a coordinate that is already determined by other constraints). This is an error.

### Layer 3: Assembly Analysis

**When:** run at solve time, during kinematic position solving.

**Assembly feasibility.** At each requested input position, Newton-Raphson either converges or doesn't. If it doesn't converge within the iteration limit (e.g., 50 iterations), the configuration is not achievable — the mechanism cannot physically reach that input angle/displacement. Report the failure position.

**Proximity to singularity.** Monitor the smallest singular value of `Φ_q` during the sweep. When it drops below a threshold (relative to the largest singular value), the mechanism is approaching a singular configuration. Report: "Mechanism approaching singular configuration at input = X. Condition number = Y. Results near this position may be unreliable."

---

## Constraint and Loop Diagnostics

Pass/fail validation is not enough for debugging user mechanisms. When something goes wrong, the engineer needs to know *where* and *why*, not just "Newton failed." This diagnostic layer turns solver failures and warnings into actionable information.

### Constraint Residual Decomposition

When Newton-Raphson fails to converge (or converges slowly), report the constraint residual `Φ(q)` decomposed by source:

| Diagnostic | What it tells the engineer |
|-----------|--------------------------|
| Per-joint residual | Which joint has the largest closure error: "Revolute J7 on ternary link L3: residual = 2.3 mm" |
| Per-loop residual | Which kinematic loop is failing to close. Identify loops via graph cycle detection and report the total closure error per loop |
| Dominant constraint | Which constraint equation(s) contribute most to the total residual. Sort by `|Φ_i|` descending |
| Conflicting driver | Whether the driver constraint conflicts with passive geometry: "Driver prescribes input angle = 185°, but mechanism cannot reach this configuration. Closest feasible input ≈ 178°" |

### Rank Deficiency Localization

When `rank(Φ_q) < m`, identify *which* constraints are redundant:

1. Compute the SVD of `Φ_q`. Identify the right singular vectors associated with near-zero singular values.
2. Map those vectors back to specific constraint rows (joint IDs).
3. Report: "Constraints at joints J3 and J5 are nearly redundant — they constrain the same degree of freedom. Passive reactions at these joints are indeterminate."

This turns "rank deficient" into "joints J3 and J5 are the problem."

### Near-Singularity Localization

When `σ_min(Φ_q)` drops below the warning threshold, identify which bodies and joints are involved:

1. The left singular vector associated with `σ_min` identifies the constraint direction that is becoming degenerate.
2. Map this back to the joints involved: "Near-singular configuration driven by joints J2–J4 on bodies L2–L3 approaching collinear alignment."
3. Report the associated transmission angle (if applicable) and the input angle at which this occurs.

### Connectivity Diagnostics

Beyond simple "all bodies reachable from ground," report:

- **Articulation points:** bodies whose removal would disconnect the mechanism. These are structural weak points.
- **Bridge joints:** joints whose removal would disconnect the mechanism. Useful for understanding load paths.
- **Independent loops:** the number of independent kinematic loops (from graph cycle analysis). Cross-reference with Grübler for consistency.

### Integration with Solver Output

Diagnostics are not a separate mode — they are integrated into every solver run:

- On convergence failure: automatic residual decomposition in the error report
- On convergence with warnings: singularity localization appended to results
- On rank deficiency: redundancy localization appended to results
- On branch jump detection: report which bodies/joints exhibited the largest pose discontinuity

The goal: "Newton failed at input = 127°" becomes "Loop 2 closure error dominated by revolute J7 on ternary link L3. Mechanism is 1.2° past the toggle point at J4–J5. Try reducing step size or starting from a different assembly mode."

---

## Versioned Benchmark Suite

Every solver capability is validated against known analytical or published results. Benchmarks are stored in `data/benchmarks/` as JSON files with expected results. The test suite loads them and compares solver output to expected values within specified tolerances.

### Benchmark Format

```json
{
    "benchmark_id": "fourbar_grashof_crank_rocker",
    "description": "Grashof crank-rocker 4-bar, textbook example",
    "source": "Norton, Design of Machinery, 6th ed., Example 4.1",
    "mechanism_file": "benchmarks/fourbar_grashof.json",
    "tests": [
        {
            "test_id": "grubler_dof",
            "type": "validation",
            "expected": {"dof_grubler": 1}
        },
        {
            "test_id": "grashof_check",
            "type": "validation",
            "expected": {"grashof": true}
        },
        {
            "test_id": "position_at_90deg",
            "type": "kinematic_position",
            "input_angle_deg": 90.0,
            "expected_body_angles_deg": {"coupler": 42.37, "rocker": 68.12},
            "tolerance_deg": 0.1
        },
        {
            "test_id": "coupler_point_at_180deg",
            "type": "kinematic_position",
            "input_angle_deg": 180.0,
            "expected_coupler_point": {"body": "coupler", "point": "P", "x_mm": 28.4, "y_mm": 15.7},
            "tolerance_mm": 0.1
        },
        {
            "test_id": "velocity_at_90deg",
            "type": "kinematic_velocity",
            "input_angle_deg": 90.0,
            "input_speed_rpm": 60.0,
            "expected_rocker_omega_rad_s": 3.14,
            "tolerance_rad_s": 0.05
        },
        {
            "test_id": "input_torque_with_gravity",
            "type": "static_force",
            "input_angle_deg": 90.0,
            "expected_input_torque_Nm": 0.245,
            "tolerance_Nm": 0.005
        }
    ]
}
```

### Planned Benchmark Mechanisms

#### Phase 1 — Kinematics

| ID | Mechanism | What it validates |
|---|---|---|
| `fourbar_grashof_cr` | Grashof crank-rocker | Basic revolute-only kinematics, Grashof check = true, full rotation of input |
| `fourbar_non_grashof` | Non-Grashof rocker-rocker | Grashof check = false, limited input range, toggle position detection |
| `slider_crank_inline` | Inline slider-crank | Prismatic joint kinematics, piston displacement = textbook formula |
| `slider_crank_offset` | Offset slider-crank | Offset prismatic joint, asymmetric motion |
| `sixbar_watt` | Watt I 6-bar with ternary link | Ternary body with 3 attachment points, validates body–constraint model |
| `sixbar_stephenson` | Stephenson III 6-bar | Different ternary link configuration |
| `symmetric_toggle` | Toggle clamp at dead point | Jacobian singularity detection, transmission angle = 0° |
| `redundant_constraints` | Parallelogram 4-bar | Grübler says 1 DOF, but has 1 redundant constraint. Jacobian rank should detect it |

#### Phase 1 — Kinematics (Robustness / Failure Cases)

| ID | Mechanism | What it validates |
|---|---|---|
| `fourbar_near_toggle` | Grashof 4-bar swept through near-toggle configuration | Branch manager halves step size, does not silently jump branches. Solver reports proximity to singularity with `σ_min` value |
| `fourbar_branch_jump` | 4-bar with step size large enough to risk branch jump at known angle | Orientation invariant tracking detects the jump and triggers step-size reduction. Solver stays on correct branch |
| `parallelogram_redundant` | Parallelogram 4-bar (1 redundant constraint) | Rank deficiency localization identifies the redundant constraint pair. Grübler and Jacobian rank disagree correctly |
| `near_disconnected` | Mechanism with one joint barely connecting two sub-chains | Connectivity diagnostics flag the articulation point. Solver warns about poor conditioning at the weak connection |
| `prismatic_awkward_axis` | Slider-crank with slide axis at 45° on a moving body | Prismatic constraint produces correct motion along the angled axis. No sign convention errors. Reaction force decomposition matches hand calculation |

#### Phase 2 — Statics

| ID | Mechanism | What it validates |
|---|---|---|
| `fourbar_gravity` | 4-bar with gravity only | Input torque = sum of gravity torques, verifiable by hand |
| `fourbar_spring` | 4-bar with one linear spring | Spring force computation, static equilibrium with spring + gravity |
| `fourbar_torsion_spring` | 4-bar with torsion spring at input | Torsion spring torque vs. angle, input torque offset |
| `slider_crank_friction` | Slider-crank with friction at slider | Coulomb friction at prismatic joint, regularized model |
| `point_mass_test` | 4-bar with point mass on coupler | Composite CG and Izz computation, gravity torque change |

#### Phase 3 — Inverse Dynamics

| ID | Mechanism | What it validates |
|---|---|---|
| `fourbar_inertia` | 4-bar with link masses, no springs | Inverse dynamics torque = inertial terms only, compare to simple pendulum limit cases |
| `slider_crank_motor` | Slider-crank with motor droop model | Motor operating point extraction, T-ω feasibility check |
| `fourbar_damped` | 4-bar with rotary damper | Damping torque contribution, energy dissipation rate |

#### Phase 4 — Forward Dynamics

| ID | Mechanism | What it validates |
|---|---|---|
| `pendulum_single` | Single link (ground + body + revolute), gravity | Simple pendulum: known analytical solution for small angles, period check |
| `pendulum_damped` | Single link with rotary damper | Damped oscillation: exponential decay envelope, correct damping ratio |
| `fourbar_free_response` | 4-bar with springs and dampers, released from displaced position | Transient oscillation, energy balance closure, constraint drift monitoring |
| `fourbar_step_torque` | 4-bar with step torque input | Transient response, steady-state reached, reaction forces bounded |

### Running Benchmarks

```bash
python -m pytest tests/ -k benchmark --benchmark-report
```

Each benchmark test:
1. Loads the mechanism from JSON
2. Runs the specified analysis mode
3. Compares outputs to expected values within tolerance
4. Reports pass/fail with actual vs. expected values

### Adding New Benchmarks

1. Create the mechanism JSON file in `data/benchmarks/`
2. Create the benchmark specification JSON with expected results
3. Document the source (textbook, analytical derivation, or cross-validated with commercial software)
4. Add the benchmark ID to the appropriate phase table in this document
5. Expected values in benchmark files use display units (mm, degrees) with explicit unit labels. The test harness converts to SI before comparison.

### Tolerance Philosophy

Tolerances should be tight enough to catch real bugs but loose enough to accommodate:
- Floating-point arithmetic differences across platforms
- Slightly different Newton-Raphson convergence paths
- Interpolation differences in tabulated data

Typical tolerances:
- Position: 0.01 mm (1e-5 m)
- Angle: 0.01° (1.7e-4 rad)
- Velocity: 0.1% relative
- Force: 0.1% relative or 0.01 N absolute, whichever is larger
- Torque: 0.1% relative or 0.001 N·m absolute, whichever is larger
- Energy balance: 0.01% of total energy
