# Linkage equations reference

**Date:** 2026-04-29
**Purpose:** Authoritative mathematical reference for the closure equations, kinematic solvers, and statics used in this simulator. Built up from the actual source code so each claim cites a `file.rs:line`. Used as the verification baseline for the *position-control / trajectory* feature work — every later inverse-kinematic equation is a direct extension of what is documented here.

This is a living math reference, not a spec. The position-control design spec will be a separate document that references this one.

---

## 1. Notation and coordinates

The simulator is **planar (2D)**. Each non-ground body $i$ has three generalized coordinates:

$$
q_i = \begin{bmatrix} x_i \\ y_i \\ \theta_i \end{bmatrix}, \qquad q = \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_n \end{bmatrix} \in \mathbb{R}^{3n}
$$

Where $(x_i, y_i)$ is the body frame's origin in world coordinates and $\theta_i$ is its rotation. **Ground is body 0** with fixed pose $(0,0,0)$ and contributes **no entries to $q$** (`solver/assembly.rs:117` skips ground in mass assembly; jacobian skips per-constraint via `is_ground()` checks).

A body-local point $\mathbf{s}$ on body $i$ maps to world coordinates as:

$$
P_i(\mathbf{s}) = \mathbf{r}_i + A(\theta_i)\, \mathbf{s},
\qquad
A(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

The angular derivative of $A$ shows up everywhere:

$$
B(\theta) \equiv \frac{dA}{d\theta} = \begin{bmatrix} -\sin\theta & -\cos\theta \\ \cos\theta & -\sin\theta \end{bmatrix} = R_{\pi/2}\, A(\theta)
$$

So $\partial P_i / \partial \mathbf{r}_i = I_2$ and $\partial P_i / \partial \theta_i = B(\theta_i)\, \mathbf{s}$. Implementations: `State::rotation_matrix`, `State::rotation_matrix_derivative`, `State::body_point_global`, `State::body_point_global_derivative`.

The total **degrees of freedom** of the assembled mechanism is $\mathrm{DOF} = 3n - m$, where $m$ is the number of constraint equations. A 1-driver linkage in working condition has $\mathrm{DOF} = 0$ (the driver constraint adds the m-th equation that makes the system square).

### Figure 1 — Body coordinates

<svg width="500" height="240" xmlns="http://www.w3.org/2000/svg" font-family="serif">
  <defs>
    <marker id="ah1" viewBox="0 0 10 10" refX="9" refY="3" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 9 3 L 0 6 z" fill="#222"/>
    </marker>
    <marker id="ah1g" viewBox="0 0 10 10" refX="9" refY="3" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 9 3 L 0 6 z" fill="#127a3e"/>
    </marker>
    <marker id="ah1b" viewBox="0 0 10 10" refX="9" refY="3" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 9 3 L 0 6 z" fill="#1f5fa0"/>
    </marker>
  </defs>
  <!-- World axes -->
  <line x1="40" y1="200" x2="120" y2="200" stroke="#222" stroke-width="1.5" marker-end="url(#ah1)"/>
  <line x1="40" y1="200" x2="40" y2="120" stroke="#222" stroke-width="1.5" marker-end="url(#ah1)"/>
  <text x="125" y="205" font-size="13" font-style="italic">x</text>
  <text x="30" y="115" font-size="13" font-style="italic">y</text>
  <text x="0" y="220" font-size="10" fill="#666">world</text>
  <!-- r_i vector -->
  <line x1="40" y1="200" x2="240" y2="130" stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="120" y="180" font-size="13" font-style="italic" fill="#444">r_i = (x_i, y_i)</text>
  <!-- Body origin -->
  <circle cx="240" cy="130" r="3.5" fill="#222"/>
  <text x="245" y="148" font-size="10" fill="#555">body i origin</text>
  <!-- Body local axes (rotated CCW ~25°) -->
  <line x1="240" y1="130" x2="328" y2="89" stroke="#1f5fa0" stroke-width="1.7" marker-end="url(#ah1b)"/>
  <line x1="240" y1="130" x2="199" y2="42" stroke="#1f5fa0" stroke-width="1.7" marker-end="url(#ah1b)"/>
  <text x="335" y="92" font-size="12" font-style="italic" fill="#1f5fa0">x_local</text>
  <text x="172" y="40" font-size="12" font-style="italic" fill="#1f5fa0">y_local</text>
  <!-- θ_i arc -->
  <line x1="240" y1="130" x2="305" y2="130" stroke="#aaa" stroke-width="0.8" stroke-dasharray="2,2"/>
  <path d="M 290 130 A 50 50 0 0 0 285 110" fill="none" stroke="#222" stroke-width="1"/>
  <text x="288" y="124" font-size="13" font-style="italic">θ_i</text>
  <!-- Local point s and P_i(s) -->
  <line x1="240" y1="130" x2="320" y2="98" stroke="#127a3e" stroke-width="1.6" marker-end="url(#ah1g)"/>
  <circle cx="320" cy="98" r="3.5" fill="#127a3e"/>
  <text x="265" y="118" font-size="13" font-style="italic" fill="#127a3e">s</text>
  <text x="328" y="98" font-size="13" fill="#127a3e">P_i(s) = r_i + A(θ_i) s</text>
  <!-- Title strip -->
  <text x="20" y="20" font-size="13" font-weight="bold">Body coordinates and local→world transform (§1)</text>
  <text x="20" y="36" font-size="11" fill="#555">Body i has q_i = (x_i, y_i, θ_i). Local point s ∈ R² → world point P_i(s).</text>
</svg>

---

## 2. Constraint catalog

Every constraint type implements the same trait at `core/constraint/trait_def.rs:14-25`:

```rust
trait Constraint {
    fn constraint(&self, state, q, t) -> DVector<f64>;   // Φ
    fn phi_t    (&self, state, q, t) -> DVector<f64>;    // ∂Φ/∂t (q held fixed)
    fn jacobian (&self, state, q, t) -> DMatrix<f64>;    // Φ_q = ∂Φ/∂q
    fn gamma    (&self, state, q, q_dot, t) -> DVector<f64>;  // accel RHS
    fn n_equations(&self) -> usize;
}
```

The four functions are exactly what the position-, velocity-, and acceleration-level solvers consume, in that order. Below, each constraint is listed with all four contributions.

### 2.1 Revolute joint (2 eqs)
*Two body-local points are coincident.* `core/constraint/revolute.rs:51-95`

$$
\Phi_{\text{rev}} = \big[\mathbf{r}_i + A(\theta_i)\mathbf{s}_i\big] - \big[\mathbf{r}_j + A(\theta_j)\mathbf{s}_j\big] = \mathbf{0} \in \mathbb{R}^2
$$

- $\Phi_t = \mathbf{0}$ — geometric, no explicit time dependence.
- Jacobian: identity blocks at $\mathbf{r}_i, \mathbf{r}_j$ (with sign), and $\pm B(\theta_*)\mathbf{s}_*$ in the $\theta$ columns. Built by `translational_jacobian_block` (helper in `core/constraint/helpers.rs`).
- $\gamma = \dot{\theta}_i^2\, A(\theta_i)\mathbf{s}_i - \dot{\theta}_j^2\, A(\theta_j)\mathbf{s}_j$ — the "centripetal" terms from $\ddot{P}_i - \ddot{P}_j = 0$. Built by `translational_gamma`.

### 2.2 Fixed joint (3 eqs)
*Coincident points + locked relative angle.* `core/constraint/fixed.rs:56-118`

$$
\Phi_{\text{fix}} = \begin{bmatrix} P_i(\mathbf{s}_i) - P_j(\mathbf{s}_j) \\ \theta_j - \theta_i - \Delta\theta_0 \end{bmatrix} = \mathbf{0} \in \mathbb{R}^3
$$

- $\Phi_t = \mathbf{0}$.
- Jacobian: revolute block in rows 0–1; row 2 has $-1$ at $\theta_i$ and $+1$ at $\theta_j$.
- $\gamma$: revolute centripetal terms in rows 0–1; **row 2 is exactly zero** because the rotation constraint is linear in $\theta$.

### 2.3 Prismatic joint (2 eqs)
*Slide along an axis on body $i$, lock relative rotation.* `core/constraint/prismatic.rs:61-178`

Let $\hat{\mathbf{n}}_i^{\,L}$ be the body-frame perpendicular to the slide axis, $\hat{\mathbf{n}}_g = A(\theta_i)\, \hat{\mathbf{n}}_i^{\,L}$ its world-frame image, and $\mathbf{d} = P_j(\mathbf{s}_j) - P_i(\mathbf{s}_i)$.

$$
\Phi_{\text{pri}} = \begin{bmatrix} \hat{\mathbf{n}}_g \cdot \mathbf{d} \\ \theta_j - \theta_i - \Delta\theta_0 \end{bmatrix} = \mathbf{0}
$$

- $\Phi_t = \mathbf{0}$.
- Jacobian (row 0): the perpendicular projection couples $\mathbf{r}$ and $\theta$ on both bodies (see `prismatic.rs:86-118`). Row 1 is the same rotation lock as fixed.
- $\gamma_0$: assembled from $\ddot\Phi_0$; explicitly computed in `prismatic.rs:130-176`. $\gamma_1 = 0$.

### 2.4 Cam follower (1 eq)
*Distance from a point on body $i$ to a parametric cam profile on body $j$ equals the follower radius.* `core/constraint/cam.rs`. Math omitted here — same pattern as linear driver but with profile derivatives. Not on the critical path for press-style trajectory work.

### 2.5 Revolute driver (1 eq)
*Prescribes relative angle as a function of time.* `core/driver.rs:109-145`

$$
\Phi_{\text{rd}} = (\theta_j - \theta_i) - f(t) = 0
$$

- $\Phi_t = -f'(t)$. (`driver.rs:115-117`)
- Jacobian: row of zeros except $-1$ at $\theta_i$, $+1$ at $\theta_j$. Constant in $q$.
- $\gamma = f''(t)$. Because the Jacobian is constant in $q$, no velocity-quadratic terms appear. (`driver.rs:136-145`)

For the constant-speed parameterization currently used in the GUI: $f(t) = \theta_0 + \omega t$, so $f' = \omega$, $f'' = 0$.

The corresponding Lagrange multiplier $\lambda_{\text{rd}}$ has units of N·m and is the **input torque** the driver must produce.

#### Figure 2 — 4-bar with revolute driver

<svg width="640" height="340" xmlns="http://www.w3.org/2000/svg" font-family="serif">
  <defs>
    <marker id="ah2" viewBox="0 0 10 10" refX="9" refY="3" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 9 3 L 0 6 z" fill="#a02020"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="20" y="22" font-size="14" font-weight="bold">4-bar driven by revolute driver D₁ at O₂ (§2.5)</text>
  <text x="20" y="40" font-size="11" fill="#555">3 moving bodies (n=3) → q ∈ R⁹ ; 4 revolute joints (8 rows) + 1 driver (1 row) → m=9 ; DOF = 0.</text>
  <!-- Ground -->
  <line x1="60" y1="280" x2="450" y2="280" stroke="#333" stroke-width="1.8"/>
  <g stroke="#333" stroke-width="1">
    <line x1="60" y1="280" x2="50" y2="290"/><line x1="80" y1="280" x2="70" y2="290"/>
    <line x1="100" y1="280" x2="90" y2="290"/><line x1="120" y1="280" x2="110" y2="290"/>
    <line x1="140" y1="280" x2="130" y2="290"/><line x1="160" y1="280" x2="150" y2="290"/>
    <line x1="180" y1="280" x2="170" y2="290"/><line x1="200" y1="280" x2="190" y2="290"/>
    <line x1="220" y1="280" x2="210" y2="290"/><line x1="240" y1="280" x2="230" y2="290"/>
    <line x1="260" y1="280" x2="250" y2="290"/><line x1="280" y1="280" x2="270" y2="290"/>
    <line x1="300" y1="280" x2="290" y2="290"/><line x1="320" y1="280" x2="310" y2="290"/>
    <line x1="340" y1="280" x2="330" y2="290"/><line x1="360" y1="280" x2="350" y2="290"/>
    <line x1="380" y1="280" x2="370" y2="290"/><line x1="400" y1="280" x2="390" y2="290"/>
    <line x1="420" y1="280" x2="410" y2="290"/><line x1="440" y1="280" x2="430" y2="290"/>
  </g>
  <!-- Bars -->
  <line x1="120" y1="280" x2="180" y2="170" stroke="#444" stroke-width="4"/>
  <line x1="180" y1="170" x2="370" y2="145" stroke="#444" stroke-width="4"/>
  <line x1="370" y1="145" x2="400" y2="280" stroke="#444" stroke-width="4"/>
  <!-- Joints (revolute = open circles) -->
  <circle cx="120" cy="280" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="180" cy="170" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="370" cy="145" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="400" cy="280" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <!-- Driver indicator (curved arrow at J1) -->
  <path d="M 102 280 A 18 18 0 1 1 138 280" stroke="#a02020" stroke-width="2" fill="none" marker-end="url(#ah2)"/>
  <text x="138" y="240" font-size="13" fill="#a02020" font-weight="bold">D₁: ω</text>
  <!-- Joint labels -->
  <text x="100" y="310" font-size="13" font-weight="bold">O₂</text>
  <text x="395" y="310" font-size="13" font-weight="bold">O₄</text>
  <text x="100" y="266" font-size="11">J₁</text>
  <text x="160" y="162" font-size="11">J₂</text>
  <text x="380" y="138" font-size="11">J₃</text>
  <text x="385" y="266" font-size="11">J₄</text>
  <!-- Body labels -->
  <text x="115" y="225" font-size="12" font-style="italic" fill="#555">crank</text>
  <text x="255" y="148" font-size="12" font-style="italic" fill="#555">coupler</text>
  <text x="395" y="225" font-size="12" font-style="italic" fill="#555">rocker</text>
  <text x="240" y="305" font-size="12" font-style="italic" fill="#555">ground</text>
  <!-- θ_crank -->
  <line x1="120" y1="280" x2="170" y2="280" stroke="#aaa" stroke-width="0.8" stroke-dasharray="2,2"/>
  <path d="M 150 280 A 30 30 0 0 0 137 250" fill="none" stroke="#222" stroke-width="1"/>
  <text x="138" y="270" font-size="12" font-style="italic">θ_crank</text>
  <!-- Sidebar legend -->
  <text x="475" y="60" font-size="13" font-weight="bold">Constraint rows</text>
  <text x="475" y="82" font-size="11">J₁,J₂,J₃,J₄: Φ_rev (§2.1) — 8 rows</text>
  <text x="475" y="100" font-size="11" fill="#a02020">D₁: θ_crank − (θ₀+ωt) = 0 (§2.5)</text>
  <text x="475" y="124" font-size="11" font-style="italic">Total m = 9 ; n_coords = 9</text>
  <text x="475" y="142" font-size="11" font-style="italic">Φ_q is 9×9 (square)</text>
  <text x="475" y="170" font-size="13" font-weight="bold">Multipliers λ</text>
  <text x="475" y="188" font-size="11">λ_J1, ..., λ_J4 ∈ R²: pin reactions [N]</text>
  <text x="475" y="206" font-size="11" fill="#a02020">λ_D1 ∈ R: input torque [N·m]</text>
</svg>

### 2.6 Linear driver (1 eq)
*Prescribes the world distance between two body-local points as a function of time.* `core/linear_driver.rs:85-187`

Let $\mathbf{d}(q) = P_b(\mathbf{s}_b) - P_a(\mathbf{s}_a)$, $L(q) = \|\mathbf{d}(q)\|$, $\hat{\mathbf{n}} = \mathbf{d}/L$.

$$
\Phi_{\text{ld}} = L(q) - d(t) = 0
$$

(Note: **not** squared — the constraint is in length units, so the Jacobian uses the unit direction $\hat{\mathbf{n}}$, and $\lambda_{\text{ld}}$ has units of N — directly the actuator force.)

- $\Phi_t = -d'(t)$. (`linear_driver.rs:94-96`)
- Jacobian:
$$
\Phi_q = \big[ -\hat{\mathbf{n}}^T,\; -\hat{\mathbf{n}}^T B(\theta_a)\mathbf{s}_a,\; +\hat{\mathbf{n}}^T,\; +\hat{\mathbf{n}}^T B(\theta_b)\mathbf{s}_b \big]
$$
at the corresponding columns. (`linear_driver.rs:98-133`)
- $\gamma$: from differentiating $L(q(t)) - d(t) = 0$ twice:
$$
\gamma = d''(t) \;-\; \frac{|\mathbf{v}_\perp|^2}{L} \;-\; \frac{\mathbf{d}\cdot \mathbf{a}_{\text{centripetal}}}{L}
$$
where $\mathbf{v}_\perp$ is the relative velocity of the two points perpendicular to the line of action and $\mathbf{a}_{\text{centripetal}} = A(\theta_a)\mathbf{s}_a\dot\theta_a^2 - A(\theta_b)\mathbf{s}_b\dot\theta_b^2$. (`linear_driver.rs:135-187`)

For the constant-velocity parameterization in the GUI: $d(t) = L_0 + v\, t$, so $d'=v$, $d''=0$.

The Lagrange multiplier $\lambda_{\text{ld}}$ is the **actuator force** along the line of action.

#### Figure 3 — 4-bar with linear driver between ground and coupler

<svg width="640" height="340" xmlns="http://www.w3.org/2000/svg" font-family="serif">
  <!-- Title -->
  <text x="20" y="22" font-size="14" font-weight="bold">Same 4-bar; revolute driver replaced by linear actuator (§2.6)</text>
  <text x="20" y="40" font-size="11" fill="#555">D₁ now prescribes the world distance ‖P_b − P_a‖ between point P_a on ground and point P_b on coupler.</text>
  <!-- Ground -->
  <line x1="60" y1="280" x2="450" y2="280" stroke="#333" stroke-width="1.8"/>
  <g stroke="#333" stroke-width="1">
    <line x1="60" y1="280" x2="50" y2="290"/><line x1="80" y1="280" x2="70" y2="290"/>
    <line x1="100" y1="280" x2="90" y2="290"/><line x1="120" y1="280" x2="110" y2="290"/>
    <line x1="140" y1="280" x2="130" y2="290"/><line x1="160" y1="280" x2="150" y2="290"/>
    <line x1="180" y1="280" x2="170" y2="290"/><line x1="200" y1="280" x2="190" y2="290"/>
    <line x1="220" y1="280" x2="210" y2="290"/><line x1="240" y1="280" x2="230" y2="290"/>
    <line x1="260" y1="280" x2="250" y2="290"/><line x1="280" y1="280" x2="270" y2="290"/>
    <line x1="300" y1="280" x2="290" y2="290"/><line x1="320" y1="280" x2="310" y2="290"/>
    <line x1="340" y1="280" x2="330" y2="290"/><line x1="360" y1="280" x2="350" y2="290"/>
    <line x1="380" y1="280" x2="370" y2="290"/><line x1="400" y1="280" x2="390" y2="290"/>
    <line x1="420" y1="280" x2="410" y2="290"/><line x1="440" y1="280" x2="430" y2="290"/>
  </g>
  <!-- Bars -->
  <line x1="120" y1="280" x2="180" y2="170" stroke="#444" stroke-width="4"/>
  <line x1="180" y1="170" x2="370" y2="145" stroke="#444" stroke-width="4"/>
  <line x1="370" y1="145" x2="400" y2="280" stroke="#444" stroke-width="4"/>
  <!-- Joints -->
  <circle cx="120" cy="280" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="180" cy="170" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="370" cy="145" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <circle cx="400" cy="280" r="7" fill="white" stroke="#222" stroke-width="2"/>
  <!-- Linear actuator: P_a on ground (260, 280) → P_b on coupler midpoint (275, 158) -->
  <!-- Cylinder body (thicker) -->
  <line x1="260" y1="280" x2="270" y2="210" stroke="#a02020" stroke-width="6" stroke-linecap="butt"/>
  <!-- Piston rod (thinner) -->
  <line x1="270" y1="210" x2="275" y2="158" stroke="#a02020" stroke-width="2.5"/>
  <!-- End points -->
  <circle cx="260" cy="280" r="4.5" fill="#a02020"/>
  <circle cx="275" cy="158" r="4.5" fill="#a02020"/>
  <!-- Actuator labels -->
  <text x="270" y="305" font-size="12" fill="#a02020" font-weight="bold">P_a</text>
  <text x="282" y="155" font-size="12" fill="#a02020" font-weight="bold">P_b</text>
  <text x="290" y="220" font-size="12" font-style="italic" fill="#a02020">L(t) = L₀ + vt</text>
  <text x="290" y="238" font-size="11" fill="#a02020">(stroke direction)</text>
  <!-- Joint labels -->
  <text x="100" y="310" font-size="13" font-weight="bold">O₂</text>
  <text x="395" y="310" font-size="13" font-weight="bold">O₄</text>
  <text x="100" y="266" font-size="11">J₁</text>
  <text x="160" y="162" font-size="11">J₂</text>
  <text x="380" y="138" font-size="11">J₃</text>
  <text x="385" y="266" font-size="11">J₄</text>
  <!-- Body labels -->
  <text x="115" y="225" font-size="12" font-style="italic" fill="#555">crank</text>
  <text x="320" y="148" font-size="12" font-style="italic" fill="#555">coupler</text>
  <text x="395" y="225" font-size="12" font-style="italic" fill="#555">rocker</text>
  <text x="100" y="305" font-size="12" font-style="italic" fill="#555">ground</text>
  <!-- Sidebar legend -->
  <text x="475" y="60" font-size="13" font-weight="bold">Constraint rows</text>
  <text x="475" y="82" font-size="11">J₁,J₂,J₃,J₄: Φ_rev (§2.1) — 8 rows</text>
  <text x="475" y="100" font-size="11" fill="#a02020">D₁: ‖P_b−P_a‖ − L(t) = 0 (§2.6)</text>
  <text x="475" y="124" font-size="11" font-style="italic">Total m = 9 ; n_coords = 9</text>
  <text x="475" y="142" font-size="11" font-style="italic">Φ_q is 9×9 (square)</text>
  <text x="475" y="170" font-size="13" font-weight="bold">Driver row</text>
  <text x="475" y="188" font-size="11">Φ_q row: ±n̂, ±n̂·B(θ)s on each body</text>
  <text x="475" y="206" font-size="11">Φ_t = −v ;  Φ_u = −1 (re-param)</text>
  <text x="475" y="230" font-size="13" font-weight="bold">Multiplier</text>
  <text x="475" y="248" font-size="11" fill="#a02020">λ_D1 ∈ R: actuator force [N]</text>
  <text x="475" y="266" font-size="11" fill="#a02020">(along the line of action n̂)</text>
</svg>

---

## 3. Position-level kinematics

Goal: given a guess $q_0$ and time $t$, find $q$ such that $\Phi(q,t) = \mathbf{0}$.

**Algorithm:** Newton-Raphson, iterating

$$
\Phi_q(q^{(k)}, t)\, \Delta q = -\Phi(q^{(k)}, t),
\qquad q^{(k+1)} = q^{(k)} + \Delta q
$$

until $\|\Phi(q^{(k)}, t)\| < \mathrm{tol}$.

Implementation: `solver/kinematics.rs:35-80` (`solve_position`). The linear solve is via SVD with cutoff $10^{-14}$, which gracefully handles rank-deficient systems by returning a least-squares update — relevant near singularities.

`assemble_constraints` (`solver/assembly.rs:13-29`) and `assemble_jacobian` (`solver/assembly.rs:32-51`) just stack rows from the per-constraint methods.

The GUI uses this via `AppState::solve_at_angle` (`gui/state/mod.rs:885`) and `solve_at_stroke` (`gui/state/mod.rs:1033`), each of which sets the current driver parameter and calls `solve_position` with $q_{\text{prev}}$ as the warm start.

**Gotcha — branch selection.** Newton converges to the assembly mode (circuit) closest to $q_0$. For a 4-bar this is "elbow up" vs "elbow down"; the warm start from the previous frame keeps the simulation on one branch unless it's pushed near a singularity.

---

## 4. Velocity-level kinematics

Differentiating $\Phi(q(t), t) = 0$ once with respect to $t$:

$$
\Phi_q(q,t)\, \dot q + \Phi_t(q,t) = 0 \quad \Rightarrow \quad \boxed{\;\Phi_q\, \dot q = -\Phi_t\;}
$$

Single linear solve given converged $q$. **No iteration.**

Implementation: `solver/kinematics.rs:85-100` (`solve_velocity`). Same SVD pattern as `solve_position`. `assemble_phi_t` is at `solver/assembly.rs:54-70`.

For a 1-driver mechanism with all geometric joints, $\Phi_t$ is non-zero **only on the driver row**: it's $-f'(t)$ for revolute, $-d'(t)$ for linear. So the velocity solve is "given the driver's commanded rate, propagate it through the closure to all body velocities."

**GUI integration status:** `solve_velocity` *is* used by the GUI today:
- Per-frame, for the mechanical-advantage readout (`gui/state/blueprint_ops.rs:570`).
- Per-sample inside sweep analysis (`gui/sweep/mod.rs:449`) — feeds energy, mechanical-advantage, and the acceleration pipeline.
- It also drives the virtual-work cross-check (`analysis/virtual_work.rs:63`).

What's *not* wired: a direct per-frame body-velocity / coupler-point-velocity readout outside sweep mode. Trajectory mode will want this so $\dot q$ is available at every interactive timestep.

---

## 5. Acceleration-level kinematics

Differentiating once more:

$$
\Phi_q\, \ddot q + (\Phi_q\, \dot q)_q\, \dot q + 2\, \Phi_{qt}\, \dot q + \Phi_{tt} = 0
$$

Defining $\gamma \equiv -\big[(\Phi_q\dot q)_q\, \dot q + 2\, \Phi_{qt}\, \dot q + \Phi_{tt}\big]$:

$$
\boxed{\;\Phi_q\, \ddot q = \gamma\;}
$$

Single linear solve given $(q, \dot q)$. Each constraint's `gamma()` returns its rows of the global RHS; the per-constraint contributions documented in §2 cover all the velocity-quadratic terms ($\dot\theta_*^2$, $|\mathbf{v}_\perp|^2$, etc.) plus the explicit $f''(t)$ or $d''(t)$ that comes from $\Phi_{tt}$.

Implementation: `solver/kinematics.rs:105-120` (`solve_acceleration`). `assemble_gamma` at `solver/assembly.rs:73-94`.

**GUI integration status:** `solve_acceleration` is called per-sample inside sweep analysis (`gui/sweep/mod.rs:468`) — its output feeds the inverse-dynamics torque series (lambda from `solve_inverse_dynamics`) and the coupler-point acceleration plots. It is **not** called per-frame; the live single-frame display has $q$ and $\dot q$ but not $\ddot q$. Trajectory mode will likely want $\ddot q$ available per-frame too — this is plumbing of the same shape as the velocity-readout gap above.

**Why this matters for trajectory analysis:** the actuator command is $u(t)$; its derivatives $\dot u(t), \ddot u(t)$ are exactly what hardware controllers consume (velocity and feedforward). Once we have an inverse map $u(t)$ from a desired output trajectory, $\dot q$ and $\ddot q$ along the trajectory follow from §4 and §5 — no second-derivative kinematics to invent. *However*, the back-mapping to $\dot u, \ddot u$ requires extra terms when the observable $g$ is non-linear in $q$; see §8.3.

---

## 6. Statics

For inertia-free equilibrium under applied generalized forces $Q(q, \dot q, t)$ (gravity, springs, dampers, applied loads, force zones):

$$
\Phi_q^T(q,t)\, \lambda = -Q(q, \mathbf{0}, t)
$$

Solved for the multiplier vector $\lambda \in \mathbb{R}^m$. Each multiplier corresponds to one constraint row:

- **Revolute joint multipliers** ($\lambda_x, \lambda_y$ pair): the **reaction force** on body $i$ from body $j$ at the joint, expressed in world coordinates. Negate for the reaction on body $j$.
- **Fixed joint multipliers**: same translational pair as revolute, plus a third component which is the **reaction torque**.
- **Prismatic joint multipliers**: the perpendicular reaction force component, plus the rotation-lock torque.
- **Revolute driver multiplier**: the **input torque** the driver must apply at the joint.
- **Linear driver multiplier**: the **input axial force** the actuator must apply along the line of action — directly the actuator sizing number.

Implementation: `solver/statics.rs:33-79`. Uses the same SVD machinery as the kinematic solvers; rank-aware so over-constrained systems return least-squares-best multipliers and a flag.

Inverse dynamics adds the inertial term $M(q)\ddot q$ to the RHS (`solver/inverse_dynamics.rs`). Forward dynamics solves the augmented $\begin{bmatrix} M & \Phi_q^T \\ \Phi_q & 0 \end{bmatrix} \begin{bmatrix} \ddot q \\ \lambda \end{bmatrix} = \begin{bmatrix} Q \\ \gamma \end{bmatrix}$ system in `solver/forward_dynamics.rs`.

---

## 7. Summary: what's already built vs the gap

| Layer | Math | Per-constraint code | System solver | GUI uses it? |
|---|---|---|---|---|
| Position | $\Phi(q,t)=0$, Newton on $\Phi_q\Delta q = -\Phi$ | `constraint()` | `solve_position` | **Yes — per-frame** (`solve_at_angle` / `solve_at_stroke`) and per-sweep |
| Velocity | $\Phi_q\dot q = -\Phi_t$, single solve | `phi_t()` | `solve_velocity` | **Yes — per-frame** (mech. advantage, `blueprint_ops.rs:570`) and per-sweep |
| Acceleration | $\Phi_q\ddot q = \gamma$, single solve | `gamma()` | `solve_acceleration` | **Sweep only** (`sweep/mod.rs:468`) — not per-frame |
| Statics | $\Phi_q^T\lambda = -Q$, single solve | `jacobian()` (transposed) | `solve_statics` | **Yes — per-frame** (force breakdown, actuator force readout) and per-sweep |
| Inverse dynamics | adds $M\ddot q$ | `assemble_mass_matrix` | `solve_inverse_dynamics` | **Sweep only** (`sweep/mod.rs:470`) — not per-frame |

**The math layer is essentially complete for forward kinematics, and is well-exercised by the sweep pipeline.** What position-control adds is an **inverse map** that wraps the existing `solve_position` (and uses §4–§5 for velocity/acceleration of the input). The trajectory feature will likely follow the sweep pipeline's pattern of "loop over input parameter, call all five solvers per sample, accumulate into a `*Data` struct" — see `gui/sweep/mod.rs:435-503` for the working template.

---

## 8. The inverse-kinematics extension (new math for trajectory mode)

We define an **output observable** as a smooth scalar function of the body coordinates:

$$
g: \mathbb{R}^{3n} \to \mathbb{R}, \qquad g(q)
$$

Examples (the `ControlTarget` enum we'll define):
- $g(q) = \theta_i$ — angle of body $i$.
- $g(q) = \mathbf{e}_x \cdot P_i(\mathbf{s})$ — world $x$ of a body-local point.
- $g(q) = \hat{\mathbf{u}} \cdot (P_i(\mathbf{s}) - \mathbf{p}_0)$ — projection of a body-local point onto a fixed line.
- $g(q) = \|P_i(\mathbf{s}) - \mathbf{p}_0\|$ — distance from a body-local point to a fixed reference.

Each variant supplies $g(q)$ and $\nabla_q g(q)$ in closed form. We do **not** require $\partial^2 g / \partial q^2$ — see §8.3.

### 8.1 Position inverse — find $u$ such that $g(q(u)) = h$

Given a target $h$ and a converged forward map $q(u)$ (i.e. $\Phi(q,u) = 0$), we want to find $u^*$ such that $g(q(u^*)) = h$. Define the residual

$$
r(u) = g(q(u)) - h
$$

Newton on $r(u) = 0$:

$$
u^{(k+1)} = u^{(k)} - \frac{r(u^{(k)})}{r'(u^{(k)})}
$$

The chain rule with the implicit function theorem gives

$$
\frac{dq}{du} = -\Phi_q^{-1}\, \Phi_u
\qquad\Rightarrow\qquad
r'(u) = \nabla_q g \cdot \frac{dq}{du} = -\nabla_q g \cdot \Phi_q^{-1}\, \Phi_u
$$

where $\Phi_u$ is the column of derivatives with respect to the driver input parameter. For our parameterizations:
- Revolute driver: $f(t) = \theta_0 + \omega t$ ⟹ $u = \theta_0 + \omega t$. $\Phi_u$ row is $-1$ on the driver row (zero elsewhere). Equivalently $\Phi_u = \Phi_t / \omega$.
- Linear driver: $d(t) = L_0 + v t$ ⟹ $u = L_0 + v t$. $\Phi_u$ row is $-1$ on the driver row, zero elsewhere. Equivalently $\Phi_u = \Phi_t / v$.

So **$\Phi_u$ is one column with a single $-1$ entry on the driver row**, regardless of the driver kind. No new code is needed beyond a function that returns this column (effectively `phi_t` divided by the parameterization rate).

**Outer loop:**
1. $u_0 \leftarrow$ current driver input.
2. Forward solve: $q_k \leftarrow$ `solve_position(q_{k-1}, u_k)`.
3. $r_k \leftarrow g(q_k) - h$. If $|r_k| < \text{tol}$, return $u_k$.
4. Compute $r'_k$ via one extra linear solve $\Phi_q s = -\Phi_u$, then $r' = \nabla_q g \cdot s$.
5. $u_{k+1} \leftarrow u_k - r_k / r'_k$. Loop to 2.

Cost: forward solve + one extra linear solve per outer iteration. Quadratic convergence. Bisection fallback on the existing sweep table when Newton diverges or branch-jumps.

### 8.2 Velocity inverse — find $\dot u$ such that $\dot g = \dot h$

Closed-form, no iteration:

$$
\dot g = \nabla_q g \cdot \dot q = \nabla_q g \cdot \frac{dq}{du}\, \dot u = r'(u)\, \dot u
\quad\Rightarrow\quad
\boxed{\;\dot u = \dot h \,/\, r'(u)\;}
$$

$r'(u)$ already computed in §8.1. The accompanying body velocities $\dot q$ come from §4: $\Phi_q \dot q = -\Phi_t$ — but with $\Phi_t$ recomputed from the *back-solved* $\dot u$, not from the user-set driver rate. Equivalently, scale the existing $\dot q$ result by $\dot u / u_{\text{rate, nominal}}$.

### 8.3 Acceleration inverse — find $\ddot u$ such that $\ddot g = \ddot h$

(Optional layer. Most actuator controllers consume $(u, \dot u)$ and synthesize $\ddot u$ internally. Implement only if the trajectory deliverable specifically calls for $\ddot u$.)

Differentiate $\dot g = r'(u)\, \dot u$:

$$
\ddot g = r''(u)\, \dot u^2 + r'(u)\, \ddot u
\quad\Rightarrow\quad
\boxed{\;\ddot u = \big(\ddot h - r''(u)\, \dot u^2\big) / r'(u)\;}
$$

The new term is $r''(u) = d^2 g / du^2$. By chain rule, this has **two contributions**:

$$
r''(u) = \underbrace{\nabla_q^2 g \,(dq/du,\, dq/du)}_{\text{Hessian of observable}} + \underbrace{\nabla_q g \cdot (d^2 q/du^2)}_{\text{constraint-acceleration term}}
$$

**The constraint-acceleration term** ($d^2q/du^2$): from differentiating $\Phi_q (dq/du) + \Phi_u = 0$ once more,

$$
\Phi_q\, \frac{d^2 q}{du^2} = -\big[\Phi_{qq}(dq/du, dq/du) + 2\, \Phi_{qu}\, (dq/du) + \Phi_{uu}\big]
$$

For our drivers (where $\Phi$ depends on $u$ only through a $-u$ term), $\Phi_{qu} = 0$ and $\Phi_{uu} = 0$, so the RHS reduces to $-\Phi_{qq}(dq/du, dq/du)$. This is **structurally the velocity-quadratic part of $\gamma$ from §5** — assemble $\gamma$ with $(dq/du)$ in place of $\dot q$ and the explicit $f''(t), d''(t)$ terms zeroed out. This reuses most of the existing `gamma()` machinery but needs a thin variant ("kinematic-only $\gamma$").

**The Hessian-of-observable term** ($\nabla_q^2 g$): this depends on the observable. For each `ControlTarget` variant:
- $g(q) = \theta_i$: $\nabla_q^2 g = 0$. Term vanishes.
- $g(q) = \mathbf{e}_x \cdot P_i(\mathbf{s}) = x_i + \cos(\theta_i)\, s_x - \sin(\theta_i)\, s_y$: Hessian has a single non-zero $(\theta_i, \theta_i)$ entry equal to $-(\cos(\theta_i) s_x - \sin(\theta_i) s_y) = -(\mathbf{e}_x \cdot A(\theta_i)\mathbf{s})$. Same shape for $g = \mathbf{e}_y \cdot P_i$.
- $g(q) = \hat{\mathbf{u}} \cdot (P_i(\mathbf{s}) - \mathbf{p}_0)$: Hessian non-zero only at $(\theta_i, \theta_i)$, equal to $-\hat{\mathbf{u}} \cdot A(\theta_i)\mathbf{s}$.
- $g(q) = \|P_i(\mathbf{s}) - \mathbf{p}_0\|$: Hessian non-trivial; comes from differentiating the unit direction vector. Derivable but algebra-heavy.

So we have **two implementation options**:

**(a) Analytic.** Each `ControlTarget` variant supplies $\nabla_q^2 g$ in addition to $\nabla_q g$. Combined with the kinematic-only $\gamma$ above, this gives $r''$ exactly. Costs: ~50 LoC per variant for the Hessian; one extra linear solve per timestep.

**(b) Finite differences.** Compute $r'(u)$ at $u \pm \delta$ via two extra forward solves; $r''(u) \approx (r'(u + \delta) - r'(u - \delta))/(2\delta)$. Sidesteps both terms entirely. $O(\delta^2)$ accurate, robust near-singularity, no Hessian bookkeeping. Cost: 2 extra forward solves per timestep, but each is warm-started and converges in 2-3 iterations — cheap.

Recommendation: **(b) for v1**. Switch to (a) only if profiling shows the FD cost matters at trajectory resolution, or if hardware export needs >1e-6 accuracy on $\ddot u$.

### 8.4 What this means for the existing $\gamma$ machinery

Once we have $u(t), \dot u(t), \ddot u(t)$ along the trajectory, the body acceleration $\ddot q(t)$ is just the existing $\Phi_q \ddot q = \gamma$ solve — but $\gamma$ now incorporates the trajectory's $\dot u, \ddot u$ via $\Phi_t = -\dot u, \Phi_{tt} = -\ddot u$ on the driver row. **No new acceleration-level math** beyond §5; the trajectory layer just feeds different RHS values into the same solver.

The same is true for actuator-force computation: at each timestep we run the existing `solve_statics` (or `solve_inverse_dynamics`) with the current $q$ — its multiplier on the driver row is $F_{\text{actuator}}(t)$.

---

## 9. What's missing for trajectory-mode position control

Marked **MATH**, **CODE**, or **GUI** by the layer at which the gap lives.

1. **CODE** — A `Phi_u` accessor (or "phi_t-as-Phi_u" helper) on `Mechanism` or a free function in `solver`. One function call, ~10 lines.
2. **CODE** — A `ControlTarget` enum + trait providing $g(q)$ and $\nabla_q g(q)$ for each variant (angle, world-x, world-y, projection, distance). ~150 lines.
3. **CODE** — `solve_for_target(mech, q0, target, h)` outer Newton wrapping `solve_position`. With bisection fallback. ~80 lines + tests.
4. **MATH/CODE** — Closed-form inverse velocity & inverse acceleration helpers per §8.2–§8.3. ~40 lines.
5. **GUI** — Expose $\dot q$ in the per-frame display (`solve_velocity` is already called per-frame for mech-advantage; just route its output to plots/readouts). ~30 lines + plot integration.
6. **GUI** — Add a per-frame `solve_acceleration` call (currently sweep-only at `gui/sweep/mod.rs:468`). ~30 lines.
7. **GUI** — Trajectory specification UI (analytic profile selector — start/end, duration, accel/decel fractions). Reuse the existing `gui/sweep/motion_profile.rs` trapezoidal code; add S-curve later. ~200 lines.
8. **GUI** — Trajectory plot panel: $\text{target}(t), g_{\text{achieved}}(t), e(t), u(t), \dot u(t), F_{\text{act}}(t)$. ~250 lines.
9. **GUI/EXPORT** — CSV export of the full time series.
10. **GUI** — Reachability visualization (overlay achievable range $[g_{\min}, g_{\max}]$ on the trajectory plot, derived from a sweep over $u$).
11. **GUI** — Singularity & branch-jump detection: surface "the actuator has lost authority on the target" and "trajectory requires switching assembly modes".

Items 1–4 are the **math/solver work**; everything else is plumbing and UI.

---

## 10. Open follow-ups (not blocking, but related)

- The deferred **R1b** refactor (`docs/ai/04-memory.yaml`) — collapsing the dual-purpose driver scalars into payload on `DriverKind` — would clean up the eventual GUI wiring of the inverse layer. Worth revisiting once trajectory mode is exercised by users.
- The current `MotionProfile` machinery in `gui/sweep/motion_profile.rs` was built for **input-side** profiles (driver follows trapezoidal velocity). Trajectory mode reuses the same shape vocabulary but applies it to the **output observable** with the inverse map filling in $u(t)$. The existing code is ~50% reusable; the rest gets refactored into a shared `MotionProfile { kind, params }` module that both sweep and trajectory consume.

---

*End of reference. Cross-references in the position-control design spec will point back to specific sections of this document.*
