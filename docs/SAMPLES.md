# Sample Mechanisms

All samples are available under **File > Load Sample** in the GUI. Each demonstrates a specific linkage type or analysis feature.

## Four-Bar Linkages

| Sample | Description |
|--------|-------------|
| **4-Bar Crank-Rocker** | Classic Grashof crank-rocker (ground=0.038, crank=0.01, coupler=0.04, rocker=0.03). The shortest link (crank) is grounded and rotates continuously. Good baseline for exploring the kinematic sweep and coupler-curve plots. |
| **Crank-Rocker (4-2-4-3)** | Grashof crank-rocker with larger link proportions (d=4, a=2, b=4, c=3). Demonstrates continuous crank rotation with a coupler point tracing a well-defined coupler curve. |
| **Double-Rocker (5-3-4-7)** | Non-Grashof four-bar (d=5, a=3, b=4, c=7). No link can complete a full revolution; both the input and output links oscillate. Shows how the solver handles limited-range motion and non-Grashof geometry. |
| **Double-Crank (2-4-3.5-3)** | Grashof drag-link four-bar where the ground link is the shortest (d=2). Both the crank and rocker rotate fully. Demonstrates double-crank behavior and driver reassignment. |
| **Parallelogram (4-2-4-2)** | Equal opposite links (a=c=2, b=d=4). The coupler translates without rotating, producing a pure translation output. Useful for verifying zero angular velocity on the coupler. |
| **Chebyshev (4-2-5-5)** | Chebyshev approximate straight-line linkage (d=4, a=2, b=5, c=5). The coupler midpoint traces a nearly horizontal straight line over part of the cycle, demonstrating coupler-curve path generation. |
| **Triple-Rocker (4-2-5-2)** | Non-Grashof four-bar where no link satisfies the Grashof condition (shortest+longest > sum of others). All three moving links oscillate. Demonstrates limited-range motion with an initial-angle offset to ensure the loop closes. |

## Slider-Crank Variants

| Sample | Description |
|--------|-------------|
| **Slider-Crank** | Standard slider-crank with a prismatic joint (crank=0.01, coupler=0.04). The slider translates along the X axis. Demonstrates prismatic constraints and linear output motion from rotary input. |
| **Inverted Slider-Crank** | The slider rides along a rotating coupler rather than a fixed rail. Produces a different motion profile than the standard slider-crank. Demonstrates inverted slider-crank kinematics. |
| **Quick-Return (crank-shaper)** | Four-bar configured as a quick-return mechanism (ground=0.060, crank=0.015, coupler=0.060, rocker=0.045). The short crank creates an asymmetric output stroke where the return is faster than the forward stroke. Demonstrates time-ratio analysis. |
| **Scotch Yoke (pure sinusoidal)** | A crank drives a slider through a prismatic joint constrained to vertical translation. The slider position is exactly r * sin(theta), producing pure sinusoidal output. Demonstrates prismatic constraints on a non-standard axis. |

## Special Purpose

| Sample | Description |
|--------|-------------|
| **Toggle Clamp (4-bar near-toggle)** | Four-bar designed so the output link approaches 180-degree alignment at one extreme (ground=0.040, crank=0.012, coupler=0.038, rocker=0.020). Near-toggle produces very high mechanical advantage, demonstrating clamping force amplification. |

## Six-Bar Linkages

| Sample | Description |
|--------|-------------|
| **6-Bar B1 (Watt I)** | Watt Type I six-bar with ternary ground. Chain B topology with 5 moving bodies and 7 joints. Demonstrates multi-loop solver behavior and ternary-link modeling. |
| **6-Bar A1 (Chain A, binary ground)** | Chain A six-bar with binary ground and two ternary moving links. Includes a coupler point on the first ternary link. Demonstrates coupler-curve tracing on a six-bar mechanism. |
| **6-Bar A2 (Chain A, ternary ground)** | Chain A six-bar with ternary ground (three ground pivots). Includes a coupler point. Demonstrates how moving the ternary link to ground changes the mechanism's motion characteristics. |
| **6-Bar B2 (Chain B, shared-binary)** | Chain B six-bar where the two ternary links share a common binary ground. Includes coupler points on both ternary links. Demonstrates shared-pivot topology. |
| **6-Bar B3 (Chain B, exclusive-binary)** | Chain B six-bar where each ternary link connects to ground via a separate binary link. Includes coupler points on both ternary links. Demonstrates exclusive-binary ground topology. |
