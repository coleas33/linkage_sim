# Parametric Studies

Parametric studies let you sweep a single design variable across a range of values and see how it affects a performance metric. Instead of manually adjusting a parameter, running the simulation, recording a number, and repeating, the tool automates the entire loop and plots the result.

This is useful for questions like:
- "How does crank length affect peak driver torque?"
- "What spring stiffness minimizes joint reactions?"
- "How sensitive is the transmission angle to ground pivot placement?"

## What You Can Sweep

Any of the following design variables can be swept:

### Geometry (attachment point coordinates)

Every attachment point on every body (including ground) exposes its X and Y local coordinates as sweepable parameters. Changing an attachment point coordinate changes the effective link length or pivot location.

For example, on a four-bar with a crank body that has two attachment points at `(0,0)` and `(2,0)`, sweeping `crank.B x` from `1.5` to `2.5` effectively sweeps the crank length.

### Body properties

| Parameter | Units | Notes |
|-----------|-------|-------|
| Body mass | kg | Must be positive |
| Body Izz (moment of inertia about CG) | kg*m^2 | Must be positive |

### Force element parameters

Each force element in the mechanism exposes its own set of sweepable fields:

| Force Element | Sweepable Fields |
|---------------|-----------------|
| Linear spring | stiffness (N/m), free length (m) |
| Torsion spring | stiffness (N*m/rad), free angle (rad) |
| Linear damper | damping (N*s/m) |
| Rotary damper | damping (N*m*s/rad) |
| Gas spring | initial force (N), extended length (m), stroke (m) |
| Motor | stall torque (N*m), no-load speed (rad/s) |
| External force | force X (N), force Y (N) |
| External torque | torque (N*m) |
| Bearing friction | constant drag (N*m), viscous coeff, coulomb coeff |
| Joint limit | stiffness (N*m/rad) |
| Linear actuator | force (N), speed limit (rad/s), stroke min/max (m), end-stop stiffness/damping/restitution |

### Driver speed

| Parameter | Units |
|-----------|-------|
| Driver omega | rad/s |

## Output Metrics

You choose one output metric to plot on the Y-axis. The simulator runs a full 0-360 degree driver sweep at each parameter step, then extracts a single scalar from that sweep. Available metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Peak Driver Torque** (N*m) | Maximum absolute driver torque over the full rotation. Useful for motor sizing. |
| **RMS Driver Torque** (N*m) | Root-mean-square driver torque. Better than peak for estimating average power or thermal load. |
| **Min Transmission Angle** (deg) | Smallest transmission angle during the cycle (four-bar only). Low values indicate poor force transmission. |
| **Max Transmission Angle** (deg) | Largest transmission angle during the cycle (four-bar only). |
| **Peak Joint Reaction** (N) | Highest reaction force magnitude at any joint, at any point in the cycle. Useful for bearing selection. |
| **Peak Kinetic Energy** (J) | Maximum kinetic energy during the cycle. Indicates how much energy is stored in moving parts. |
| **Mean Mechanical Advantage** (ratio) | Average velocity ratio between driver and output over the cycle. Extreme values near toggle positions are filtered out automatically. |

## Setting Up a Study

### 1. Load a mechanism

Open or build a mechanism in the simulator. The parametric study panel requires a loaded mechanism to populate the parameter list.

### 2. Open the Parametric Study panel

The panel is in the side panel area. You will see the heading "Parametric Study" with a dropdown for selecting the sweep parameter.

### 3. Select the parameter to sweep

Use the "Parameter to sweep" dropdown to pick your design variable. The dropdown lists every available parameter, organized as:
- Body properties (mass, Izz for each moving body)
- Attachment point coordinates (X and Y for each point on each body, including ground)
- Force element fields (stiffness, damping, etc. for each force in the model)
- Driver omega

When you select a parameter, the Min and Max fields auto-populate to +/-50% of the current value. You can (and usually should) adjust these.

### 4. Set the sweep range

- **Min**: the starting value for the parameter
- **Max**: the ending value for the parameter
- **Steps**: how many evenly spaced values to evaluate (2 to 50)

The Min must be less than Max. For parameters that must be physically positive (mass, stiffness, spring free length, etc.), Min must be greater than zero. The Run button will be grayed out with a tooltip if validation fails.

### 5. Choose the output metric

Use the "Output metric" dropdown to pick what you want to plot on the Y-axis.

### 6. Run the study

Click **Run Study**. The simulator clones the current mechanism at each parameter step, modifies the selected parameter, runs a full kinematic and static/dynamic sweep (0-360 degrees in 1-degree steps), and extracts the chosen metric.

When the study completes, you see:
- A plot of the metric vs. the swept parameter value
- A numeric summary showing the range of the output metric across all steps

## Walkthrough: Sweep Crank Length

Suppose you have a four-bar crank-rocker loaded and you want to understand how crank length affects peak driver torque.

1. **Identify the parameter.** The crank length is determined by the distance between its two attachment points. If the crank body has points at `(0, 0)` and `(2, 0)`, then the crank length is 2 m. To sweep the length, you sweep the X coordinate of the second point.

2. **Select the parameter.** In the "Parameter to sweep" dropdown, choose the entry that looks like `crank.B x (m)` (the exact name depends on your body and point names).

3. **Set the range.** The auto-populated range will be 50% to 150% of the current value. For a current length of 2 m, that gives Min = 1.0, Max = 3.0. Adjust as needed -- for example, set Min = 1.5, Max = 2.5 for a tighter study.

4. **Set the steps.** Start with 10-15 steps. More steps give a smoother curve but take longer to compute.

5. **Choose the metric.** Select "Peak Driver Torque (N*m)".

6. **Run.** Click "Run Study" and review the resulting plot. Look for trends (e.g., torque increases with crank length) and any sharp changes that might indicate a toggle condition or mechanism lock-up.

## Counterbalance Assistant

The Parametric Study panel also includes a **Counterbalance Assistant** that automates a common design task: finding the optimal spring to minimize driver torque ripple.

### How it works

The assistant performs a 2D grid search over spring stiffness (k) and free length (L0). For each combination, it adds a linear spring between two attachment points you specify, runs a full driver sweep, and measures the peak-to-peak torque variation. The combination that produces the smallest peak-to-peak torque wins.

### Setup

1. **Spring point A** and **Spring point B**: pick the two attachment points where the counterbalance spring connects. Typically one is on the ground and the other is on a moving link.
2. **k min / k max / k steps**: the stiffness search range in N/m and number of grid points.
3. **L0 min / L0 max / L0 steps**: the free length search range and grid points.
4. Click **Optimize Counterbalance**.

### Results

The assistant reports:
- **Optimal k** and **optimal L0**: the best spring parameters found.
- **Torque reduction**: baseline peak-to-peak vs. optimized peak-to-peak, with a percentage improvement.
- **Before/after torque plot**: overlay of the driver torque curve with and without the optimal spring, so you can visually confirm the improvement.

## Tips

- **Start coarse, then refine.** Use 5-10 steps for initial exploration, then narrow the range and increase steps once you know the interesting region.
- **Watch for NaN gaps in the plot.** If the solver cannot close the loop at a particular parameter value (mechanism locks up or Grashof condition changes), that step produces NaN and is skipped in the plot. Gaps in the curve are a signal that the mechanism is approaching a kinematic limit.
- **Transmission angle metrics only work for four-bar mechanisms.** The simulator auto-detects classic four-bar topology. For six-bar or slider-crank mechanisms, these metrics will read as zero.
- **Mechanical advantage filters extreme values.** Near toggle positions, the mechanical advantage goes to infinity. The metric automatically excludes values above 1e6 to keep the average meaningful.
- **The study uses the current mechanism state as its starting guess.** If the current configuration is far from the swept values, the solver may have trouble converging at extreme ends of the range. Position the driver angle near mid-stroke before running.
- **Each step is a full 360-degree sweep.** A 50-step study runs 50 complete kinematic solves (each with 361 position solves). This is fast for four-bar mechanisms but may take a few seconds for six-bar or mechanisms with many force elements.

## Limitations

- **Single-parameter sweep only.** You cannot sweep two parameters simultaneously (e.g., crank length and coupler length together). For multi-parameter optimization, use the counterbalance assistant (which does a 2D grid over k and L0) or run separate single-parameter studies.
- **No interactive selection on the plot.** You can see the plotted curve and numeric range, but you cannot click a point on the parametric curve to inspect the full sweep data at that parameter value (the `selected_sweep` field exists in the data model but is not yet wired to the UI).
- **Attachment point sweeps change local coordinates, not link lengths directly.** There is no "crank length" parameter; you sweep the X or Y coordinate of a specific point. This gives full control but requires you to understand which coordinate controls the dimension you care about.
- **No export.** Results are displayed in the panel but cannot currently be exported to CSV or other formats.
