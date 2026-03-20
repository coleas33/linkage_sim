# GUI Engineering Validation Playbook

A manual validation plan to verify the linkage simulator produces correct engineering results. Work through tests in order — each builds confidence for the next. Stop at any failure and investigate before continuing.

**What this validates:** That the GUI correctly drives the solver, displays results accurately, applies unit conversions, and handles edge cases. The underlying solver has 714 automated tests and golden fixture data — this playbook validates the last mile between solver and user.

**Time estimate:** 1–2 hours for the full playbook. Tests 1–3 cover 80% of the risk.

---

## Test 1 — Four-Bar Kinematics (Position Solver)

**Goal:** Verify the position solver produces correct body angles and coupler point positions.

**Setup:**
1. Launch GUI: `cargo run --bin linkage-gui`
2. File > Load Sample > **4-Bar Crank-Rocker**
3. Verify status bar: green dot, residual < 1e-10, DOF: 0
4. Verify Diagnostics panel: Grashof = **Crank-Rocker**

**Check 1.1 — Spot-check positions at known angles:**

Use the angle slider to set the crank to these angles. Compare the status bar theta and the body angle plot to the reference values.

| Crank (deg) | Coupler angle (rad) | Rocker angle (rad) | Coupler P (x, y) meters |
|-------------|--------------------|--------------------|-------------------------|
| 30 | 0.4993 | 1.8236 | (1.9435, 1.6572) |
| 45 | 0.4172 | 1.8495 | (1.8758, 1.7721) |
| 90 | 0.2458 | 2.0469 | (1.4714, 1.8397) |
| 135 | 0.1379 | 2.4189 | (0.9301, 1.6685) |
| 270 | 0.6367 | 1.3789 | (1.1543, 0.5943) |

**Tolerance:** Angles within ±0.001 rad. Coupler position within ±0.001 m.

**How to read:** Select the coupler body in the property panel to see its angle. The coupler point coordinates appear in the coupler trace plot tab or by hovering.

**Check 1.2 — Full sweep continuity:**
1. Enable View > Plot Panel
2. Play animation (press Play, speed 90 deg/s, Loop mode)
3. Watch the **Body Angles** plot tab: crank should be a straight line (0-360), coupler and rocker should be smooth curves with no discontinuities or jumps
4. Watch the **Coupler Trace** plot tab: should form a closed smooth curve — the first and last points must coincide

**PASS criteria:**
- [ ] All 5 spot-check positions match within tolerance
- [ ] Body angle curves are smooth with no jumps
- [ ] Coupler trace forms a closed curve
- [ ] Status bar shows green convergence at every angle

---

## Test 2 — Four-Bar Statics (Force Solver)

**Goal:** Verify driver torque and joint reactions under gravity loading.

**Setup:**
1. Same 4-bar mechanism from Test 1
2. Verify View > Gravity is **enabled** (checkbox checked)
3. Verify View > Force Arrows is **enabled**

**Check 2.1 — Spot-check driver torque:**

| Crank (deg) | Driver torque (N*m) | Notes |
|-------------|--------------------:|-------|
| 30 | +21.086 | Positive = motor driving against gravity |
| 80 | -6.326 | Negative = gravity assists motion |
| 130 | -42.047 | Large negative = gravity strongly assists |
| 230 | -0.513 | Near zero = balanced |
| 280 | +21.643 | Positive again = motor fights gravity |
| 330 | +34.506 | Peak positive region |

**How to read:** The driver torque appears in the status bar (tau = ...) and in the **Driver Torque** plot tab.

**Tolerance:** ±0.01 N*m

**Check 2.2 — Spot-check joint reactions at crank = 30 deg:**

| Joint | Resultant force (N) |
|-------|--------------------:|
| J1 (input pin) | 37.774 |
| J2 (coupler-crank) | 18.592 |
| J3 (coupler-rocker) | 12.939 |
| J4 (output pin) | 31.795 |

**How to read:** Select each joint in the property panel — reaction forces appear under "Reaction Forces." Or check the **Joint Reactions** plot tab for the full sweep.

**Tolerance:** ±0.1 N

**Check 2.3 — Torque curve shape:**
1. Open the **Driver Torque** plot tab
2. The curve should cross zero twice (near 80 deg and 230 deg)
3. Peak positive should be near 330 deg (~35 N*m)
4. Peak negative should be near 130 deg (~-42 N*m)
5. Curve should be smooth — any sharp spikes indicate a solver issue

**Check 2.4 — Toggle point (180 deg):**
1. Slide to crank = 180 deg
2. The mechanism is at or near toggle (dead point)
3. Reactions will spike to very large values — this is physically correct
4. The Diagnostics panel should show a high condition number
5. Toggle markers (red dashed vertical lines) should appear on plots near 180 deg

**PASS criteria:**
- [ ] All 6 torque values match within tolerance
- [ ] All 4 reaction magnitudes match at 30 deg
- [ ] Torque curve is smooth except near toggle at 180 deg
- [ ] Toggle markers appear on plots

---

## Test 3 — Slider-Crank Kinematics (Prismatic Joint)

**Goal:** Verify prismatic joints work correctly using a closed-form comparison.

**Setup:**
1. File > Load Sample > **Slider-Crank**
2. Verify status bar: green dot, DOF: 0

**Check 3.1 — Slider position vs. closed-form:**

The slider x-position has a known analytical solution:

```
x_slider = r*cos(theta) + sqrt(L^2 - r^2*sin^2(theta))
```

where r = crank length, L = conrod length (check property panel for exact values).

| Crank (deg) | Expected x_slider | How to verify |
|-------------|------------------:|---------------|
| 0 | r + L (maximum extension) | Slider at rightmost position |
| 90 | L (crank vertical) | Select slider body, check position in property panel |
| 180 | L - r (minimum extension) | Slider at leftmost position (TDC) |

**Check 3.2 — Velocity zero at TDC:**
1. Open the **Coupler Velocity** plot tab
2. Slider velocity must be zero at crank = 0 deg and 180 deg (top/bottom dead center)
3. Maximum velocity should occur near crank = 90 deg and 270 deg

**Check 3.3 — Slider motion is linear:**
1. Watch animation — the slider body should move horizontally only, no vertical drift
2. The prismatic joint constrains motion to one axis

**PASS criteria:**
- [ ] Slider x-position matches closed-form at 3 check points
- [ ] Velocity is zero at dead centers
- [ ] Slider motion is purely horizontal

---

## Test 4 — Inverse Dynamics (Inertial Effects)

**Goal:** Verify that inertial loads are computed correctly. Inverse dynamics torque should differ from statics by the M*q_ddot contribution.

**Setup:**
1. Load the 4-bar crank-rocker (same as Tests 1-2)
2. Open the **Inverse Dynamics** plot tab

**Check 4.1 — Inverse dynamics vs. statics overlay:**

The plot should show two curves: statics torque and inverse dynamics torque. The difference is the inertial contribution.

| Crank (deg) | Statics tau | Inv. Dyn. tau | Delta (inertia) |
|-------------|------------:|--------------:|----------------:|
| 30 | +21.086 | +21.407 | +0.322 |
| 80 | -6.326 | -5.107 | +1.219 |
| 130 | -42.047 | -41.506 | +0.540 |
| 280 | +21.643 | +23.007 | +1.365 |

**What to verify:**
- The two curves should be close but not identical
- The inertial delta should be smooth and small compared to the total torque (< 5% away from toggle)
- Near toggle (180 deg), both curves diverge to extreme values — this is expected

**Tolerance:** ±0.1 N*m on the delta

**PASS criteria:**
- [ ] Inverse dynamics torque is consistently higher/lower than statics by the inertial contribution
- [ ] Delta is smooth and physically reasonable (proportional to acceleration)
- [ ] Both curves are overlaid on the same plot for visual comparison

---

## Test 5 — Forward Dynamics (Energy Conservation)

**Goal:** Verify the time-domain integrator conserves energy and maintains constraint satisfaction.

**Setup:**
1. Load the 4-bar crank-rocker
2. In the input panel, click **Simulate** (forward dynamics button)
3. Set simulation duration to 3 seconds

**Check 5.1 — Energy balance:**
1. Open the **Energy** plot tab
2. Three curves: KE (kinetic), PE (potential/gravity), Total
3. **Without damping:** Total energy should be approximately constant (oscillating KE + PE, but total flat)
4. **With damping (add a damper):** Total energy should decrease monotonically

Reference (undamped, t=0 to t=3):

| Time (s) | KE (J) | PE (J) | Total (J) |
|----------|-------:|-------:|----------:|
| 0.000 | 6.375 | 67.108 | 73.483 |
| 0.745 | 8.155 | 61.229 | 69.384 |
| 3.000 | 5.080 | 63.885 | 68.965 |

**Note:** Some energy loss is expected from the explicit RK4 integrator — total energy should stay within ~5% of the initial value over 3 seconds. Larger drift indicates a solver problem.

**Check 5.2 — Constraint drift:**
1. The status bar shows "drift = ..." during simulation playback
2. Drift should stay below 1e-3 for most of the simulation
3. If drift exceeds 0.01, the constraint projection may not be keeping up

**Check 5.3 — Physical plausibility:**
1. Scrub the timeline — the mechanism should move smoothly
2. No bodies should fly apart or interpenetrate
3. If you release from a displaced position with a spring, it should oscillate and (with damping) settle to equilibrium

**PASS criteria:**
- [ ] Total energy stays within 5% of initial over 3 seconds
- [ ] Constraint drift stays below 1e-3
- [ ] Mechanism motion is smooth and physically plausible

---

## Test 6 — Unit Conversion Fidelity

**Goal:** Verify the GUI displays correct values in both unit systems.

**Setup:**
1. Load the 4-bar crank-rocker
2. Set View > Units > **Millimeters** and **Degrees**

**Check 6.1 — Length conversion:**
1. Select a body in the property panel
2. Attachment point coordinates should be displayed in mm (1000x the SI values)
3. Ground link: O2 at (0, 0) mm, O4 at (4000, 0) mm — wait, the sample uses meters internally. Verify the property panel shows coordinates scaled by 1000.

**Check 6.2 — Angle conversion:**
1. The status bar theta should show degrees (e.g., "30.0 deg" not "0.5236 rad")
2. The body angle plot should have a Y-axis in degrees
3. Slider at crank 90: should display "90.0 deg"

**Check 6.3 — Toggle between units:**
1. Switch to Meters/Radians: values should change by exact conversion factors
2. Switch back: values should return to original display
3. No rounding drift — toggle 5 times and verify values are identical each time

**PASS criteria:**
- [ ] mm values are exactly 1000x the meter values
- [ ] Degree values are exactly (180/pi)x the radian values
- [ ] No drift from repeated unit toggling

---

## Test 7 — Save/Load Round-Trip

**Goal:** Verify mechanism data survives serialization.

**Setup:**
1. Load the 4-bar crank-rocker
2. Add a force element (e.g., torsion spring on the input joint, k=100 N*m/rad)
3. Add a point mass (e.g., 2 kg on the coupler at local (0.05, 0))
4. Note the driver torque at crank = 30 deg

**Check 7.1 — Save and reload:**
1. Ctrl+S to save as `test_roundtrip.json`
2. File > New (Ctrl+N) to clear
3. File > Open JSON, load `test_roundtrip.json`
4. Verify: same driver torque at crank = 30 deg (within ±0.001 N*m)
5. Verify: torsion spring still appears in force element list
6. Verify: point mass still appears on the coupler body

**Check 7.2 — Export CSV and verify:**
1. File > Export Sweep CSV
2. Open the CSV in a spreadsheet
3. Verify the driver_torque column at 30 deg matches the GUI display
4. Verify column headers make sense (angle_deg, body angles, torque, reactions, energy)

**PASS criteria:**
- [ ] Driver torque matches before and after save/load within ±0.001 N*m
- [ ] All force elements and point masses survive roundtrip
- [ ] CSV export matches GUI display values

---

## Test 8 — Build From Scratch (Full Workflow)

**Goal:** Verify a user can build a working mechanism entirely in the GUI.

**Build a simple slider-crank:**
1. File > New (Ctrl+N)
2. Right-click canvas > Add Ground Pivot at approximately (0, 0)
3. Right-click canvas > Add Ground Pivot at approximately (0.3, 0)
4. Select the **+ Body** tool, click two points to create the crank (from ground pivot 1)
5. Select the **+ Body** tool, click two points for the conrod
6. Right-click an attachment point > **Create Joint > Revolute** to connect crank to ground
7. Right-click an attachment point > **Create Joint > Revolute** to connect crank to conrod
8. Right-click an attachment point > **Create Joint > Prismatic** to connect conrod to ground rail
9. Right-click a grounded revolute joint > **Set as Driver**
10. Verify: status bar shows DOF: 0 and green convergence dot
11. Play animation — slider should move back and forth

**PASS criteria:**
- [ ] Mechanism solves at all crank angles (no red dot)
- [ ] Animation shows expected slider-crank motion
- [ ] Undo (Ctrl+Z) can step back through every edit

---

## Test 9 — Diagnostics and Edge Cases

**Goal:** Verify the tool handles non-ideal mechanisms gracefully.

**Check 9.1 — Double-Rocker (limited rotation):**
1. File > Load Sample > **Double-Rocker (5-3-4-7)**
2. The crank cannot make a full 360 — sweep should solve partially
3. Status bar should show solver failures (red dot) at angles beyond the mechanism's range
4. Plots should show data only for valid angles

**Check 9.2 — Parallelogram (redundant constraints):**
1. File > Load Sample > **Parallelogram (4-2-4-2)**
2. Diagnostics should warn about conditioning (high condition number)
3. The mechanism should still solve and animate correctly

**Check 9.3 — Remove a joint (underconstrained):**
1. Load any 4-bar sample
2. Select a joint, press Delete
3. Status bar should show DOF > 0 warning
4. The solver should fail gracefully (no crash)

**PASS criteria:**
- [ ] Double-rocker shows partial sweep, no crash
- [ ] Parallelogram shows conditioning warning
- [ ] Removing a joint shows DOF warning, no crash

---

## Test 10 — Transmission Angle and Mechanical Advantage

**Goal:** Verify engineering analysis outputs.

**Setup:**
1. Load the 4-bar crank-rocker

**Check 10.1 — Transmission angle plot:**
1. Open the **Transmission Angle** plot tab
2. For a Grashof crank-rocker, transmission angle should stay between ~40 deg and ~140 deg
3. Minimum transmission angle should occur near the toggle positions
4. The curve should be smooth and periodic

**Check 10.2 — Mechanical advantage:**
1. Open the **Mechanical Advantage** plot tab
2. MA should be finite everywhere except near toggle (where it spikes)
3. MA = 0 where the output link is at a dead point

**Check 10.3 — Torque envelopes (Diagnostics panel):**
1. Expand the Diagnostics section
2. Verify envelope stats are shown: min torque, max torque, RMS torque
3. These should be consistent with the torque plot visual extremes

**PASS criteria:**
- [ ] Transmission angle stays in physically reasonable range (no negative values, no > 180 deg)
- [ ] MA spikes only near toggle points
- [ ] Envelope stats match visual plot inspection

---

## Validation Summary

After completing all tests, fill in this summary:

| Test | Result | Notes |
|------|--------|-------|
| 1. 4-bar kinematics | PASS / FAIL | |
| 2. 4-bar statics | PASS / FAIL | |
| 3. Slider-crank (prismatic) | PASS / FAIL | |
| 4. Inverse dynamics | PASS / FAIL | |
| 5. Forward dynamics (energy) | PASS / FAIL | |
| 6. Unit conversion | PASS / FAIL | |
| 7. Save/load round-trip | PASS / FAIL | |
| 8. Build from scratch | PASS / FAIL | |
| 9. Edge cases | PASS / FAIL | |
| 10. Engineering outputs | PASS / FAIL | |

**Tests 1-3 are critical.** If those pass, the core solver and GUI pipeline are working correctly. Tests 4-10 validate secondary features and edge cases.

**If any test fails:** File an issue with the test number, the expected value, and the actual value you observed. Include a screenshot if possible.
