# Importing from SolidWorks

This guide explains how to transfer a planar linkage mechanism from SolidWorks into the Linkage Simulator.

---

## Method 1: Image Trace (Fastest)

1. Screenshot your SolidWorks assembly (front view, orthographic)
2. In the simulator: **Image > Import** (desktop) or drag-drop the image onto the canvas (web)
3. Use **Image Settings** to scale the image to match real dimensions (set the width in mm)
4. Draw links on top of the image using the **Draw Link** tool
5. Place ground pivots at fixed joint locations using **+ Ground**

---

## Method 2: Manual JSON File

Create a `.json` file matching the schema below, then open it in the simulator (File > Open JSON).

### JSON Schema

All units are **SI**: meters, radians, kg, N.

```json
{
  "schema_version": "1.1.0",

  "bodies": {
    "ground": {
      "attachment_points": {
        "O1": [0.0, 0.0],
        "O2": [0.076, 0.0]
      },
      "mass": 0.0,
      "cg_local": [0.0, 0.0],
      "izz_cg": 0.0,
      "label": "ground"
    },
    "crank": {
      "attachment_points": {
        "A": [0.0, 0.0],
        "B": [0.044, 0.0]
      },
      "mass": 1.0,
      "cg_local": [0.022, 0.0],
      "izz_cg": 0.000161,
      "label": "crank"
    },
    "coupler": {
      "attachment_points": {
        "B": [0.0, 0.0],
        "C": [0.092, 0.0]
      },
      "mass": 1.0,
      "cg_local": [0.046, 0.0],
      "izz_cg": 0.000706,
      "label": "coupler"
    },
    "rocker": {
      "attachment_points": {
        "C": [0.0, 0.0],
        "D": [0.092, 0.0]
      },
      "mass": 1.0,
      "cg_local": [0.046, 0.0],
      "izz_cg": 0.000706,
      "label": "rocker"
    }
  },

  "joints": {
    "J1": {
      "Revolute": {
        "body_i": "ground",
        "point_i": "O1",
        "body_j": "crank",
        "point_j": "A"
      }
    },
    "J2": {
      "Revolute": {
        "body_i": "crank",
        "point_i": "B",
        "body_j": "coupler",
        "point_j": "B"
      }
    },
    "J3": {
      "Revolute": {
        "body_i": "coupler",
        "point_i": "C",
        "body_j": "rocker",
        "point_j": "C"
      }
    },
    "J4": {
      "Revolute": {
        "body_i": "rocker",
        "point_i": "D",
        "body_j": "ground",
        "point_j": "O2"
      }
    }
  },

  "drivers": {
    "D1": {
      "ConstantSpeed": {
        "body_i": "ground",
        "body_j": "crank",
        "omega": 6.283185,
        "theta_0": 0.0
      }
    }
  },

  "forces": [],
  "mounting_angle": 0.0
}
```

### Field Reference

| Field | Description |
|---|---|
| `schema_version` | Always `"1.1.0"` |
| `bodies` | Map of body ID to body definition. Must include `"ground"`. |
| `attachment_points` | Body-local coordinates `[x, y]` in meters. `[0, 0]` = body origin. |
| `mass` | Body mass in kg. Ground = 0. Use 1.0 as default for moving bodies. |
| `cg_local` | Center of gravity in body-local coords. For a bar: `[length/2, 0]`. |
| `izz_cg` | Moment of inertia about CG (kg*m^2). For uniform rod: `mass * length^2 / 12`. |
| `label` | Display name (optional, defaults to body ID). |
| `joints` | Map of joint ID to joint definition. |
| `Revolute` | Pin joint: `body_i/point_i` connects to `body_j/point_j`. |
| `Prismatic` | Sliding joint: includes `axis` vector and `offset`. |
| `Fixed` | Rigid connection (0 DOF between bodies). |
| `drivers` | Map of driver ID to driver definition. |
| `ConstantSpeed` | Rotates at `omega` rad/s from `theta_0` rad. `omega=6.283` = 1 rev/s. |
| `forces` | Array of force elements (springs, dampers, gravity, etc.). Optional. |
| `mounting_angle` | Mechanism tilt relative to gravity in radians. Default 0. |

### Key Rules

- **Ground body** must be named `"ground"` with `mass: 0`.
- Ground attachment points are in **world coordinates** (fixed in space).
- Moving body attachment points are in **body-local coordinates** (relative to body origin).
- Same point name on different bodies is fine (e.g., crank "B" and coupler "B").
- A body with 3+ attachment points is automatically a ternary link.
- Only ONE driver needed. Place it between ground and the input link.
- Forces, mass, and coupler points are optional — the mechanism will solve without them.

### Converting SolidWorks Dimensions

1. In SolidWorks, use **Measure** tool to get joint-to-joint distances
2. Divide all mm values by 1000 to get meters
3. Each link becomes a body with attachment points at each joint location
4. Fixed joints (ground mounts) become attachment points on the `"ground"` body
5. For a bar from joint A to joint B: `"A": [0, 0], "B": [length_m, 0]`
6. CG = `[length_m / 2, 0]`, Izz = `mass * length_m^2 / 12`

### Ternary Links (3+ joints on one body)

If a SolidWorks part has 3 pivot holes:

```json
"bellcrank": {
  "attachment_points": {
    "P1": [0.0, 0.0],
    "P2": [0.05, 0.0],
    "P3": [0.025, 0.03]
  },
  "mass": 1.0,
  "cg_local": [0.025, 0.01],
  "izz_cg": 0.001
}
```

P1 is the body origin. P2 and P3 are relative to P1 in the body's local frame.

---

## Method 3: SolidWorks VBA Macro (Advanced)

A macro can automate the export. The general approach:

1. Iterate through assembly mates (concentric mates = revolute joints)
2. For each mate, extract the pivot point coordinates
3. Project all coordinates onto the mechanism plane (e.g., Front Plane)
4. Group pivot points by component to identify bodies
5. Write the JSON file

### Macro Pseudocode

```vb
Sub ExportLinkage()
    Dim swApp As SldWorks.SldWorks
    Dim swModel As SldWorks.ModelDoc2
    Dim swAsm As SldWorks.AssemblyDoc

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc
    Set swAsm = swModel

    ' Collect all concentric mates -> joint locations
    Dim feat As SldWorks.Feature
    Set feat = swModel.FirstFeature

    Do While Not feat Is Nothing
        If feat.GetTypeName2 = "MateGroup" Then
            Dim subfeat As SldWorks.Feature
            Set subfeat = feat.GetFirstSubFeature
            Do While Not subfeat Is Nothing
                If subfeat.GetTypeName2 = "MateConcentric" Then
                    ' Extract mate entities -> get pivot point
                    ' Extract component names -> body IDs
                    ' Store: body_a, body_b, x, y coordinates
                End If
                Set subfeat = subfeat.GetNextSubFeature
            Loop
        End If
        Set feat = feat.GetNextFeature
    Loop

    ' Write JSON file from collected data
    ' ... (format as shown in schema above)
End Sub
```

### Tips for the Macro

- Use `IMate2.MateEntity` to get the cylindrical faces
- Use `ICylindricalSurfaceParams.Origin` for pivot point coordinates
- Project 3D coordinates to 2D: pick the plane your mechanism operates in
- Name bodies after SolidWorks component names (strip instance numbers)
- The first component touching the assembly origin becomes the ground body
- Set the first grounded revolute as the driver joint

---

## After Import

Once loaded in the simulator:
1. Set a **driver** — right-click a grounded revolute joint > "Set as Driver"
2. Click **Play** to animate
3. Adjust **gravity** if needed (default 9.81 m/s^2 downward)
4. Add **forces** (springs, dampers) via the Force Toolbar
5. Check the **Health Report** in the sidebar for warnings
