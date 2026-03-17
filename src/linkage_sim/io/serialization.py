"""JSON serialization and deserialization of mechanisms.

File format follows the schema defined in ARCHITECTURE.md.
All values are in SI units. Schema versioning supports forward migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from linkage_sim.core.bodies import Body
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID

SCHEMA_VERSION = "1.0.0"


def mechanism_to_dict(mechanism: Mechanism) -> dict[str, Any]:
    """Serialize a Mechanism to a JSON-compatible dict.

    Args:
        mechanism: A built or unbuilt Mechanism instance.

    Returns:
        Dictionary matching the JSON schema in ARCHITECTURE.md.
    """
    bodies_dict: dict[str, Any] = {}
    for body_id, body in mechanism.bodies.items():
        body_data: dict[str, Any] = {
            "attachment_points": {
                name: pt.tolist() for name, pt in body.attachment_points.items()
            },
            "mass": body.mass,
            "cg_local": body.cg_local.tolist(),
            "Izz_cg": body.Izz_cg,
        }
        if body.coupler_points:
            body_data["coupler_points"] = {
                name: pt.tolist() for name, pt in body.coupler_points.items()
            }
        bodies_dict[body_id] = body_data

    joints_dict: dict[str, Any] = {}
    for joint in mechanism.joints:
        joint_data: dict[str, Any] = {
            "body_i": joint.body_i_id,
            "body_j": joint.body_j_id,
        }

        # Determine joint type and type-specific fields
        from linkage_sim.core.constraints import FixedJoint, RevoluteJoint
        from linkage_sim.core.drivers import RevoluteDriver

        if isinstance(joint, RevoluteJoint):
            joint_data["type"] = "revolute"
            joint_data["point_i"] = joint._point_i_name
            joint_data["point_j"] = joint._point_j_name
        elif isinstance(joint, FixedJoint):
            joint_data["type"] = "fixed"
            joint_data["point_i"] = joint._point_i_name
            joint_data["point_j"] = joint._point_j_name
            joint_data["delta_theta_0"] = joint._delta_theta_0
        elif isinstance(joint, RevoluteDriver):
            joint_data["type"] = "revolute_driver"
            # Note: callable functions can't be serialized.
            # We store a marker; the user must re-attach the driver
            # function on load.
            joint_data["note"] = "driver function not serializable"
        else:
            joint_data["type"] = "unknown"

        joints_dict[joint.id] = joint_data

    return {
        "schema_version": SCHEMA_VERSION,
        "bodies": bodies_dict,
        "joints": joints_dict,
    }


def dict_to_mechanism(data: dict[str, Any]) -> Mechanism:
    """Deserialize a Mechanism from a JSON-compatible dict.

    Rebuilds bodies and geometric joints (revolute, fixed).
    Driver constraints are NOT restored (functions aren't serializable);
    the user must re-add them after loading.

    Args:
        data: Dictionary matching the JSON schema.

    Returns:
        A built Mechanism instance.

    Raises:
        ValueError: If schema_version is unsupported.
    """
    version = data.get("schema_version", "unknown")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version '{version}'. Expected '{SCHEMA_VERSION}'."
        )

    mech = Mechanism()

    # Rebuild bodies
    for body_id, body_data in data["bodies"].items():
        pts = {
            name: np.array(coords, dtype=np.float64)
            for name, coords in body_data["attachment_points"].items()
        }
        cg = np.array(body_data.get("cg_local", [0.0, 0.0]), dtype=np.float64)

        if body_id == GROUND_ID:
            body = Body(
                id=GROUND_ID,
                attachment_points=pts,
                mass=0.0,
                cg_local=cg,
                Izz_cg=0.0,
            )
        else:
            body = Body(
                id=body_id,
                attachment_points=pts,
                mass=body_data.get("mass", 0.0),
                cg_local=cg,
                Izz_cg=body_data.get("Izz_cg", 0.0),
            )

        # Restore coupler points
        for cp_name, cp_coords in body_data.get("coupler_points", {}).items():
            body.coupler_points[cp_name] = np.array(cp_coords, dtype=np.float64)

        mech.add_body(body)

    # Rebuild joints (geometric constraints only)
    for joint_id, joint_data in data["joints"].items():
        joint_type = joint_data["type"]

        if joint_type == "revolute":
            mech.add_revolute_joint(
                joint_id,
                joint_data["body_i"],
                joint_data["point_i"],
                joint_data["body_j"],
                joint_data["point_j"],
            )
        elif joint_type == "fixed":
            mech.add_fixed_joint(
                joint_id,
                joint_data["body_i"],
                joint_data["point_i"],
                joint_data["body_j"],
                joint_data["point_j"],
                delta_theta_0=joint_data.get("delta_theta_0", 0.0),
            )
        elif joint_type == "revolute_driver":
            # Drivers can't be deserialized (functions aren't serializable).
            # Skip and let the user re-add them.
            pass
        else:
            raise ValueError(f"Unknown joint type '{joint_type}' for '{joint_id}'.")

    mech.build()
    return mech


def save_mechanism(mechanism: Mechanism, path: str | Path) -> None:
    """Save a Mechanism to a JSON file.

    Args:
        mechanism: The mechanism to save.
        path: File path to write to.
    """
    data = mechanism_to_dict(mechanism)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_mechanism(path: str | Path) -> Mechanism:
    """Load a Mechanism from a JSON file.

    Driver constraints are NOT restored — they must be re-added
    by the user after loading.

    Args:
        path: File path to read from.

    Returns:
        A built Mechanism instance (without drivers).
    """
    with open(path) as f:
        data = json.load(f)
    return dict_to_mechanism(data)
