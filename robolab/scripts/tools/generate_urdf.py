"""URDF parameterized generator for RPO robot leg length co-design.

Generates a modified URDF with scaled thigh and calf lengths, along with
proportional scaling of mass, inertia, collision geometry, and CoM position.

STANDALONE — does NOT modify any existing source files.

Usage:
    python generate_urdf.py --thigh 0.25 --calf 0.30 --output /tmp/rpo_test/
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# ── Default physical parameters (from original URDF) ────────────────────
DEFAULT_THIGH = 0.25       # m
DEFAULT_CALF  = 0.30       # m

# Joints whose origin z defines segment lengths
THIGH_JOINTS = ["left_knee_joint", "right_knee_joint"]
CALF_JOINTS  = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]

# Parent links that contain mass/inertia/collision for each segment
# thigh_pitch_joint → parent is thigh_roll_link
# knee_joint → parent is thigh_pitch_link  ← THIS is the thigh segment
# ankle_pitch_joint → parent is knee_link  ← THIS is the calf segment

MESH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "robots", "roboparty", "rpo", "meshes"
)

TEMPLATE_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "robots", "roboparty", "rpo", "urdf", "rpo.urdf"
)


def _ns(tag: str) -> str:
    """Strip namespace from XML tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_link(root: ET.Element, name: str):
    """Find a <link> element by name."""
    for elem in root.iter():
        if _ns(elem.tag) == "link" and elem.get("name") == name:
            return elem
    return None


def _find_joint(root: ET.Element, name: str):
    """Find a <joint> element by name."""
    for elem in root.iter():
        if _ns(elem.tag) == "joint" and elem.get("name") == name:
            return elem
    return None


def _scale_link(link_elem: ET.Element, scale: float):
    """In-place scale mass, inertia tensor, CoM z, and collision box z of a <link>."""

    # --- Mass ---
    inertial = link_elem.find("inertial")
    if inertial is not None:
        mass = inertial.find("mass")
        if mass is not None:
            mass.set("value", f"{float(mass.get('value', 0)) * scale:.10g}")

        # --- Inertia tensor (exact transform for z-only scaling) ---
        # Only the z-extent changes (uniform density -> mass scales x s), so
        # the tensor does NOT scale uniformly by s^3:
        #   I'xx = s*∫y²dm + s³*∫z²dm   (s³ from the length, s from the cross-section)
        #   I'yy = s*∫x²dm + s³*∫z²dm
        #   I'zz = s*Izz                (spin about the link axis has no length term)
        #   I'xy = s*Ixy,  I'xz = s²*Ixz,  I'yz = s²*Iyz
        # The uniform x s³ previously used over/underestimated Izz by ~30-45%
        # and mis-scaled the products of inertia. Second moments are recovered
        # from the original tensor:
        #   ∫y²dm = (Izz + Ixx - Iyy)/2,  ∫x²dm = (Izz - Ixx + Iyy)/2,  ∫z²dm = Ixx - ∫y²dm
        inertia = inertial.find("inertia")
        if inertia is not None:
            s, s2, s3 = scale, scale**2, scale**3
            ixx = inertia.get("ixx")
            iyy = inertia.get("iyy")
            izz = inertia.get("izz")
            if ixx is not None and iyy is not None and izz is not None:
                ixx, iyy, izz = float(ixx), float(iyy), float(izz)
                y2 = (izz + ixx - iyy) / 2.0
                x2 = (izz - ixx + iyy) / 2.0
                z2 = ixx - y2
                inertia.set("ixx", f"{s * y2 + s3 * z2:.10g}")
                inertia.set("iyy", f"{s * x2 + s3 * z2:.10g}")
                inertia.set("izz", f"{s * izz:.10g}")
                if inertia.get("ixy") is not None:
                    inertia.set("ixy", f"{s * float(inertia.get('ixy')):.10g}")
                if inertia.get("ixz") is not None:
                    inertia.set("ixz", f"{s2 * float(inertia.get('ixz')):.10g}")
                if inertia.get("iyz") is not None:
                    inertia.set("iyz", f"{s2 * float(inertia.get('iyz')):.10g}")
            else:
                # fallback: uniform s³ for incomplete tensors
                for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    v = inertia.get(attr)
                    if v is not None:
                        inertia.set(attr, f"{float(v) * s3:.10g}")

        # --- CoM z-offset ---
        origin = inertial.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            parts = [float(x) for x in xyz.split()]
            parts[2] *= scale
            origin.set("xyz", f"{parts[0]:.10g} {parts[1]:.10g} {parts[2]:.10g}")

    # --- Collision geometry (origin z and box z-size) ---
    # Origin z must scale with the segment so the geometry stays centered on
    # the scaled segment. Without it, a shortened calf keeps its box center at
    # the original -0.14 m while the ankle joint rises, so the box bottom
    # penetrates the foot mesh (self-collisions are all enabled in the RPO
    # URDF) and the robot explodes at spawn — the calf < 0.25 m "crash line".
    for coll in link_elem.findall("collision"):
        origin = coll.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            parts = [float(x) for x in origin.get("xyz").split()]
            parts[2] *= scale
            origin.set("xyz", f"{parts[0]:.10g} {parts[1]:.10g} {parts[2]:.10g}")
        geom = coll.find("geometry")
        if geom is not None:
            box = geom.find("box")
            if box is not None and "size" in box.attrib:
                parts = [float(x) for x in box.get("size").split()]
                parts[2] *= scale
                box.set("size", f"{parts[0]:.10g} {parts[1]:.10g} {parts[2]:.10g}")


def generate_urdf(
    thigh_length: float,
    calf_length: float,
    output_dir: str,
    template_path: str | None = None,
) -> str:
    """Generate parameterized RPO URDF.

    Returns:
        Absolute path to the generated URDF file.
    """
    thigh_scale = thigh_length / DEFAULT_THIGH
    calf_scale  = calf_length  / DEFAULT_CALF

    if template_path is None:
        template_path = os.path.abspath(TEMPLATE_URDF)

    # ── Output directory structure ──────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    urdf_dir = os.path.join(output_dir, "urdf")
    meshes_dst = os.path.join(output_dir, "meshes")
    os.makedirs(urdf_dir, exist_ok=True)

    # Symlink meshes (fast, no copying of large STL files)
    meshes_src = os.path.abspath(MESH_DIR)
    if os.path.exists(meshes_src) and not os.path.exists(meshes_dst):
        os.symlink(meshes_src, meshes_dst, target_is_directory=True)

    # ── Parse URDF ──────────────────────────────────────────────────────
    tree = ET.parse(template_path)
    root = tree.getroot()

    # ── 1. Modify joint origins ─────────────────────────────────────────
    for jname in THIGH_JOINTS:
        joint = _find_joint(root, jname)
        if joint is None:
            print(f"[WARN] joint '{jname}' not found")
            continue
        origin = joint.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            parts = [float(x) for x in origin.get("xyz").split()]
            parts[2] = -thigh_length
            origin.set("xyz", f"{parts[0]:.10g} {parts[1]:.10g} {parts[2]:.10g}")

    for jname in CALF_JOINTS:
        joint = _find_joint(root, jname)
        if joint is None:
            print(f"[WARN] joint '{jname}' not found")
            continue
        origin = joint.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            parts = [float(x) for x in origin.get("xyz").split()]
            parts[2] = -calf_length
            origin.set("xyz", f"{parts[0]:.10g} {parts[1]:.10g} {parts[2]:.10g}")

    # ── 2. Scale link properties ────────────────────────────────────────
    # Thigh segment = parent of knee_joint -> thigh_pitch_link
    for jname in THIGH_JOINTS:
        joint = _find_joint(root, jname)
        if joint is None:
            continue
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            continue
        link = _find_link(root, parent.get("link"))
        if link is not None:
            _scale_link(link, thigh_scale)
            print(f"  scaled THIGH link '{parent.get('link')}' ×{thigh_scale:.3f}")

    # Calf segment = parent of ankle_pitch_joint -> knee_link
    for jname in CALF_JOINTS:
        joint = _find_joint(root, jname)
        if joint is None:
            continue
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            continue
        link = _find_link(root, parent.get("link"))
        if link is not None:
            _scale_link(link, calf_scale)
            print(f"  scaled CALF link '{parent.get('link')}' ×{calf_scale:.3f}")

    # ── Write ───────────────────────────────────────────────────────────
    urdf_path = os.path.join(urdf_dir, "rpo.urdf")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"[generate_urdf] → {urdf_path}")
    print(f"[generate_urdf]   thigh={thigh_length:.4f}m  calf={calf_length:.4f}m")
    return urdf_path


# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thigh", type=float, required=True)
    parser.add_argument("--calf", type=float, required=True)
    parser.add_argument("--output", type=str, default="/tmp/rpo_urdf_co_design")
    args = parser.parse_args()

    path = generate_urdf(args.thigh, args.calf, args.output)
    print(f"SUCCESS: {path}")
