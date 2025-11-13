#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment

from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics.tesseract_common import Isometry3d


# def add_sphere_obstacle(env, name, center, radius=0.2, )

# The actual file locations (change as needed)
URDF = Path(r"E:/scholar/2025 Fall/Algorithm of Robotic/hw/project5/workspace/resources/panda/panda.urdf")
SRDF = Path(r"E:/scholar/2025 Fall/Algorithm of Robotic/hw/project5/workspace/resources/panda/panda_with_hand_links.srdf")

MESH_ROOT = (URDF.parent / "meshes").as_posix()  # .../panda/meshes

# Generate a temporary URDF with absolute paths for meshes
URDF_RESOLVED = URDF.with_name(URDF.stem + "_resolved.urdf")

def rewrite_urdf_mesh_paths(src: Path, dst: Path, pkg_prefix="package://meshes/"):
    text = src.read_text(encoding="utf-8")
    # Directly replace with filesystem path (no file:// to avoid URL encoding issues)
    text = text.replace(pkg_prefix, MESH_ROOT + "/")
    dst.write_text(text, encoding="utf-8")

def main():
    assert URDF.exists(), f"URDF not found: {URDF}"
    assert SRDF.exists(), f"SRDF not found: {SRDF}"
    assert Path(MESH_ROOT).exists(), f"Mesh folder not found: {MESH_ROOT}"

    # 1) Generate resolved URDF
    rewrite_urdf_mesh_paths(URDF, URDF_RESOLVED)
    print(f"[OK] Generated resolved URDF: {URDF_RESOLVED}")
    print(f"     (meshes root: {MESH_ROOT})")


    # 2) FilesystemPath + GeneralResourceLocator
    locator = GeneralResourceLocator()
    urdf_path = FilesystemPath(URDF_RESOLVED.as_posix())
    srdf_path = FilesystemPath(SRDF.as_posix())

    # 3) Initialize Environment
    env = Environment()
    ok = env.init(urdf_path, srdf_path, locator)
    assert ok, "Environment.init(...) failed"

    # 4) Confirm information
    sg = env.getSceneGraph()
    try:
        n_links = sg.getLinks().size(); n_joints = sg.getJoints().size()
    except Exception:
        n_links = len(sg.getLinks()); n_joints = len(sg.getJoints())

    print("[OK] Panda environment loaded")
    print("  - Links :", n_links)
    print("  - Joints:", n_joints)
    print("  - Groups:", list(env.getGroupNames()))
    print("  - Active joints:", list(env.getActiveJointNames()))

if __name__ == "__main__":
    main()
