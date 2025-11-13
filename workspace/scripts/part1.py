#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator, Isometry3d, Translation3d, CollisionMarginData
from tesseract_robotics.tesseract_environment import Environment

from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics.tesseract_environment import AddLinkCommand



def add_sphere_obstacles(env, name, center, radius=0.2, parent_link="panda_link0"):
    
    # Create a sphere link and attach it to the father link with a fixed joint
    geometry = Sphere(radius)

    vision = Visual()
    vision.geometry = geometry
    collision = Collision()
    collision.geometry = geometry

    link = Link(name)
    link.visual.push_back(vision)
    link.collision.push_back(collision)

    joint = Joint(f"{name}_joint")
    joint.type = JointType_FIXED
    joint.parent_link_name = parent_link
    joint.child_link_name = name

    transformation = Isometry3d.Identity() * Translation3d(center[0], center[1], center[2])
    joint.parent_to_joint_origin_transform = transformation

    env.applyCommand(AddLinkCommand(link, joint))


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

    obstacles = [
    (0.55, 0.0, 0.25), (0.35, 0.35, 0.25), (0.0, 0.55, 0.25),
        (-0.55, 0.0, 0.25), (-0.35, -0.35, 0.25), (0.0, -0.55, 0.25),
        (0.35, -0.35, 0.8), (0.35, 0.35, 0.8), (0.0, 0.55, 0.8),
        (-0.35, 0.35, 0.8), (-0.55, 0.0, 0.8), (-0.35, -0.35, 0.8),
        (0.0, -0.55, 0.8)
    ]

    for i, center in enumerate(obstacles):
        add_sphere_obstacles(env, f"sphere_{i}", center, radius=0.2)

    print(f"[OK] Added {len(obstacles)} sphere obstacles.")

#    # Get the state solver. This must be called again after environment is updated
#     solver = env.getStateSolver()

#     # Get the discrete contact manager. This must be called again after the environment is updated
#     manager = env.getDiscreteContactManager()
#     manager.setActiveCollisionObjects(env.getActiveLinkNames())

#     #set the collision margin for check. Objects with closer than the specified margin will be returned
#     margin_data = CollisionMarginData(0.1) #10cm margin
#     manager.setCollisionMarginData(margin_data)

if __name__ == "__main__":
    main()
