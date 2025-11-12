#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment

# 你的實際檔案位置（請確認）
URDF = Path(r"E:/scholar/2025 Fall/Algorithm of Robotic/hw/project5/workspace/resources/panda/panda.urdf")
SRDF = Path(r"E:/scholar/2025 Fall/Algorithm of Robotic/hw/project5/workspace/resources/panda/panda.srdf")
MESH_ROOT = (URDF.parent / "meshes").as_posix()  # .../panda/meshes

# 產生一份「已把 package://meshes/ 轉成絕對路徑」的臨時 URDF
URDF_RESOLVED = URDF.with_name(URDF.stem + "_resolved.urdf")

def rewrite_urdf_mesh_paths(src: Path, dst: Path, pkg_prefix="package://meshes/"):
    text = src.read_text(encoding="utf-8")
    # 直接替換成檔案系統路徑（不用 file://，避免 URL 編碼問題）
    text = text.replace(pkg_prefix, MESH_ROOT + "/")
    dst.write_text(text, encoding="utf-8")

def main():
    assert URDF.exists(), f"URDF not found: {URDF}"
    assert SRDF.exists(), f"SRDF not found: {SRDF}"
    assert Path(MESH_ROOT).exists(), f"Mesh folder not found: {MESH_ROOT}"

    # 1) 生成已解析路徑的 URDF
    rewrite_urdf_mesh_paths(URDF, URDF_RESOLVED)

    # 2) 官方風格：FilesystemPath + GeneralResourceLocator
    locator = GeneralResourceLocator()
    urdf_path = FilesystemPath(URDF_RESOLVED.as_posix())
    srdf_path = FilesystemPath(SRDF.as_posix())

    # 3) 初始化 Environment
    env = Environment()
    ok = env.init(urdf_path, srdf_path, locator)
    assert ok, "Environment.init(...) failed"

    # 4) 確認資訊
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
