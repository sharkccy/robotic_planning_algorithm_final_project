"""Minimal TrajOpt example for Panda using only Cartesian waypoints.

This mirrors the official no-composer example style: build a composite program
with Cartesian poses, densify it with generateInterpolatedProgram, optimize with
TrajOpt, then time-parameterize and visualize.
"""
from pathlib import Path
import numpy as np
import tesseract_robotics

from tesseract_robotics.tesseract_common import (
    FilesystemPath,
    GeneralResourceLocator,
    Isometry3d,
    Translation3d,
    Quaterniond,
    ManipulatorInfo,
)
from tesseract_robotics.tesseract_environment import (
    Environment,
    AddContactManagersPluginInfoCommand,
    AddKinematicsInformationCommand,
)
from tesseract_robotics.tesseract_command_language import (
    CartesianWaypoint,
    MoveInstructionType_FREESPACE,
    MoveInstruction,
    CompositeInstruction,
    ProfileDictionary,
    CartesianWaypointPoly_wrap_CartesianWaypoint,
    MoveInstructionPoly_wrap_MoveInstruction,
    InstructionPoly_as_MoveInstructionPoly,
    WaypointPoly_as_StateWaypointPoly,
    JointWaypoint,
    JointWaypointPoly_wrap_JointWaypoint,
)
from tesseract_robotics.tesseract_motion_planners import PlannerRequest, PlannerResponse
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
from tesseract_robotics.tesseract_motion_planners_trajopt import (
    TrajOptDefaultPlanProfile,
    TrajOptDefaultCompositeProfile,
    TrajOptMotionPlanner,
)
from tesseract_robotics.tesseract_time_parameterization import (
    TimeOptimalTrajectoryGeneration,
    InstructionsTrajectory,
)
from tesseract_robotics_viewer import TesseractViewer
from tesseract_robotics.tesseract_srdf import (
    parseContactManagersPluginConfigString,
    parseKinematicsPluginConfigString,
)

ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "resources" / "panda" / "panda.urdf"
SRDF_PATH = ROOT / "resources" / "panda" / "panda_with_hand_links.srdf"
URDF_RESOLVED_PATH = URDF_PATH.with_name(URDF_PATH.stem + "_resolved.urdf")
MESH_ROOT = (URDF_PATH.parent / "meshes").as_posix()
CONTACT_PLUGIN_PATH = ROOT / "config" / "contact_plugins.yaml"
KINEMATICS_PLUGIN_PATH = ROOT / "config" / "kinematics_plugins.yaml"
TRAJOPT_NS = "TrajOptMotionPlannerTask"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

PANDA_MAX_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175], dtype=np.float64)
PANDA_MAX_ACC = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PANDA_MAX_JERK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
OBSTACLE_SPHERES = [
    (0.55, 0.0, 0.25), (0.35, 0.35, 0.25), (0.0, 0.55, 0.25),
    (-0.55, 0.0, 0.25), (-0.35, -0.35, 0.25), (0.0, -0.55, 0.25),
    (0.35, -0.35, 0.8), (0.35, 0.35, 0.8), (0.0, 0.55, 0.8),
    (-0.35, 0.35, 0.8), (-0.55, 0.0, 0.8), (-0.35, -0.35, 0.8),
    (0.0, -0.55, 0.8), (0.35, -0.35, 0.8)
]

def _load_yaml_template(path: Path, **fmt: str) -> str:
    text = path.read_text(encoding="utf-8")
    return text.format(**fmt)


def configure_plugins(env: Environment) -> None:
    libs_dir = Path(tesseract_robotics.__file__).resolve().parent.parent / "tesseract_robotics.libs"
    if not libs_dir.exists():
        raise FileNotFoundError(f"Cannot locate tesseract plugin directory: {libs_dir}")

    contact_yaml = _load_yaml_template(CONTACT_PLUGIN_PATH, LIB_DIR=libs_dir.as_posix())
    contact_info = parseContactManagersPluginConfigString(contact_yaml)
    env.applyCommand(AddContactManagersPluginInfoCommand(contact_info))

    kin_yaml = _load_yaml_template(KINEMATICS_PLUGIN_PATH, LIB_DIR=libs_dir.as_posix())
    kin_info = env.getKinematicsInformation()
    kin_info.kinematics_plugin_info = parseKinematicsPluginConfigString(kin_yaml)
    env.applyCommand(AddKinematicsInformationCommand(kin_info))


def rewrite_urdf_mesh_paths(src: Path, dst: Path, pkg_prefix: str = "package://meshes/") -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace(pkg_prefix, MESH_ROOT + "/")
    dst.write_text(text, encoding="utf-8")




def build_cartesian_program(manip_info: ManipulatorInfo, joint_names: list[str], start_joint: np.ndarray) -> CompositeInstruction:
    start_state = JointWaypoint(joint_names, start_joint.astype(np.float64))
    wp_start = CartesianWaypoint(
        Isometry3d.Identity()
        * Translation3d(0.4, 0.0, 0.4)
        * Quaterniond(1, 0, 0, 0)
    )
    wp_mid = CartesianWaypoint(
        Isometry3d.Identity()
        * Translation3d(0.6, 0.0, 0.7)
        * Quaterniond(1, 0, 0, 0)
    )
    wp_goal = CartesianWaypoint(
        Isometry3d.Identity()
        * Translation3d(0.6, 0.2, 0.5)
        * Quaterniond(1, 0, 0, 0)
    )

    instr_joint = MoveInstruction(
        JointWaypointPoly_wrap_JointWaypoint(start_state),
        MoveInstructionType_FREESPACE,
        "DEFAULT",
    )
    instr_start = MoveInstruction(
        CartesianWaypointPoly_wrap_CartesianWaypoint(wp_start),
        MoveInstructionType_FREESPACE,
        "DEFAULT",
    )
    instr_mid = MoveInstruction(
        CartesianWaypointPoly_wrap_CartesianWaypoint(wp_mid),
        MoveInstructionType_FREESPACE,
        "DEFAULT",
    )
    instr_goal = MoveInstruction(
        CartesianWaypointPoly_wrap_CartesianWaypoint(wp_goal),
        MoveInstructionType_FREESPACE,
        "DEFAULT",
    )

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_joint))
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_start))
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_mid))
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_goal))
    return program


def optimize_with_trajopt(env: Environment, instructions: CompositeInstruction) -> PlannerResponse:
    plan_profile = TrajOptDefaultPlanProfile()
    composite_profile = TrajOptDefaultCompositeProfile()

    profiles = ProfileDictionary()
    profiles.addProfile(TRAJOPT_NS, "DEFAULT", plan_profile)
    profiles.addProfile(TRAJOPT_NS, "DEFAULT", composite_profile)

    request = PlannerRequest()
    request.instructions = instructions
    request.env = env
    request.profiles = profiles

    planner = TrajOptMotionPlanner(TRAJOPT_NS)
    response = planner.solve(request)
    
    if not response.successful:
        raise RuntimeError("TrajOpt optimization failed")
    else:
        print("TrajOpt result instructions:", len(response.results.flatten()))
        return response


def _symmetric_limits(limits: np.ndarray) -> np.ndarray:
    column = limits.astype(np.float64).reshape(-1, 1)
    return np.hstack((-column, column))


def time_parameterize(instructions: CompositeInstruction) -> None:
    trajectory = InstructionsTrajectory(instructions)
    totg = TimeOptimalTrajectoryGeneration()

    vel_limits = _symmetric_limits(PANDA_MAX_VEL)
    acc_limits = _symmetric_limits(PANDA_MAX_ACC)
    jerk_limits = _symmetric_limits(PANDA_MAX_JERK)

    if not totg.compute(trajectory, vel_limits, acc_limits, jerk_limits):
        raise RuntimeError("Time parameterization failed")


def print_waypoints(instructions: CompositeInstruction) -> None:
    for instr in instructions.flatten():
        if not instr.isMoveInstruction():
            continue
        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move.getWaypoint()
        if not wp.isStateWaypoint():
            continue
        state_wp = WaypointPoly_as_StateWaypointPoly(wp)
        pos = state_wp.getPosition().flatten()
        print(f"Joint: {pos}  t={state_wp.getTime():.4f}")


def main() -> None:
    rewrite_urdf_mesh_paths(URDF_PATH, URDF_RESOLVED_PATH)
    locator = GeneralResourceLocator()

    env = Environment()
    assert env.init(
        FilesystemPath(URDF_RESOLVED_PATH.as_posix()),
        FilesystemPath(SRDF_PATH.as_posix()),
        locator,
    )

    configure_plugins(env)

    manip_info = ManipulatorInfo()
    manip_info.manipulator = "panda_arm"
    manip_info.tcp_frame = "panda_link8"
    manip_info.working_frame = "panda_link0"
    manip_info.manipulator_ik_solver = "KDLInvKinChainLMA"

    start_joint = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8], dtype=np.float64)
    env.setState(JOINT_NAMES, start_joint)

    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.start_serve_background()

    cart_program = build_cartesian_program(manip_info, JOINT_NAMES, start_joint)

    interpolated_program = generateInterpolatedProgram(
        cart_program,
        env,
        3.14,
        1.0,
        3.14,
        15,
    )
    interp_flat = interpolated_program.flatten()
    print(f"Interpolated program instructions: {len(interp_flat)}")
    for idx, instr in enumerate(interp_flat[:5]):
        print("  interp", idx, instr.isMoveInstruction(), instr.getDescription())

    try:
        optimized = optimize_with_trajopt(env, interpolated_program)
        print(f"TrajOpt success: {len(optimized.results)} instructions")
        for idx, instr in enumerate(optimized.results[:5]):
            print(idx, instr.isMoveInstruction(), instr.getDescription())
    except Exception as e:
        print(f"Error occurred: {e}")
        print("===================================")

    time_parameterize(optimized.results)
    print_waypoints(optimized.results)

    viewer.update_trajectory(optimized.results)
    viewer.plot_trajectory(optimized.results, manip_info, axes_length=0.05)

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
