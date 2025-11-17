"""Minimal TrajOpt example for Panda using only Cartesian waypoints.

This mirrors the official no-composer example style: build a composite program
with Cartesian poses, densify it with generateInterpolatedProgram, optimize with
TrajOpt, then time-parameterize and visualize.
"""
from pathlib import Path
import numpy as np
import os
import time
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
    AddLinkCommand,
)
from tesseract_robotics.tesseract_scene_graph import (
    Joint,
    Link,
    Visual,
    Collision,
    JointType_FIXED,
)
from tesseract_robotics.tesseract_geometry import (
    Sphere
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
    WaypointPoly_as_JointWaypointPoly,
    WaypointPoly_as_CartesianWaypointPoly,
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
    (0.55, 0.0, 0.25), 
    (0.35, 0.35, 0.25), 
    (0.0, 0.55, 0.25),
    (-0.55, 0.0, 0.25), 
    (-0.35, -0.35, 0.25), 
    (0.0, -0.55, 0.25),
    (0.35, -0.35, 0.8), 
    (0.35, 0.35, 0.8), 
    (0.0, 0.55, 0.8),
    (-0.35, 0.35, 0.8), 
    (-0.55, 0.0, 0.8), 
    (-0.35, -0.35, 0.8),
    (0.0, -0.55, 0.8), 
    (0.35, -0.35, 0.8)
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


def add_spheres(env: Environment, centers, radius: float = 0.2, parent_link: str = "panda_link0", name_prefix: str = "sphere_") -> None:
    """Attach fixed collision spheres to emulate simple obstacles."""
    if isinstance(centers, tuple) and len(centers) == 3:
        centers = [centers]

    for idx, center in enumerate(centers):
        geometry = Sphere(radius)

        visual = Visual()
        visual.geometry = geometry
        collision = Collision()
        collision.geometry = geometry

        link_name = f"{name_prefix}{idx}"
        link = Link(link_name)
        link.visual.push_back(visual)
        link.collision.push_back(collision)

        joint = Joint(f"{link_name}_joint")
        joint.type = JointType_FIXED
        joint.parent_link_name = parent_link
        joint.child_link_name = link_name
        joint.parent_to_joint_origin_transform = (
            Isometry3d.Identity() * Translation3d(center[0], center[1], center[2])
        )

        env.applyCommand(AddLinkCommand(link, joint))

def build_cartesian_program(manip_info: ManipulatorInfo, joint_names: list[str], joint_path: list[np.ndarray]) -> CompositeInstruction:
    # Env2
    # wp_start = CartesianWaypoint(
    #     Isometry3d.Identity()
    #     * Translation3d(0.3, 0.0, 0.55)
    #     * Quaterniond(1, 0, 0, 0)
    # )
    # wp_mid = CartesianWaypoint(
    #     Isometry3d.Identity()
    #     * Translation3d(0.35, 0.0, 0.6)
    #     * Quaterniond(1, 0, 0, 0)
    # )
    # wp_goal = CartesianWaypoint(
    #     Isometry3d.Identity()
    #     * Translation3d(0.2, 0.05, 0.65)
    #     * Quaterniond(1, 0, 0, 0)
    # )
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    for q in joint_path:
        wp = JointWaypoint(joint_names, q.astype(np.float64))
        instr = MoveInstruction(
            JointWaypointPoly_wrap_JointWaypoint(wp),
            MoveInstructionType_FREESPACE,
            "DEFAULT",
        )
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr))
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


def _collect_joint_positions(instructions: CompositeInstruction) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for instr in instructions.flatten():
        if not instr.isMoveInstruction():
            continue

        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp_poly = move.getWaypoint()

        if not wp_poly.isStateWaypoint():
            continue

        state_wp = WaypointPoly_as_StateWaypointPoly(wp_poly)
        positions.append(np.asarray(state_wp.getPosition(), dtype=np.float64).flatten())
    return positions

def _collect_cartesian_positions(env: Environment, instructions: CompositeInstruction, joint_names: list[str], tcp_frame: str) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    solver = env.getStateSolver()

    for instr in instructions.flatten():
        if not instr.isMoveInstruction():
            continue

        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp_poly = move.getWaypoint()

        if wp_poly.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp_poly)
            q = np.asarray(state_wp.getPosition(), dtype=np.float64).flatten()
            state = solver.getState(joint_names, q)
            pose = state.link_transforms[tcp_frame]
            points.append(np.asarray(pose.translation(), dtype=np.float64))
            continue

        if wp_poly.isJointWaypoint():
            joint_wp = WaypointPoly_as_JointWaypointPoly(wp_poly)
            q = np.asarray(joint_wp.getPosition(), dtype=np.float64).flatten()
            state = solver.getState(joint_names, q)
            pose = state.link_transforms[tcp_frame]
            points.append(np.asarray(pose.translation(), dtype=np.float64))
            continue

        if wp_poly.isCartesianWaypoint():
            cart_wp = WaypointPoly_as_CartesianWaypointPoly(wp_poly)
            transform = cart_wp.getTransform()
            points.append(np.asarray(transform.translation(), dtype=np.float64))
            continue

    return points

def radial_length(instructions: CompositeInstruction) -> tuple[float, bool]:
    """Compute joint-space length of the executed StateWaypoints."""
    q_list = _collect_joint_positions(instructions)

    if len(q_list) < 2:
        return 0.0, False

    length = 0.0
    for i in range(len(q_list) - 1):
        length += float(np.linalg.norm(q_list[i + 1] - q_list[i]))
    return length, True

def cartesian_path_length(env: Environment, instructions: CompositeInstruction, joint_names: list[str], tcp_frame: str) -> tuple[float, bool]:
    """TCP path length in meters using FK for state/joint waypoints or direct Cartesian poses."""
    points = _collect_cartesian_positions(env, instructions, joint_names, tcp_frame)
    if len(points) < 2:
        return 0.0, False

    length = 0.0
    for i in range(len(points) - 1):
        length += float(np.linalg.norm(points[i + 1] - points[i]))
    return length, True


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


def run_pipeline(env: Environment, manip_info: ManipulatorInfo, joint_path: list[np.ndarray], label: str) -> PlannerResponse:
    env.setState(JOINT_NAMES, joint_path[0])
    state = env.getStateSolver().getState(JOINT_NAMES, joint_path[0])
    tcp_pose = state.link_transforms["panda_link8"]
    print("TCP xyz:", tcp_pose.translation().flatten())

    cart_program = build_cartesian_program(manip_info, JOINT_NAMES, joint_path)

    interp_start = time.perf_counter()
    interpolated_program = generateInterpolatedProgram(
        cart_program,
        env,
        3.14,
        1.0,
        3.14,
        15,
    )
    interp_duration = time.perf_counter() - interp_start
    interp_flat = interpolated_program.flatten()
    print(f"[{label}] Interpolated program instructions: {len(interp_flat)}")

    opt_start = time.perf_counter()
    optimized = optimize_with_trajopt(env, interpolated_program)
    opt_duration = time.perf_counter() - opt_start
    print(f"[{label}] TrajOpt success: {len(optimized.results)} instructions")

    tp_start = time.perf_counter()
    time_parameterize(optimized.results)
    tp_duration = time.perf_counter() - tp_start
    print(f"[{label}] timed waypoints:")
    print_waypoints(optimized.results)

    interp_length, interp_has_joint = radial_length(interpolated_program)
    optimized_length, opt_has_joint = radial_length(optimized.results)
    tcp_frame = manip_info.tcp_frame or manip_info.working_frame or JOINT_NAMES[-1]
    interp_cart, interp_has_cart = cartesian_path_length(env, interpolated_program, JOINT_NAMES, tcp_frame)
    optimized_cart, opt_has_cart = cartesian_path_length(env, optimized.results, JOINT_NAMES, tcp_frame)
    total_time = interp_duration + opt_duration + tp_duration
    print(
        f"[{label}] timing: interp {interp_duration:.3f}s | trajopt {opt_duration:.3f}s | totg {tp_duration:.3f}s | total {total_time:.3f}s"
    )
    joint_interp_str = f"{interp_length:.3f} rad" if interp_has_joint else "n/a (Cartesian seed)"
    joint_opt_str = f"{optimized_length:.3f} rad" if opt_has_joint else "n/a"
    print(f"[{label}] joint lengths: interpolated {joint_interp_str} | optimized {joint_opt_str}")
    cart_interp_str = f"{interp_cart:.3f} m" if interp_has_cart else "n/a"
    cart_opt_str = f"{optimized_cart:.3f} m" if opt_has_cart else "n/a"
    print(f"[{label}] Cartesian lengths: interpolated {cart_interp_str} | optimized {cart_opt_str}")

    return optimized


def main() -> None:
    rewrite_urdf_mesh_paths(URDF_PATH, URDF_RESOLVED_PATH)
    locator = GeneralResourceLocator()

    env = Environment()
    assert env.init(
        FilesystemPath(URDF_RESOLVED_PATH.as_posix()),
        FilesystemPath(SRDF_PATH.as_posix()),
        locator,
    )

    env_obs = Environment()
    assert env_obs.init(
        FilesystemPath(URDF_RESOLVED_PATH.as_posix()),
        FilesystemPath(SRDF_PATH.as_posix()),
        locator,
    )
    
    configure_plugins(env)
    configure_plugins(env_obs)
    add_spheres(env_obs, OBSTACLE_SPHERES, radius=0.2)

    manip_info = ManipulatorInfo()
    manip_info.manipulator = "panda_arm"
    manip_info.tcp_frame = "panda_link8"
    manip_info.working_frame = "panda_link0"
    manip_info.manipulator_ik_solver = "KDLInvKinChainLMA"

    # start_joint = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8], dtype=np.float64)
    # start_joint = np.array([0, 0, 0, 0, 0, 1.571, 0.785], dtype=np.float64)
    # Panda joint limits (approximate)
    # joint_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    # joint_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    # np.random.seed()  

    # start_joint = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
    # mid_joint = np.random.uniform(joint_lower, joint_upper)
    # goal_joint = np.random.uniform(joint_lower, joint_upper)

    start_joint = np.array([2.8, 0.0, 0.0, 0.0, 2.2, 2.2, -0.5], dtype=np.float64)   
    mid_joint   = np.array([0.0, 0.0, 0.0, -0.5, 1.2, 1.0, 0.0], dtype=np.float64)     
    mid2_joint  = np.array([0.0, 0.0, 0.25, -1.8, -1.0, 0.5, 0.5], dtype=np.float64) 
    goal_joint  = np.array([0.0, 0.0, -0.5, -2.2, -1.0, 2.0, 1.5], dtype=np.float64)      

    joint_path = []
    joint_path.append(start_joint)
    joint_path.append(mid_joint)
    joint_path.append(mid2_joint)
    joint_path.append(goal_joint)
    free_path = run_pipeline(env, manip_info, joint_path, "Free env")
    viewer_free = TesseractViewer(server_address=('127.0.0.1',8000))
    # viewer_free = TesseractViewer(server_address=('0.0.0.0',8080))
    viewer_free.update_environment(env, [0, 0, 0])
    viewer_free.update_trajectory(free_path.results.flatten())
    viewer_free.plot_trajectory(free_path.results.flatten(), manip_info, axes_length=0.05)
    try:
        viewer_free.start_serve_background()
        print("View free environment at http://localhost:8000 on windows or http://0.0.0.0:8000 on other devices")
        input("Press Enter to continue to obstacle environment...")
    except Exception as e:
        print(f"Error starting viewer: {e}")

    obs_path = run_pipeline(env_obs, manip_info, joint_path, "Obstacle env")
    viewer_obs = TesseractViewer(server_address=('127.0.0.1',8081))
    # viewer_obs = TesseractViewer(server_address=('0.0.0.0',8081))
    viewer_obs.update_environment(env_obs, [0, 0, 0])
    viewer_obs.update_trajectory(obs_path.results.flatten())
    viewer_obs.plot_trajectory(obs_path.results.flatten(), manip_info, axes_length=0.05)

    try:
        viewer_obs.start_serve_background()
        print("View obstacle environment at http://localhost:8081 on windows or http://0.0.0.0:8081 on other devices")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Error starting viewer: {e}")




if __name__ == "__main__":
    main()
