"""Minimal TrajOpt example for Panda using only Cartesian waypoints.

This mirrors the official no-composer example style: build a composite program
with Cartesian poses, densify it with generateInterpolatedProgram, optimize with
TrajOpt, then time-parameterize and visualize.
"""
from pathlib import Path
import numpy as np
import os
import time
from statistics import mean
import tesseract_robotics

import random
import vamp
# import pandas as pd

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
from tesseract_robotics.tesseract_srdf import (
    parseContactManagersPluginConfigString,
    parseKinematicsPluginConfigString,
)

from tesseract_robotics_viewer import TesseractViewer

ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "resources" / "panda" / "panda.urdf"
SRDF_PATH = ROOT / "resources" / "panda" / "panda_with_hand_links.srdf"
URDF_RESOLVED_PATH = URDF_PATH.with_name(URDF_PATH.stem + "_resolved.urdf")
MESH_ROOT = (URDF_PATH.parent / "meshes").as_posix()
CONTACT_PLUGIN_PATH = ROOT / "config" / "contact_plugins.yaml"
KINEMATICS_PLUGIN_PATH = ROOT / "config" / "kinematics_plugins.yaml"
TRAJOPT_NS = "TrajOptMotionPlannerTask"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

PARAMETERIZED_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175], dtype=np.float64)
PARAMETERIZED_ACC = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PARAMETERIZED_JERK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

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

def build_cartesian_program(manip_info: ManipulatorInfo, joint_names: list[str], joint_path) -> CompositeInstruction:
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    for q in joint_path:
        # print(q)
        q_arr = np.array(q, dtype=np.float64)
        if not np.isfinite(q_arr).all():
            print(f'Didn\'t work: {q}')
            continue
        wp = JointWaypoint(joint_names, q_arr)
        # wp = JointWaypoint(joint_names, q)
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

    vel_limits = _symmetric_limits(PARAMETERIZED_VEL)
    acc_limits = _symmetric_limits(PARAMETERIZED_ACC)
    jerk_limits = _symmetric_limits(PARAMETERIZED_JERK)

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


def run_pipeline(env: Environment, manip_info: ManipulatorInfo, joint_path, label: str) -> tuple[PlannerResponse, dict[str, float]]:
    env.setState(JOINT_NAMES, np.array(joint_path[0], dtype=np.float64))
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
    # print(f"[{label}] Interpolated program instructions: {len(interp_flat)}")

    # opt_start = time.perf_counter()
    optimized = optimize_with_trajopt(env, interpolated_program)
    # opt_duration = time.perf_counter() - opt_start
    # print(f"[{label}] TrajOpt success: {len(optimized.results)} instructions")

    tp_start = time.perf_counter()
    time_parameterize(optimized.results)
    tp_duration = time.perf_counter() - tp_start
    # print(f"[{label}] timed waypoints:")
    print_waypoints(optimized.results)

    interp_length, interp_has_joint = radial_length(interpolated_program)
    optimized_length, opt_has_joint = radial_length(optimized.results)
    tcp_frame = manip_info.tcp_frame or manip_info.working_frame or JOINT_NAMES[-1]
    interp_cart, interp_has_cart = cartesian_path_length(env, interpolated_program, JOINT_NAMES, tcp_frame)
    optimized_cart, opt_has_cart = cartesian_path_length(env, optimized.results, JOINT_NAMES, tcp_frame)
    # total_time = interp_duration + opt_duration + tp_duration
    # print(
    #     f"[{label}] timing: interp {interp_duration:.3f}s | trajopt {opt_duration:.3f}s | totg {tp_duration:.3f}s | total {total_time:.3f}s"
    # )
    # joint_interp_str = f"{interp_length:.3f} rad" if interp_has_joint else "n/a (Cartesian seed)"
    # joint_opt_str = f"{optimized_length:.3f} rad" if opt_has_joint else "n/a"
    # print(f"[{label}] joint lengths: interpolated {joint_interp_str} | optimized {joint_opt_str}")
    # cart_interp_str = f"{interp_cart:.3f} m" if interp_has_cart else "n/a"
    # cart_opt_str = f"{optimized_cart:.3f} m" if opt_has_cart else "n/a"
    # print(f"[{label}] Cartesian lengths: interpolated {cart_interp_str} | optimized {cart_opt_str}")

    metrics = {
        "interp_duration": interp_duration,
        # "trajopt_duration": opt_duration,
        "totg_duration": tp_duration,
        # "total_time": total_time,
        "cartesian_length": optimized_cart if opt_has_cart else float("nan"),
    }

    return optimized, metrics


def summarize_metric(records: list[dict[str, float]], key: str) -> float:
    values = [entry[key] for entry in records if not np.isnan(entry[key])]
    if not values:
        return float("nan")
    return float(mean(values))

def prepare_environment(start, goal, sphere_radius = 0.2, validate = False):
    vamp_env = vamp.Environment()

    for sphere in OBSTACLE_SPHERES:
        vamp_env.add_sphere(vamp.Sphere(sphere, sphere_radius))

    if not validate or (vamp.panda.validate(start, vamp_env) and vamp.panda.validate(goal, vamp_env)):
        return vamp_env
    else:
        raise RuntimeError('Invalid start or goal state')

def run_planner(planner_name: str, spheres, result, seed: int = 42):
    pass
    

def main(
        n_trials: int = 1,
        planner: str = 'rrtc',
        sampler_name: str = 'halton',
        **kwargs,
    ):
    rewrite_urdf_mesh_paths(URDF_PATH, URDF_RESOLVED_PATH)
    locator = GeneralResourceLocator()

    env_obs = Environment()
    assert env_obs.init(
        FilesystemPath(URDF_RESOLVED_PATH.as_posix()),
        FilesystemPath(SRDF_PATH.as_posix()),
        locator,
    )

    (vamp_module, planner_func, plan_settings, simp_settings) = vamp.configure_robot_and_planner_with_kwargs('panda', planner, **kwargs)

    sampler = getattr(vamp_module, sampler_name)()
    
    
    random.seed(0)
    np.random.seed(0)

    results = [[]] * n_trials
    spheres = [np.array(sphere) for sphere in OBSTACLE_SPHERES]   

    start = [2.8, 0.0, 0.0, 0.0, 2.2, 2.2, -0.5]   
    goal  = [0.0, 0.0, -0.5, -2.2, -1.0, 2.0, 1.5]

    for trial in range(n_trials):
        from vamp import pybullet_interface as vpb
        print(f'Trial {trial} / {n_trials}')
        vamp_env = prepare_environment(start, goal)

        # result = planner_func(start, goal, vamp_env, plan_settings, sampler)

        # TODO no point in simplifying the solution, right?
        robot_dir = Path(__file__).parent.parent / 'resources' / 'panda'
        sim = vpb.PyBulletSimulator(str(robot_dir / f"panda_spherized.urdf"), vamp_module.joint_names(), True)

        # results[trial] = result
        result = planner_func(start, goal, vamp_env, plan_settings, sampler)
        results[trial] = vamp_module.simplify(result.path, vamp_env, simp_settings, sampler)
        results[trial].path.interpolate_to_resolution(vamp.panda.resolution())

        sim.animate(results[trial].path)

        input("INPUT")
        exit()


        # print('RESULT FORMAT:')
        # print(type(results[0].path))
        # # print(type(results[0].path.data))
        # # print(results[0].path.shape())
        # print(results[0].path[1])
        # exit()
    
    configure_plugins(env_obs)
    add_spheres(env_obs, OBSTACLE_SPHERES, radius=0.2)

    manip_info = ManipulatorInfo()
    manip_info.manipulator = "panda_arm"
    manip_info.tcp_frame = "panda_link8"
    manip_info.working_frame = "panda_link0"
    manip_info.manipulator_ik_solver = "KDLInvKinChainLMA"

    # for p in results[0].path:
    #     print(p)

    for trial in range(n_trials):
        optimized = run_pipeline(env_obs, manip_info, results[trial].path, f'Obstacle Env #{trial}')
    
    

    



            
    
    # configure_plugins(env)
    # configure_plugins(env_obs)
    # add_spheres(env_obs, OBSTACLE_SPHERES, radius=0.2)

    # manip_info = ManipulatorInfo()
    # manip_info.manipulator = "panda_arm"
    # manip_info.tcp_frame = "panda_link8"
    # manip_info.working_frame = "panda_link0"
    # manip_info.manipulator_ik_solver = "KDLInvKinChainLMA"

    # # start_joint = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8], dtype=np.float64)
    # # start_joint = np.array([0.1, -0.8, 0.15, -2.4, 0.05, 1.6, 0.9], dtype=np.float64)
    # start_joint = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)

    # # viewer_free = TesseractViewer(server_address=('127.0.0.1',8000))
    # # viewer_free.update_environment(env, [0, 0, 0])

    # # try:
    # #     viewer_free.start_serve_background()
    # #     print("View free environment at http://localhost:8000")
    # #     input("Press Enter to continue to obstacle environment...")
    # # except Exception as e:
    # #     print(f"Error starting viewer: {e}")

    # free_path = run_pipeline(env, manip_info, start_joint, "Free env")
    # viewer_free = TesseractViewer(server_address=('0.0.0.0',8000))
    # viewer_free.update_environment(env, [0, 0, 0])
    # viewer_free.update_trajectory(free_path.results.flatten())
    # viewer_free.plot_trajectory(free_path.results.flatten(), manip_info, axes_length=0.05)
    # try:
    #     viewer_free.start_serve_background()
    #     print("View free environment at http://localhost:8000")
    #     input("Press Enter to continue to obstacle environment...")
    # except Exception as e:
    #     print(f"Error starting viewer: {e}")

    # obs_path = run_pipeline(env_obs, manip_info, start_joint, "Obstacle env")
    viewer_obs = TesseractViewer(server_address=('0.0.0.0',8080))
    viewer_obs.update_environment(env_obs, [0, 0, 0])
    viewer_obs.update_trajectory(optimized.results.flatten())
    viewer_obs.plot_trajectory(optimized.results.flatten(), manip_info, axes_length=0.05)

    try:
        viewer_obs.start_serve_background()
        print("View obstacle environment at http://localhost:8080")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Error starting viewer: {e}")




if __name__ == "__main__":
    main()
