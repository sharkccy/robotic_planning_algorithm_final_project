"""Part 1 Planner: Panda TrajOpt (empty environment first)

This script loads the Panda URDF/SRDF (sanitized version without obstacles),
constructs a simple Cartesian waypoint program, optionally seeds with a straight
(interpolated) joint path, and then optimizes with TrajOpt. Finally it assigns
time stamps using TimeOptimalTrajectoryGeneration and prints the joint states.

Next steps (future extension):
- Add multiple initial seeds (linear, random, perturbed)
- Compare success/time/length with and without obstacles
- Save statistics to CSV/JSON
- Visualize trajectories (viewer already integrated)
"""
import os
import time
import numpy as np
from pathlib import Path
from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator, Isometry3d, Translation3d, Quaterniond, ManipulatorInfo
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_trajopt import TrajOptDefaultPlanProfile, TrajOptDefaultCompositeProfile, TrajOptMotionPlanner
from tesseract_robotics.tesseract_command_language import CartesianWaypoint, MoveInstructionType_FREESPACE, MoveInstruction, CompositeInstruction, ProfileDictionary, \
    CartesianWaypointPoly_wrap_CartesianWaypoint, MoveInstructionPoly_wrap_MoveInstruction, InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly, \
    JointWaypoint, JointWaypointPoly_wrap_JointWaypoint
from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, InstructionsTrajectory
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
from tesseract_robotics_viewer import TesseractViewer
from tesseract_robotics.tesseract_collision import ContactResultMap, ContactTestType_ALL, ContactRequest, ContactResultVector

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = SCRIPT_DIR / "resources" / "panda"
URDF_PATH = FilesystemPath(str(RESOURCES_DIR / "panda_resolved.urdf"))
# Use sanitized SRDF (no invalid hand group); switch to hand links version if desired
SRDF_PATH = FilesystemPath(str(RESOURCES_DIR / "panda_with_hand_links.srdf"))

TRAJOPT_NS = "TrajOptMotionPlannerTask"
N_STEPS_INTERP = 15  # number of waypoints between cartesian targets after interpolation (if extended later)

# Simple joint velocity/accel/jerk limits for Panda (rad/s etc.)
PANDA_MAX_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175], dtype=np.float64)
PANDA_MAX_ACC = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
PANDA_MAX_JERK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

OBSTACLE_SPHERES = [
    (0.55, 0.0, 0.25), (0.35, 0.35, 0.25), (0.0, 0.55, 0.25),
    (-0.55, 0.0, 0.25), (-0.35, -0.35, 0.25), (0.0, -0.55, 0.25),
    (0.35, -0.35, 0.8), (0.35, 0.35, 0.8), (0.0, 0.55, 0.8),
    (-0.35, 0.35, 0.8), (-0.55, 0.0, 0.8), (-0.35, -0.35, 0.8),
    (0.0, -0.55, 0.8), (0.35, -0.35, 0.8)
]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# def build_cartesian_program(manip_info: ManipulatorInfo) -> CompositeInstruction:
#     """Create a minimal Cartesian waypoint program with two free-space moves.
#     need to set IK solver in env beforehand."""
#     # Example end-effector poses (tweak if unreachable)
#     wp_start = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.4, 0.0, 0.4) * Quaterniond(1,0,0,0))
#     wp_goal = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.6, 0.2, 0.5) * Quaterniond(1,0,0,0))

#     instr_start = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp_start), MoveInstructionType_FREESPACE, "DEFAULT")
#     instr_goal = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp_goal), MoveInstructionType_FREESPACE, "DEFAULT")

#     program = CompositeInstruction("DEFAULT")
#     program.setManipulatorInfo(manip_info)
#     program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_start))
#     program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr_goal))
#     return program

def build_simple_joint_seed(manip_info: ManipulatorInfo, joint_names: list[str], n_steps: int = 15, noise_std: float = 0.05, rng: np.random.Generator | None = None) -> CompositeInstruction:
    """Build a simple linear joint-space seed between two joint states, with small random noise."""
    if rng is None:
        rng = np.random.default_rng()

    start_q = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8], dtype=np.float64)
    goal_q  = np.array([0.2, -0.2, 0.2, -2.0, 0.2, 1.6, 0.8], dtype=np.float64)

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)

    for i in range(n_steps):
        alpha = i / float(n_steps - 1)
        q = (1.0 - alpha) * start_q + alpha * goal_q

        # add gaussian noise to each waypoint, controlled by noise_std
        noise = rng.normal(loc=0.0, scale=noise_std, size=q.shape).astype(np.float64)
        q_noisy = q + noise

        wp = JointWaypoint(joint_names, q_noisy)
        instr = MoveInstruction(
            JointWaypointPoly_wrap_JointWaypoint(wp),
            MoveInstructionType_FREESPACE,
            "DEFAULT",
        )
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instr))

    return program

def optimize_with_trajopt(env: Environment, input_program: CompositeInstruction):
    """Run TrajOpt on the input program and return optimized instructions."""
    plan_profile = TrajOptDefaultPlanProfile()
    composite_profile = TrajOptDefaultCompositeProfile()

    profiles = ProfileDictionary()
    profiles.addProfile(TRAJOPT_NS, "DEFAULT", plan_profile)
    profiles.addProfile(TRAJOPT_NS, "DEFAULT", composite_profile)

    request = PlannerRequest()
    request.instructions = input_program
    request.env = env
    request.profiles = profiles

    planner = TrajOptMotionPlanner(TRAJOPT_NS)
    response = planner.solve(request)
    if not response.successful:
        raise RuntimeError("TrajOpt optimization failed")
    return response.results


def time_parameterize(instructions: CompositeInstruction):
    """Assign timestamps using TimeOptimalTrajectoryGeneration."""
    trajectory = InstructionsTrajectory(instructions)
    top = TimeOptimalTrajectoryGeneration()
    # Expand limits to expected shape (if needed replicate example pattern)
    vel = PANDA_MAX_VEL.copy()
    acc = PANDA_MAX_ACC.copy()
    jerk = PANDA_MAX_JERK.copy()
    if not top.compute(trajectory, vel, acc, jerk):
        raise RuntimeError("Time parameterization failed")


def print_results(instructions: CompositeInstruction):
    flat = instructions.flatten()
    for instr in flat:
        if not instr.isMoveInstruction():
            continue
        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp_poly = move.getWaypoint()
        if not wp_poly.isStateWaypoint():
            continue
        state_wp = WaypointPoly_as_StateWaypointPoly(wp_poly)
        pos = state_wp.getPosition().flatten()
        print(f"Joint: {pos}  t={state_wp.getTime():.4f}")

def trajectory_length(instructions: CompositeInstruction) -> float:
    """Compute joint-space length of a flattened set of StateWaypoints."""
    flat = instructions.flatten()
    q_list = []
    for instr in flat:
        if not instr.isMoveInstruction():
            continue

        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp_poly = move.getWaypoint()

        if not wp_poly.isStateWaypoint():
            continue

        state_wp = WaypointPoly_as_StateWaypointPoly(wp_poly)
        q_list.append(state_wp.getPosition().astype(np.float64).flatten())
    if len(q_list) < 2:
        return 0.0
    length = 0.0
    for i in range(len(q_list) - 1):
        length += float(np.linalg.norm(q_list[i+1] - q_list[i]))
    return length

def is_trajectory_collision_free(env: Environment, joint_names: list[str], instructions: CompositeInstruction, margin: float = 0.0) -> bool:
    """Discrete collision check per waypoint using the environment manager.
    Returns True if no contacts closer than margin across all waypoints.
    """
    solver = env.getStateSolver()
    manager = env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(env.getLinkNames())
    # Set exact margin (0.0 -> strict collision only)
    # Not setting CollisionMarginData here keeps default; acceptable for strict check

    flat = instructions.flatten()
    for instr in flat:
        if not instr.isMoveInstruction():
            continue
        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp_poly = move.getWaypoint()
        if not wp_poly.isStateWaypoint():
            continue
        state_wp = WaypointPoly_as_StateWaypointPoly(wp_poly)
        q = state_wp.getPosition().astype(np.float64).flatten()

        solver.setState(joint_names, q)
        scene_state = solver.getState()
        manager.setCollisionObjectsTransform(scene_state.link_transforms)

        result_map = ContactResultMap()
        manager.contactTest(result_map, ContactRequest(ContactTestType_ALL))
        result_vec = ContactResultVector()
        result_map.flattenMoveResults(result_vec)
        # Any strictly negative distance indicates penetration
        for i in range(len(result_vec)):
            if result_vec[i].distance < margin:
                return False
    return True

def add_spheres(env: Environment, centers, radius: float = 0.2, parent_link: str = "panda_link0", name_prefix: str = "sphere_"):
    """Add one or more spheres. `centers` may be a tuple (x,y,z) or an iterable of centers."""
    # Normalize centers to list
    if isinstance(centers, tuple) and len(centers) == 3:
        centers = [centers]
    for i, c in enumerate(centers):
        geometry = Sphere(radius)

        vision = Visual(); vision.geometry = geometry
        collision = Collision(); collision.geometry = geometry

        link_name = f"{name_prefix}{i}"
        link = Link(link_name)
        link.visual.push_back(vision)
        link.collision.push_back(collision)

        joint = Joint(f"{link_name}_joint")
        joint.type = JointType_FIXED
        joint.parent_link_name = parent_link
        joint.child_link_name = link_name

        transform = Isometry3d.Identity() * Translation3d(c[0], c[1], c[2])
        joint.parent_to_joint_origin_transform = transform

        env.applyCommand(AddLinkCommand(link, joint))

def plan_once(env: Environment, manip_info: ManipulatorInfo, joint_names: list[str]) -> tuple[bool, float, float, CompositeInstruction]:
    """Plan with a given seed program: interpolate, TrajOpt, time-param, metrics."""
    rng = np.random.default_rng(42)
    seed_prog = build_simple_joint_seed(manip_info, joint_names, n_steps=N_STEPS_INTERP, noise_std=0.05, rng=rng)

    t0 = time.perf_counter()
    optimized = optimize_with_trajopt(env, seed_prog)
    t1 = time.perf_counter()
    plan_time = t1 - t0
    time_parameterize(optimized)
    ok = is_trajectory_collision_free(env, joint_names, optimized, margin=0.0)
    length = trajectory_length(optimized)
    return ok, plan_time, length, optimized

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Two identical environments: one free, one with obstacles
    env_free = Environment(); 
    env_obs = Environment()
    locator = GeneralResourceLocator()
    if not env_free.init(URDF_PATH, SRDF_PATH, locator):
        raise RuntimeError("Failed to initialize Panda (free env)")
    if not env_obs.init(URDF_PATH, SRDF_PATH, locator):
        raise RuntimeError("Failed to initialize Panda (obstacle env)")

    # Add obstacle spheres to env_obs
    # add_spheres(env_obs, OBSTACLE_SPHERES, radius=0.2, parent_link="panda_link0")

    # Manipulator info (matches SRDF group name and frames)
    manip_info = ManipulatorInfo()
    manip_info.manipulator = "panda_arm"  # SRDF group
    manip_info.tcp_frame = "panda_link8"   # end effector link
    manip_info.working_frame = "panda_link0"  # base

    # Set a nominal start joint state
    env_free.setState(JOINT_NAMES, np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]))
    env_obs.setState(JOINT_NAMES, np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]))

    # Run 3 identical Cartesian programs (same start/goal), keep metrics
    N = 1
    times_free = []
    lengths_free = []
    times_obs = []
    lengths_obs = []
    paired_success = 0
    first_ok_pair = (None, None)

    for si in range(N):
        # Deterministic Cartesian program: same TCP start/goal each time
        try:
            collide_f, times_f, lengths_f, optmized_f = plan_once(env_free, manip_info, JOINT_NAMES)
            collide_o, times_o, lengths_o, optmized_o = plan_once(env_obs, manip_info, JOINT_NAMES)
        except Exception as e:
            print(f"  Seed {si+1}/{N}: planning exception, skipping")
            print(f"exception details:", e)
            continue

        if collide_f and collide_o:
            times_free.append(times_f)
            lengths_free.append(lengths_f)
            times_obs.append(times_o)
            lengths_obs.append(lengths_o)

            if first_ok_pair[0] is None:
                first_ok_pair = (optmized_f, optmized_o)
            paired_success += 1

    print(f"Seeds tried: {N}, paired successes (both envs): {paired_success}")
    if paired_success:
        print(f"No-obstacles  -> time: {np.mean(times_free):.4f}s ± {np.std(times_free):.4f}s, length: {np.mean(lengths_free):.4f} ± {np.std(lengths_free):.4f}")
        print(f"With obstacles-> time: {np.mean(times_obs):.4f}s ± {np.std(times_obs):.4f}s, length: {np.mean(lengths_obs):.4f} ± {np.std(lengths_obs):.4f}")
    else:
        print("No paired successes. Loosen random ranges or increase waypoints/seeds.")

    # Viewer: show first pair
    if first_ok_pair[0] is not None:
        try:
            viewer = TesseractViewer()
            viewer.update_environment(env_free, [0,0,0])
            viewer.update_trajectory(first_ok_pair[0].flatten())
            viewer.plot_trajectory(first_ok_pair[0].flatten(), manip_info, axes_length=0.05)
            viewer.start_serve_background()
            input("Press Enter to exit first (no-obstacles) viewer...")
        except Exception as e:
            print("Viewer failed (free):", e)
        try:
            viewer2 = TesseractViewer()
            viewer2.update_environment(env_obs, [0,0,0])
            viewer2.update_trajectory(first_ok_pair[1].flatten())
            viewer2.plot_trajectory(first_ok_pair[1].flatten(), manip_info, axes_length=0.05)
            viewer2.start_serve_background()
            input("Press Enter to exit second (obstacles) viewer...")
        except Exception as e:
            print("Viewer failed (obs):", e)

if __name__ == "__main__":
    main()
