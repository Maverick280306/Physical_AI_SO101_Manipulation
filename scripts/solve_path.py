#!/usr/bin/env python
"""Generate physically executed red-cube pick/place demonstrations for ACT.

    python scripts/solve_path.py --no-record --episodes 1 --seed 1000
    python scripts/solve_path.py --motion-method optimized --repo-id local/so101_optimized --episodes 1
    python scripts/solve_path.py --motion-method optimized --no-record --episodes 1 --preview
    python scripts/solve_path.py --motion-method optimizer2 --no-record --episodes 1 --preview
    python scripts/solve_path.py --repo-id local/so101_expert --episodes 20

Run in the same LeRobot environment as teleop_collect.py. Existing datasets are
never overwritten. Failed starts are reported and discarded, not saved as demos.

The expert uses privileged MuJoCo state; recorded observations contain only the
same joint state and two RGB cameras as teleop_collect.py. No live state is
teleported by the planner. Arm calculations use radians and world-frame metres.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from teleop_collect import (
    ACTION, FPS, JOINTS, OBS_STR, action_to_values, build_dataset, build_dataset_frame,
    make_env, obs_to_values,
)

TASK_DESCRIPTION = "Pick up the red cube and place it in the green box."


@dataclass(frozen=True)
class ExpertConfig:
    position_tolerance: float = 0.0015
    orientation_tolerance: float = 0.07  # tolerate the SO-101 wrist's small axis skew
    max_velocity: float = 0.8  # rad/s, including the gripper hinge
    max_acceleration: float = 1.6  # rad/s²
    approach_height: float = 0.08
    transfer_height: float = 0.11
    grasp_offset: float = -0.003  # site height relative to cube centre
    open_gripper: float = 60.0
    closed_gripper: float = 0.0
    optimizer_iterations: int = 160
    max_jerk: float = 8.0  # rad/s³, optimized trajectories only
    contact_grasp: bool = True  # optimizer2: stop closure once the grasp settles


@dataclass
class EnvironmentHandles:
    venv: object
    world: object
    model: mujoco.MjModel
    data: mujoco.MjData
    planning_data: mujoco.MjData
    qadr: np.ndarray
    dof: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    site: int
    cube: int
    box: int
    robot_bodies: set[int]
    finger_bodies: set[int]
    preview: object = None


class LivePreview:
    """Display existing camera observations; no additional MuJoCo render calls."""

    def __init__(self):
        import pygame

        self.pygame = pygame
        try:
            pygame.display.init()
            self.screen = pygame.display.set_mode((640, 480), pygame.SWSURFACE)
        except pygame.error as exc:
            # Capture SDL's error before shutdown clears its diagnostic state.
            detail = str(exc) or pygame.get_error() or "SDL returned no error description"
            display = os.environ.get("DISPLAY", "unset")
            driver = os.environ.get("SDL_VIDEODRIVER", "auto")
            pygame.display.quit()
            raise RuntimeError(
                f"Cannot open preview: {detail} (DISPLAY={display}, SDL_VIDEODRIVER={driver}). "
                "Run from a local desktop terminal with an accessible display, "
                "or omit --preview for headless execution."
            ) from exc
        self.clock = pygame.time.Clock()

    def poll(self):
        for event in self.pygame.event.get():
            if (event.type == self.pygame.QUIT
                    or (event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE)):
                raise KeyboardInterrupt

    def show(self, obs, stage):
        self.poll()
        frame = np.asarray(obs["pixels"]["front"])[0]
        surface = self.pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))
        self.pygame.display.set_caption(f"SO-101 expert | {stage} | ESC to stop")
        self.pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        self.pygame.display.quit()


def create_environment():
    """Use the same single, synchronous Hub environment as the teleop script."""
    return make_env(
        "lerobot/so101-tabletop-mujoco", n_envs=1, trust_remote_code=True
    )["so101_tabletop"][0]


def inspect_environment(venv) -> EnvironmentHandles:
    """Resolve this Hub model's required names after reset; reject incompatible models."""
    if venv.num_envs != 1 or not hasattr(venv, "envs"):
        raise ValueError("The expert requires one synchronous environment")
    env = venv.envs[0].unwrapped
    world = getattr(env, "_world", None)
    if world is None:
        raise ValueError("Reset the environment before inspecting it")
    m = world.model
    joints = [m.joint(name).id for name in JOINTS]
    actuators = [m.actuator(name).id for name in JOINTS]
    if not np.array_equal(m.actuator_trnid[actuators, 0], joints):
        raise ValueError("Expected direct joint-position actuators")
    lower = np.maximum(m.jnt_range[joints, 0], m.actuator_ctrlrange[actuators, 0])
    upper = np.minimum(m.jnt_range[joints, 1], m.actuator_ctrlrange[actuators, 1])
    base = m.body("base").id
    robot = set()
    for body in range(m.nbody):
        ancestor = body
        while ancestor and ancestor != base:
            ancestor = int(m.body_parentid[ancestor])
        if ancestor == base:
            robot.add(body)
    return EnvironmentHandles(
        venv, world, m, world.data, mujoco.MjData(m),
        m.jnt_qposadr[joints].copy(), m.jnt_dofadr[joints].copy(), lower, upper,
        m.site("gripperframe").id, m.body("red_cube").id, m.body("box").id,
        robot, {m.body("gripper").id, m.body("moving_jaw_so101_v1").id},
    )


def read_scene_state(h: EnvironmentHandles) -> dict:
    """Copy current state so snapshots stay valid after subsequent simulator steps."""
    return {
        "joints": h.data.qpos[h.qadr].copy(),
        "gripper_pos": h.data.site_xpos[h.site].copy(),
        "gripper_rotation": h.data.site_xmat[h.site].reshape(3, 3).copy(),
        "cube_pos": h.data.xpos[h.cube].copy(),
        "cube_rotation": h.data.xmat[h.cube].reshape(3, 3).copy(),
        "box_pos": h.data.xpos[h.box].copy(),
        "box_rotation": h.data.xmat[h.box].reshape(3, 3).copy(),
    }


def env_action_to_model_joints(action: np.ndarray, h: EnvironmentHandles) -> np.ndarray:
    """Convert six environment units to hinge radians without silently clipping."""
    action = np.asarray(action, dtype=float)
    if action.shape != (6,) or not np.all(np.isfinite(action)):
        raise ValueError("Action must contain six finite values")
    q = np.deg2rad(action)
    q[5] = h.world.lo[5] + action[5] / 100 * (h.world.hi[5] - h.world.lo[5])
    return q


def model_joints_to_env_action(q: np.ndarray, h: EnvironmentHandles) -> np.ndarray:
    """Convert six hinge positions to the action format consumed by venv.step."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Joint configuration must contain six finite values")
    action = np.rad2deg(q)
    action[5] = 100 * (q[5] - h.world.lo[5]) / (h.world.hi[5] - h.world.lo[5])
    return action.astype(np.float32)


class PlanningError(RuntimeError):
    """A target cannot be reached safely; discard this attempt."""


def _pose_error(target_pos, target_rotation, data, site):
    position = target_pos - data.site_xpos[site]
    current = data.site_xmat[site].reshape(3, 3)
    # Quaternion difference expressed in world coordinates, matching mj_jacSite.
    delta = np.empty(4)
    mujoco.mju_mat2Quat(delta, (target_rotation @ current.T).ravel())
    angular = np.empty(3)
    mujoco.mju_quat2Vel(angular, delta, 1.0)
    return position, angular


def solve_ik(target_pos, target_rotation, q_seed, h: EnvironmentHandles,
             config: ExpertConfig, *, seed_limits=None) -> np.ndarray:
    """Damped least-squares site IK, keeping the gripper hinge fixed.

    Full orientation is requested only for the top-down poses supported by this
    arm. Incompatible poses fail explicitly rather than returning a poor fit.
    seed_limits optionally accepts a bounded approximate result ONLY for a
    downstream constrained optimizer; normal control calls remain strict.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if (target_pos.shape != (3,) or target_rotation.shape != (3, 3)
            or not np.all(np.isfinite(target_pos))
            or not np.all(np.isfinite(target_rotation))):
        raise ValueError("Expected a finite position and 3x3 rotation")
    q = np.clip(np.asarray(q_seed, dtype=float).copy(), h.lower, h.upper)
    d, m = h.planning_data, h.model
    d.qpos[:] = h.data.qpos
    d.qvel[:] = 0
    jac_pos, jac_rot = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    orientation_weight = 0.05  # metres/radian
    for _ in range(250):
        d.qpos[h.qadr] = q
        mujoco.mj_forward(m, d)
        pos_error, rot_error = _pose_error(target_pos, target_rotation, d, h.site)
        if (np.linalg.norm(pos_error) <= config.position_tolerance
                and np.linalg.norm(rot_error) <= config.orientation_tolerance):
            return q
        error = np.r_[pos_error, orientation_weight * rot_error]
        mujoco.mj_jacSite(m, d, jac_pos, jac_rot, h.site)
        jac = np.vstack((jac_pos[:, h.dof[:5]], orientation_weight * jac_rot[:, h.dof[:5]]))
        change = np.linalg.solve(jac.T @ jac + 1e-5 * np.eye(5), jac.T @ error)
        change *= min(1.0, 0.15 / max(np.max(np.abs(change)), 1e-12))
        improved = False
        for scale in (1, 0.5, 0.25, 0.1):
            trial = q.copy()
            trial[:5] = np.clip(q[:5] + scale * change, h.lower[:5], h.upper[:5])
            d.qpos[h.qadr] = trial
            mujoco.mj_forward(m, d)
            pe, re = _pose_error(target_pos, target_rotation, d, h.site)
            if np.linalg.norm(np.r_[pe, orientation_weight * re]) < np.linalg.norm(error):
                q, improved = trial, True
                break
        if not improved:
            break
    # Evaluate the last accepted iterate, including an update on iteration 250.
    d.qpos[h.qadr] = q
    mujoco.mj_forward(m, d)
    pos_error, rot_error = _pose_error(target_pos, target_rotation, d, h.site)
    if seed_limits is not None:
        position_limit, orientation_limit = seed_limits
        if (np.linalg.norm(pos_error) <= position_limit
                and np.linalg.norm(rot_error) <= orientation_limit):
            return q
    raise PlanningError(
        f"IK failed: position error={np.linalg.norm(pos_error):.6f} m, "
        f"orientation error={np.linalg.norm(rot_error):.4f} rad"
    )


def top_down_rotation(jaw_yaw: float) -> np.ndarray:
    """Site +X points down the fingers; site +Z runs across the jaw gap."""
    x = np.array([0., 0., -1.])
    z = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.])
    return np.column_stack((x, np.cross(z, x), z))


def generate_joint_trajectory(q_start, q_goal, config: ExpertConfig,
                              min_duration: float = 0.5) -> np.ndarray:
    """Sample a quintic rest-to-rest motion; bounds are analytical, not guessed.

    Includes both endpoints. Execution skips the initial sample, already held.
    """
    start, goal = np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float)
    if start.shape != (6,) or goal.shape != (6,) or not np.all(np.isfinite([start, goal])):
        raise ValueError("Expected two finite six-joint configurations")
    if config.max_velocity <= 0 or config.max_acceleration <= 0 or min_duration <= 0:
        raise ValueError("Motion limits and duration must be positive")
    distance = float(np.max(np.abs(goal - start)))
    duration = max(min_duration, 1.875 * distance / config.max_velocity,
                   np.sqrt(5.774 * distance / config.max_acceleration))
    intervals = max(2, int(np.ceil(duration * FPS)))
    t = np.linspace(0., 1., intervals + 1)
    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
    return start + blend[:, None] * (goal - start)


def _unintended_contacts(h: EnvironmentHandles, data, allow_cube: bool) -> list[str]:
    collisions = []
    for contact in data.contact:
        if contact.dist > -0.001:  # tolerate sub-millimetre contact softness
            continue
        a, b = (int(h.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if a not in h.robot_bodies and b not in h.robot_bodies:
            continue
        if allow_cube and ((a == h.cube and b in h.finger_bodies)
                           or (b == h.cube and a in h.finger_bodies)):
            continue
        collisions.append(f"{h.model.body(a).name or 'world'}/{h.model.body(b).name or 'world'}")
    return collisions


def validate_joint_trajectory(trajectory, h: EnvironmentHandles, allow_cube=False):
    """Screen each sample for limits and robot collisions on a separate state.

    Other objects stay at their current poses. This is not a predictive contact
    simulation or a guarantee about a carried object's swept volume.
    """
    if (trajectory.ndim != 2 or trajectory.shape[1] != 6
            or not np.all(np.isfinite(trajectory))):
        raise PlanningError("Invalid trajectory")
    if np.any(trajectory < h.lower - 1e-7) or np.any(trajectory > h.upper + 1e-7):
        raise PlanningError("Trajectory exceeds joint limits")
    d = h.planning_data
    d.qpos[:] = h.data.qpos
    d.qvel[:] = 0
    for q in trajectory:
        d.qpos[h.qadr] = q
        mujoco.mj_forward(h.model, d)
        collisions = _unintended_contacts(h, d, allow_cube)
        if collisions:
            raise PlanningError(f"Predicted collision: {collisions[0]}")


@dataclass
class EpisodeContext:
    handles: EnvironmentHandles
    config: ExpertConfig
    obs: dict
    command: np.ndarray
    dataset: object = None
    max_steps: int = 1800
    steps: int = 0
    stage: str = "reset"


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    seed: int
    steps: int
    stage: str
    reason: str = ""


def step_and_record(q: np.ndarray, context: EpisodeContext, allow_cube=False):
    """Execute through the public action interface; buffer the preceding observation.

    LeRobot spools camera images to disk, avoiding an episode-sized RGB RAM buffer.
    Nothing is committed as an episode until placement has been verified.
    """
    h = context.handles
    if h.preview is not None:
        h.preview.poll()
    if context.steps >= context.max_steps:
        raise PlanningError("Expert step limit reached")
    if np.any(q < h.lower - 1e-6) or np.any(q > h.upper + 1e-6):
        raise PlanningError("Command exceeds joint limits")
    action = model_joints_to_env_action(q, h)
    frame = None
    if context.dataset is not None:
        features = context.dataset.features
        frame = {
            **build_dataset_frame(features, obs_to_values(context.obs), prefix=OBS_STR),
            **build_dataset_frame(features, action_to_values(action), prefix=ACTION),
            "task": TASK_DESCRIPTION,
        }
    obs, _, terminated, truncated, _ = h.venv.step(action[None, :])
    context.obs = obs
    context.steps += 1
    context.command = np.asarray(q).copy()
    if h.preview is not None:
        h.preview.show(obs, context.stage)
    if frame is not None:
        context.dataset.add_frame(frame)
    if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
        raise PlanningError("Environment terminated or truncated before expert validation")
    if not np.all(np.isfinite(h.data.qpos)):
        raise PlanningError("Non-finite simulator state")
    if np.max(np.abs(h.data.qpos[h.qadr[:5]] - q[:5])) > 0.45:
        raise PlanningError("Arm tracking error exceeds 0.45 rad")
    contacts = _unintended_contacts(h, h.data, allow_cube)
    if contacts:
        raise PlanningError(f"Execution collision: {contacts[0]}")


def execute_trajectory(trajectory, context: EpisodeContext, allow_cube=False):
    """Validate then execute a smooth trajectory, checking physical tracking."""
    h = context.handles
    validate_joint_trajectory(trajectory, h, allow_cube)
    for q in trajectory[1:]:
        step_and_record(q, context, allow_cube)
    # Allow the physical servo to settle without a new command discontinuity.
    for _ in range(round(0.3 * FPS)):
        step_and_record(trajectory[-1], context, allow_cube)
    if np.max(np.abs(h.data.qpos[h.qadr[:5]] - trajectory[-1, :5])) > 0.12:
        raise PlanningError("Arm did not settle within 0.12 rad")


def move_gripper(target: float, context: EpisodeContext):
    """Ramp the jaw command while holding the arm; permit intended cube contact."""
    h = context.handles
    action = model_joints_to_env_action(context.command, h)
    action[5] = target
    goal = env_action_to_model_joints(action, h)
    trajectory = generate_joint_trajectory(context.command, goal, context.config)
    execute_trajectory(trajectory, context, allow_cube=True)


def discard_episode(dataset):
    """Drain image-writer threads before deleting a failed attempt's temp images."""
    if dataset is not None:
        # LeRobot's delete operation does not itself drain pending image writes.
        if dataset.writer.image_writer is not None:
            dataset.writer.image_writer.wait_until_done()
        dataset.clear_episode_buffer()


def transfer_rotation(position: np.ndarray) -> np.ndarray:
    """Tilt fingers outward for high targets beyond the vertical wrist workspace."""
    yaw = np.arctan2(position[1], position[0] - 0.01)
    c, s = np.cos(yaw), np.sin(yaw)
    radial = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    tilt = 0.7
    c, s = np.cos(tilt), np.sin(tilt)
    pitch = np.array([[c, 0, -s], [0, 1., 0], [s, 0, c]])
    return radial @ pitch @ top_down_rotation(0)


def get_stage_target(stage: str, scene: dict, config: ExpertConfig):
    """Object-relative position and rotation for the current motion stage."""
    if stage in ("approach", "pregrasp", "descend", "test_lift", "lift"):
        pos = scene["cube_pos"].copy()
        if stage == "approach":
            pos[2] += config.approach_height
        elif stage == "pregrasp":
            pos[2] += 0.035
        elif stage == "descend":
            pos[2] += config.grasp_offset
        elif stage == "test_lift":
            pos = scene["gripper_pos"].copy()
            pos[2] += 0.035
        else:
            pos = scene["gripper_pos"].copy()
            pos[2] = config.transfer_height
    elif stage in ("transfer", "lower", "retreat"):
        pos = scene["box_pos"].copy()
        pos[2] += config.transfer_height if stage != "lower" else 0.065
    else:
        raise ValueError(f"Unknown motion stage: {stage}")
    return pos, transfer_rotation(pos)


def _move_to(pos, rotation, context: EpisodeContext, allow_cube=False):
    q = solve_ik(pos, rotation, context.command, context.handles, context.config)
    trajectory = generate_joint_trajectory(context.command, q, context.config)
    execute_trajectory(trajectory, context, allow_cube)


def choose_grasp(scene: dict, context: EpisodeContext) -> np.ndarray:
    """Choose the shortest feasible cube-aligned grasp, screening nearby obstacles."""
    h, config = context.handles, context.config
    cube_yaw = np.arctan2(scene["cube_rotation"][1, 0], scene["cube_rotation"][0, 0])
    grasp_pos, _ = get_stage_target("descend", scene, config)
    pre_pos, _ = get_stage_target("pregrasp", scene, config)
    candidates = []
    for yaw in cube_yaw + np.arange(4) * np.pi / 2:
        rotation = top_down_rotation(yaw)
        try:
            grasp = solve_ik(grasp_pos, rotation, context.command, h, config)
            pre = solve_ik(pre_pos, rotation, grasp, h, config)
            closed = grasp.copy()
            closed[5] = env_action_to_model_joints(
                np.array([0, 0, 0, 0, 0, config.closed_gripper]), h)[5]
            for start, goal, allow_cube in (
                (context.command, pre, False), (pre, grasp, True), (grasp, closed, True)
            ):
                validate_joint_trajectory(generate_joint_trajectory(start, goal, config), h, allow_cube)
            candidates.append((np.linalg.norm(pre - context.command), rotation))
        except PlanningError:
            continue
    if not candidates:
        raise PlanningError("No reachable, collision-free vertical grasp/pregrasp pair")
    return min(candidates, key=lambda item: item[0])[1]


def check_grasp(before: dict, after: dict) -> bool:
    """Require actual cube lift and a stable cube-to-site displacement."""
    rise = after["cube_pos"][2] - before["cube_pos"][2]
    relative_before = before["cube_pos"] - before["gripper_pos"]
    relative_after = after["cube_pos"] - after["gripper_pos"]
    return bool(rise > 0.02 and np.linalg.norm(relative_after - relative_before) < 0.025)


def check_placement(h: EnvironmentHandles) -> bool:
    """Require the whole cube inside the box footprint and resting near its floor."""
    rotation = h.data.xmat[h.box].reshape(3, 3)
    local = rotation.T @ (h.data.xpos[h.cube] - h.data.xpos[h.box])
    geom = h.model.body_geomadr[h.cube]
    half_size = h.model.geom_size[geom]
    relative_rotation = rotation.T @ h.data.xmat[h.cube].reshape(3, 3)
    extent = np.abs(relative_rotation) @ half_size
    cube_joint = h.model.body_jntadr[h.cube]
    dof = h.model.jnt_dofadr[cube_joint]
    return bool(
        np.all(np.abs(local[:2]) + extent[:2] < 0.044)
        and 0.014 < local[2] < 0.035
        and np.linalg.norm(h.data.qvel[dof:dof + 3]) < 0.02
    )


def run_expert_episode(h: EnvironmentHandles, config: ExpertConfig, seed: int,
                       dataset=None, max_steps=1800) -> EpisodeResult:
    """Execute a checked pick/place; a failure always clears pending recording."""
    discard_episode(dataset)
    obs, _ = h.venv.reset(seed=seed)
    context = EpisodeContext(h, config, obs, h.data.qpos[h.qadr].copy(), dataset, max_steps)
    try:
        for _ in range(10):
            step_and_record(context.command, context)
        context.stage = "open"
        move_gripper(config.open_gripper, context)
        context.stage = "approach"
        scene = read_scene_state(h)
        _move_to(*get_stage_target("approach", scene, config), context)
        context.stage = "plan_grasp"
        rotation = choose_grasp(read_scene_state(h), context)
        for stage in ("pregrasp", "descend"):
            context.stage = stage
            pos, _ = get_stage_target(stage, read_scene_state(h), config)
            _move_to(pos, rotation, context, allow_cube=(stage == "descend"))
        context.stage = "close"
        move_gripper(config.closed_gripper, context)
        before = read_scene_state(h)
        context.stage = "test_lift"
        pos, _ = get_stage_target("test_lift", before, config)
        _move_to(pos, rotation, context, allow_cube=True)
        if not check_grasp(before, read_scene_state(h)):
            raise PlanningError("Cube did not remain grasped during test lift")
        for stage in ("lift", "transfer", "lower"):
            context.stage = stage
            _move_to(*get_stage_target(stage, read_scene_state(h), config), context, allow_cube=True)
            if h.data.xpos[h.cube, 2] < 0.035:
                raise PlanningError("Cube dropped during transfer")
        context.stage = "release"
        move_gripper(config.open_gripper, context)
        context.stage = "retreat"
        _move_to(*get_stage_target("retreat", read_scene_state(h), config), context, allow_cube=True)
        context.stage = "validate"
        for _ in range(20):
            step_and_record(context.command, context)
            if not check_placement(h):
                raise PlanningError("Cube is not resting fully inside the box")
        return EpisodeResult(True, seed, context.steps, context.stage)
    except PlanningError as exc:
        discard_episode(dataset)
        return EpisodeResult(False, seed, context.steps, context.stage, str(exc))
    except BaseException:
        discard_episode(dataset)
        raise


def _sample_spline(knots, samples):
    """Clamped cubic joint spline: C2 continuous, zero endpoint velocity.

    Derivatives are with respect to normalized time, before duration scaling.
    Interior knots are pass-through points, not stops.
    """
    n = len(knots)
    spacing = 1.0 / (n - 1)
    matrix = np.diag(np.full(n, 4.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    matrix[0, 0] = matrix[-1, -1] = 2
    rhs = np.zeros_like(knots)
    rhs[1:-1] = 6 * np.diff(knots, n=2, axis=0) / spacing**2
    rhs[0] = 6 * (knots[1] - knots[0]) / spacing**2
    rhs[-1] = -6 * (knots[-1] - knots[-2]) / spacing**2
    second = np.linalg.solve(matrix, rhs)
    clock = np.linspace(0, 1, samples)
    t = 10 * clock**3 - 15 * clock**4 + 6 * clock**5
    rate = (30 * clock**2 - 60 * clock**3 + 30 * clock**4)[:, None]
    rate2 = (60 * clock - 180 * clock**2 + 120 * clock**3)[:, None]
    rate3 = (60 - 360 * clock + 360 * clock**2)[:, None]
    i = np.minimum((t / spacing).astype(int), n - 2)
    b = ((t - i * spacing) / spacing)[:, None]
    a = 1 - b
    left, right = knots[i], knots[i + 1]
    ml, mr = second[i], second[i + 1]
    position = a * left + b * right + ((a**3 - a) * ml + (b**3 - b) * mr) * spacing**2 / 6
    velocity = (right - left) / spacing + ((1 - 3 * a**2) * ml + (3 * b**2 - 1) * mr) * spacing / 6
    acceleration = a * ml + b * mr
    jerk = (mr - ml) / spacing
    return (position, velocity * rate, acceleration * rate**2 + velocity * rate2,
            jerk * rate**3 + 3 * acceleration * rate * rate2 + velocity * rate3)


def _planning_collisions(trajectory, h, carrying=False):
    """Count sampled penetrations, including the carried cube's swept poses.

    Only planning_data is changed. The rigid attachment is a planning
    approximation; execution still relies exclusively on physical finger contact.
    Checks are discrete and do not certify continuous collision freedom.
    """
    d = h.planning_data
    d.qpos[:] = h.data.qpos
    d.qvel[:] = 0
    site_rotation = h.data.site_xmat[h.site].reshape(3, 3)
    offset = site_rotation.T @ (h.data.xpos[h.cube] - h.data.site_xpos[h.site])
    relative_rotation = site_rotation.T @ h.data.xmat[h.cube].reshape(3, 3)
    cube_qadr = h.model.jnt_qposadr[h.model.body_jntadr[h.cube]]
    violations = 0
    for q in trajectory:
        d.qpos[h.qadr] = q
        mujoco.mj_forward(h.model, d)
        if carrying:
            rotation = d.site_xmat[h.site].reshape(3, 3)
            d.qpos[cube_qadr:cube_qadr + 3] = d.site_xpos[h.site] + rotation @ offset
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, (rotation @ relative_rotation).ravel())
            d.qpos[cube_qadr + 3:cube_qadr + 7] = quaternion
            mujoco.mj_forward(h.model, d)
        # Before grasping, the cube is an obstacle too. Only the carried-cube
        # phase permits intentional finger/cube contact.
        violations += len(_unintended_contacts(h, d, allow_cube=carrying))
        if carrying:
            for contact in d.contact:
                if contact.dist >= -0.001:
                    continue
                a, b = (int(h.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                if h.cube in (a, b):
                    other = b if a == h.cube else a
                    if other not in h.finger_bodies:
                        violations += 1
    return violations


def optimize_joint_path(waypoints, context, carrying=False):
    """Local derivative-free optimization of a continuous joint spline.

    Existing IK waypoints initialize the search and softly guide it. Endpoints
    and gripper command are fixed. Random proposals shorten/smooth the path and
    penalize collisions; no SciPy or external planner is required. Failed final
    validation rejects the attempt rather than executing an unsafe fallback.
    """
    h, config = context.handles, context.config
    guides = np.asarray(waypoints, dtype=float)
    if guides.ndim != 2 or guides.shape[1] != 6 or len(guides) < 2:
        raise ValueError("Expected at least two six-joint waypoints")
    knots = np.vstack([guides[0]] + [
        start + fraction * (end - start)
        for start, end in zip(guides[:-1], guides[1:])
        for fraction in (0.5, 1.0)
    ])
    reference = knots.copy()
    rng = np.random.default_rng(0)  # reproducible optimization, independent of reset RNG

    def objective(candidate):
        if h.preview is not None:
            h.preview.poll()
        q, v, a, j = _sample_spline(candidate, 65)
        excess = np.maximum(h.lower - q, 0) + np.maximum(q - h.upper, 0)
        return (np.mean(v[:, :5]**2) + 0.005 * np.mean(a[:, :5]**2)
                + 0.00001 * np.mean(j[:, :5]**2)
                + 4 * np.mean((candidate - reference)**2)
                + 1e6 * np.sum(excess**2)
                + 1e4 * _planning_collisions(q, h, carrying))

    score = objective(knots)
    for iteration in range(config.optimizer_iterations):
        candidate = knots.copy()
        index = int(rng.integers(1, len(knots) - 1))
        scale = 0.08 * (1 - iteration / max(1, config.optimizer_iterations)) + 0.005
        candidate[index, :5] += rng.normal(0, scale, 5)
        candidate[index, :5] = np.clip(candidate[index, :5], h.lower[:5], h.upper[:5])
        proposal = objective(candidate)
        if proposal < score:
            knots, score = candidate, proposal
    _, v, a, j = _sample_spline(knots, 4097)
    duration = 1.02 * max(1.0, np.max(np.abs(v)) / config.max_velocity,
                          np.sqrt(np.max(np.abs(a)) / config.max_acceleration),
                          np.cbrt(np.max(np.abs(j)) / config.max_jerk))
    count = int(np.ceil(duration * FPS)) + 1
    trajectory = _sample_spline(knots, count)[0]
    trajectory[[0, -1]] = guides[[0, -1]]
    dense = _sample_spline(knots, max(257, 2 * count))[0]
    if np.any(dense < h.lower - 1e-7) or np.any(dense > h.upper + 1e-7):
        raise PlanningError("Optimized spline exceeds joint limits")
    if _planning_collisions(dense, h, carrying):
        raise PlanningError("Optimized spline failed collision validation")
    if context.steps + count > context.max_steps:
        raise PlanningError("Optimized motion exceeds remaining episode step budget")
    return trajectory


def execute_optimized_path(waypoints, context, carrying=False):
    """Execute one optimized reach/transfer with no holds at interior waypoints."""
    trajectory = optimize_joint_path(waypoints, context, carrying)
    h = context.handles
    for q in trajectory[1:]:
        step_and_record(q, context, allow_cube=carrying)
        if carrying and h.data.xpos[h.cube, 2] < 0.035:
            raise PlanningError("Cube dropped during optimized transfer")
    # Settle only at the actual grasp/place destination, not at intermediate guides.
    for _ in range(round(0.15 * FPS)):
        step_and_record(trajectory[-1], context, allow_cube=carrying)
    if np.max(np.abs(h.data.qpos[h.qadr[:5]] - trajectory[-1, :5])) > 0.12:
        raise PlanningError("Optimized endpoint did not settle")


def run_optimized_episode(h, config, seed, dataset=None, max_steps=1800):
    """Alternative controller; the original run_expert_episode stays unchanged."""
    discard_episode(dataset)
    obs, _ = h.venv.reset(seed=seed)
    context = EpisodeContext(h, config, obs, h.data.qpos[h.qadr].copy(), dataset, max_steps)
    try:
        for _ in range(10):
            step_and_record(context.command, context)
        context.stage = "open"
        move_gripper(config.open_gripper, context)
        context.stage = "optimized_reach"
        scene = read_scene_state(h)
        # Reuse the existing grasp selection without physically moving to approach.
        approach = solve_ik(*get_stage_target("approach", scene, config),
                            context.command, h, config)
        grasp_context = EpisodeContext(h, config, context.obs, approach)
        rotation = choose_grasp(scene, grasp_context)
        waypoints = [context.command.copy(), approach]
        # Pregrasp is now the fixed endpoint, not an adjustable interior knot.
        # The optimizer cannot move it or bend the final grasp descent.
        pos, _ = get_stage_target("pregrasp", scene, config)
        waypoints.append(solve_ik(pos, rotation, waypoints[-1], h, config))
        execute_optimized_path(waypoints, context)
        context.stage = "descend"
        # Keep the original scripted descent outside the optimizer. Re-read the
        # scene and reject cube movement rather than chasing a disturbed target.
        current_scene = read_scene_state(h)
        if np.linalg.norm(current_scene["cube_pos"] - scene["cube_pos"]) > 0.005:
            raise PlanningError("Cube moved before descent; fixed grasp target is stale")
        pos, _ = get_stage_target("descend", scene, config)
        _move_to(pos, rotation, context, allow_cube=True)
        context.stage = "close"
        move_gripper(config.closed_gripper, context)
        before = read_scene_state(h)
        context.stage = "test_lift"
        pos, _ = get_stage_target("test_lift", before, config)
        _move_to(pos, rotation, context, allow_cube=True)
        if not check_grasp(before, read_scene_state(h)):
            raise PlanningError("Cube did not remain grasped during test lift")
        context.stage = "optimized_transfer"
        scene = read_scene_state(h)
        waypoints = [context.command.copy()]
        for stage in ("lift", "transfer", "lower"):
            pos, rot = get_stage_target(stage, scene, config)
            waypoints.append(solve_ik(pos, rot, waypoints[-1], h, config))
        execute_optimized_path(waypoints, context, carrying=True)
        context.stage = "release"
        move_gripper(config.open_gripper, context)
        context.stage = "retreat"
        _move_to(*get_stage_target("retreat", read_scene_state(h), config), context, allow_cube=True)
        context.stage = "validate"
        for _ in range(20):
            step_and_record(context.command, context)
            if not check_placement(h):
                raise PlanningError("Cube is not resting fully inside the box")
        return EpisodeResult(True, seed, context.steps, context.stage)
    except PlanningError as exc:
        discard_episode(dataset)
        return EpisodeResult(False, seed, context.steps, context.stage, str(exc))
    except BaseException:
        discard_episode(dataset)
        raise


@dataclass
class Optimizer2Plan:
    trajectory: np.ndarray
    vertical_end: int
    closure_tail: np.ndarray | None = None
    closing_speed: float = 0.0
    vertical_start: int = 0
    vertical_xy: np.ndarray | None = None


def _jaw_openings(scene, rotation, context):
    """Find cube-sized open and handoff angles using the actual jaw-tip geometry.

    Handoff retains a small gap; contact is handled by the physical closure tail.
    This is a geometric estimate, not a force or friction model.
    """
    h = context.handles
    cube_geom = h.model.body_geomadr[h.cube]
    width = 2 * np.sum(np.abs(rotation[:, 2] @ scene["cube_rotation"])
                       * h.model.geom_size[cube_geom])
    fixed = h.model.geom("fixed_jaw_sph_tip1").id
    moving = h.model.geom("moving_jaw_sph_tip1").id
    d = h.planning_data
    grasp_pos, _ = get_stage_target("descend", scene, context.config)
    grasp_q = solve_ik(grasp_pos, rotation, context.command, h, context.config)
    d.qpos[:] = h.data.qpos
    d.qpos[h.qadr] = grasp_q
    choices = []
    for angle in np.linspace(h.lower[5], h.upper[5], 401):
        d.qpos[h.qadr[5]] = angle
        mujoco.mj_forward(h.model, d)
        axis = d.site_xmat[h.site].reshape(3, 3)[:, 2]
        fixed_offset = np.dot(d.geom_xpos[fixed] - d.site_xpos[h.site], axis)
        if fixed_offset > 0:
            axis = -axis
            fixed_offset = -fixed_offset
        moving_offset = np.dot(d.geom_xpos[moving] - d.site_xpos[h.site], axis)
        # The fixed jaw is offset from the site. Total jaw-to-jaw width alone
        # overestimates the space available to a cube centered on that site.
        gap = 2 * min(-fixed_offset - h.model.geom_size[fixed, 0],
                      moving_offset - h.model.geom_size[moving, 0])
        clear = True
        for contact in d.contact:
            bodies = [int(h.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if h.cube in bodies and any(b in h.robot_bodies for b in bodies) and contact.dist < -0.0001:
                clear = False
                break
        choices.append((angle, gap, clear))
    closed = env_action_to_model_joints(np.array([0, 0, 0, 0, 0, context.config.closed_gripper]), h)[5]
    def opening(clearance):
        candidates = [angle for angle, gap, clear in choices
                      if angle > closed and clear and gap >= width + clearance]
        if not candidates:
            raise PlanningError("Cube does not fit the jaw opening with clearance")
        return min(candidates)
    handoff, opened = opening(0.002), opening(0.008)
    if not closed < handoff < opened:
        raise PlanningError("Insufficient jaw travel for a continuous closing handoff")
    return opened, handoff, closed


def _cartesian_curve(points, rotations, vertical_segment, parameters, tangent_scale):
    """Quintic Hermite segments with shared velocity and acceleration at joins.

    The designated segment has identical XY endpoints and vertical tangents,
    so its commanded Cartesian path is vertical for its entire length.
    """
    lengths = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1), 0.025)
    boundaries = np.r_[0., np.cumsum(lengths)] / lengths.sum()
    velocities = np.zeros_like(points)
    for i in range(1, len(points) - 1):
        velocities[i] = (points[i + 1] - points[i - 1]) / (boundaries[i + 1] - boundaries[i - 1])
    direction = np.sign(points[vertical_segment + 1, 2] - points[vertical_segment, 2])
    for i in (vertical_segment, vertical_segment + 1):
        if i not in (0, len(points) - 1):
            velocities[i] = [0., 0., direction * lengths.sum() * tangent_scale]
    positions, orientations = [], []
    for parameter in parameters:
        i = min(np.searchsorted(boundaries, parameter, side="right") - 1, len(points) - 2)
        i = max(i, 0)
        span = boundaries[i + 1] - boundaries[i]
        u = np.clip((parameter - boundaries[i]) / span, 0, 1)
        v0, v1 = velocities[i] * span, velocities[i + 1] * span
        remainder = points[i + 1] - points[i] - v0
        change = v1 - v0
        p = (points[i] + v0 * u + (10 * remainder - 4 * change) * u**3
             + (-15 * remainder + 7 * change) * u**4 + (6 * remainder - 3 * change) * u**5)
        # Smooth orientation interpolation; vertical segment rotations are equal.
        delta = np.empty(4)
        mujoco.mju_mat2Quat(delta, (rotations[i + 1] @ rotations[i].T).ravel())
        axis_angle = np.empty(3)
        mujoco.mju_quat2Vel(axis_angle, delta, 1.)
        angle = np.linalg.norm(axis_angle)
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        quaternion = np.array([1., 0., 0., 0.])
        if angle > 1e-10:
            mujoco.mju_axisAngle2Quat(quaternion, axis_angle / angle, angle * blend)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        positions.append(p)
        orientations.append(matrix.reshape(3, 3) @ rotations[i])
    return np.asarray(positions), orientations, boundaries


def _optimizer2_ik(points, rotations, vertical_segment, tangent, context, samples):
    h = context.handles
    parameters = np.linspace(0, 1, samples)
    positions, matrices, boundaries = _cartesian_curve(
        points, rotations, vertical_segment, parameters, tangent)
    q = context.command.copy()
    trajectory = [q.copy()]
    # Try for accurate seeds, but don't require execution-level accuracy before
    # CasADi has had a chance to optimize. Approximate seeds are never executed.
    ik_config = replace(context.config, position_tolerance=0.0005, orientation_tolerance=0.1)
    for index, (position, rotation) in enumerate(zip(positions[1:], matrices[1:]), 1):
        if h.preview is not None:
            h.preview.poll()
        try:
            q = solve_ik(position, rotation, q, h, ik_config, seed_limits=(0.005, 0.25))
        except PlanningError as exc:
            raise PlanningError(
                f"IK seed sample {index}/{samples-1} (u={parameters[index]:.4f}) "
                f"exceeds initialization bounds (5 mm, 0.25 rad): {exc}"
            ) from exc
        trajectory.append(q.copy())
    return np.asarray(trajectory), boundaries


def _closing_profile(opened, handoff, closed, config):
    """Cosine velocity ramps meeting at peak speed with zero acceleration.

    Their unequal distances permit a modest opening rather than requiring a
    symmetric, very wide opening just to put peak velocity at the handoff.
    """
    distance = min(opened - handoff, handoff - closed)
    peak = min(config.max_velocity,
               np.sqrt(4 * distance * config.max_acceleration / np.pi),
               np.cbrt(8 * distance**2 * config.max_jerk / np.pi**2))
    before, after = 2 * (opened - handoff) / peak, 2 * (handoff - closed) / peak
    return peak, before, after


def _joint_spline_basis(count, parameters):
    """Clamped cubic interpolation basis and its first three derivatives."""
    spacing = 1. / (count - 1)
    identity = np.eye(count)
    system = np.diag(np.full(count, 4.)) + np.diag(np.ones(count - 1), 1) + np.diag(np.ones(count - 1), -1)
    system[0, 0] = system[-1, -1] = 2
    rhs = np.zeros_like(identity)
    rhs[1:-1] = 6 * np.diff(identity, n=2, axis=0) / spacing**2
    rhs[0] = 6 * (identity[1] - identity[0]) / spacing**2
    rhs[-1] = -6 * (identity[-1] - identity[-2]) / spacing**2
    second = np.linalg.solve(system, rhs)
    parameters = np.clip(np.asarray(parameters), 0, 1)
    index = np.minimum((parameters / spacing).astype(int), count - 2)
    b = ((parameters - index * spacing) / spacing)[:, None]
    a = 1 - b
    left, right = identity[index], identity[index + 1]
    ml, mr = second[index], second[index + 1]
    return (
        a*left + b*right + ((a**3-a)*ml + (b**3-b)*mr)*spacing**2/6,
        (right-left)/spacing + ((1-3*a**2)*ml + (3*b**2-1)*mr)*spacing/6,
        a*ml + b*mr, (mr-ml)/spacing,
    )


@dataclass
class SmoothJointPath:
    knots: np.ndarray

    def evaluate(self, parameters):
        return _joint_spline_basis(len(self.knots), parameters)[0] @ self.knots

    def minimum_duration(self, config):
        clock = np.linspace(0, 1, 4097)
        u = 10*clock**3 - 15*clock**4 + 6*clock**5
        du = (30*clock**2 - 60*clock**3 + 30*clock**4)[:, None]
        ddu = (60*clock - 180*clock**2 + 120*clock**3)[:, None]
        dddu = (60 - 360*clock + 360*clock**2)[:, None]
        _, first, second, third = [basis @ self.knots for basis in _joint_spline_basis(len(self.knots), u)]
        velocity = first * du
        acceleration = second * du**2 + first * ddu
        jerk = third * du**3 + 3*second*du*ddu + first*dddu
        return 1.05 * max(1., np.max(abs(velocity))/config.max_velocity,
                         np.sqrt(np.max(abs(acceleration))/config.max_acceleration),
                         np.cbrt(np.max(abs(jerk))/config.max_jerk))


def _casadi_site_kinematics(h, ca):
    """Symbolic FK for this model's hinge chain, derived from the compiled MJCF.

    Collision geometry stays in MuJoCo. Verify the symbolic transform before
    using it for constraints, so model/reference-frame mismatches fail clearly.
    """
    m = h.model
    chain = []
    body = int(m.site_bodyid[h.site])
    while body:
        chain.append(body)
        body = int(m.body_parentid[body])
    q = ca.MX.sym("arm", 5)
    rotation, position = ca.DM.eye(3), ca.DM.zeros(3)
    dof_to_index = {int(address): i for i, address in enumerate(h.qadr[:5])}
    def constant_rotation(quaternion):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return ca.DM(matrix.reshape(3, 3))
    for body in reversed(chain):
        position = position + rotation @ ca.DM(m.body_pos[body])
        rotation = rotation @ constant_rotation(m.body_quat[body])
        for joint in range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]):
            if m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise PlanningError("CasADi FK requires a hinge-only arm chain")
            address = int(m.jnt_qposadr[joint])
            if address not in dof_to_index:
                raise PlanningError("Unmapped joint in CasADi arm chain")
            angle = q[dof_to_index[address]] - m.qpos0[address]
            x, y, z = m.jnt_axis[joint]
            skew = ca.DM([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            turn = ca.DM.eye(3) + ca.sin(angle)*skew + (1-ca.cos(angle))*(skew @ skew)
            anchor = ca.DM(m.jnt_pos[joint])
            position = position + rotation @ (anchor - turn @ anchor)
            rotation = rotation @ turn
    position = position + rotation @ ca.DM(m.site_pos[h.site])
    rotation = rotation @ constant_rotation(m.site_quat[h.site])
    fk = ca.Function("so101_site_fk", [q], [position, rotation])
    # mj_step's derived transforms can precede its final integration. Recompute
    # on planning_data before comparing FK at the current generalized positions.
    check = h.planning_data
    check.qpos[:] = h.data.qpos
    mujoco.mj_forward(m, check)
    actual_pos, actual_rot = fk(h.data.qpos[h.qadr[:5]])
    if (np.linalg.norm(np.asarray(actual_pos).ravel() - check.site_xpos[h.site]) > 1e-7
            or np.linalg.norm(np.asarray(actual_rot) - check.site_xmat[h.site].reshape(3, 3)) > 1e-7):
        raise PlanningError("Symbolic FK disagrees with MuJoCo; refusing optimization")
    return fk


def _smooth_optimizer2_path(seed, points, rotations, vertical_segment, tangent, context):
    """Fit joint spline with Cartesian constraints using CasADi + IPOPT.

    IK provides an initial guess only. The final path is a C2 joint spline,
    rather than a piecewise-linear sequence of tolerance-limited IK solutions.
    Constraints are collocated; the existing dense MuJoCo checks still apply.
    """
    try:
        import casadi as ca
    except ImportError as exc:
        raise PlanningError("optimizer2 requires CasADi in the active LeRobot environment") from exc
    h = context.handles
    fk = _casadi_site_kinematics(h, ca)
    count = 33
    grid = np.linspace(0, 1, len(seed))
    initial = np.column_stack([np.interp(np.linspace(0, 1, count), grid, seed[:, j]) for j in range(5)])
    # Include every vertical boundary explicitly in the collocation set.
    _, _, boundaries = _cartesian_curve(points, rotations, vertical_segment, [0., 1.], tangent)
    parameters = np.unique(np.r_[np.linspace(0, 1, 129), boundaries])
    targets, target_rotations, _ = _cartesian_curve(points, rotations, vertical_segment, parameters, tangent)
    basis, first, second, third = _joint_spline_basis(count, parameters)
    opt = ca.Opti()
    knots = opt.variable(count, 5)
    curve = ca.DM(basis) @ knots
    opt.subject_to(knots[0, :].T == seed[0, :5])
    # Pin the actual starting configuration, but let the solver repair the
    # approximate endpoint seed. The target pose, not its seed joints, is fixed.
    final_position, final_rotation = fk(knots[-1, :].T)
    opt.subject_to(opt.bounded(-0.0005, final_position-ca.DM(points[-1]), 0.0005))
    opt.subject_to(ca.sumsqr(final_rotation-ca.DM(rotations[-1])) <= 0.025)
    cost = (ca.sumsqr(ca.DM(first) @ knots) + 0.005*ca.sumsqr(ca.DM(second) @ knots)
            + 0.00001*ca.sumsqr(ca.DM(third) @ knots)
            + 0.1*ca.sumsqr(knots-ca.DM(initial))) / len(parameters)
    for i, u in enumerate(parameters):
        arm = curve[i, :].T
        opt.subject_to(opt.bounded(ca.DM(h.lower[:5]), arm, ca.DM(h.upper[:5])))
        position, rotation = fk(arm)
        error = position-ca.DM(targets[i])
        opt.subject_to(opt.bounded(-0.0015, error, 0.0015))
        cost += 0.05*ca.sumsqr(rotation-ca.DM(target_rotations[i])) / len(parameters)
        if boundaries[vertical_segment] <= u <= boundaries[vertical_segment + 1]:
            lateral = position[:2] - ca.DM(points[vertical_segment, :2])
            opt.subject_to(opt.bounded(-0.00075, lateral, 0.00075))
            opt.subject_to(ca.sumsqr(rotation-ca.DM(target_rotations[i])) <= 0.025)
    opt.minimize(cost)
    opt.set_initial(knots, initial)
    opt.solver("ipopt", {"expand": True, "print_time": False},
               {"max_iter": 300, "tol": 1e-6, "print_level": 0, "sb": "yes"})
    if h.preview is not None:
        h.preview.poll()
    try:
        solution = opt.solve()
    except RuntimeError as exc:
        status = opt.stats().get("return_status", "unknown")
        raise PlanningError(f"CasADi spline fit failed ({status})") from exc
    arm_knots = np.asarray(solution.value(knots))
    all_knots = np.column_stack((arm_knots, np.full(count, seed[0, 5])))
    return SmoothJointPath(all_knots)


def _optimizer2_timed_path(path, context, boundaries, vertical_segment, jaw=None):
    """Retiming includes all six joints and the closure tail in sampled limits."""
    config = context.config
    duration = path.minimum_duration(config)
    if jaw is not None:
        opened, handoff, closed = jaw
        peak, closing_before, closing_after = _closing_profile(*jaw, config)
        duration = max(duration, closing_before + 1.0)
    for _ in range(12):
        count = int(np.ceil(duration * FPS))
        duration = count / FPS
        clock = np.linspace(0, 1, count + 1)
        parameter = 10 * clock**3 - 15 * clock**4 + 6 * clock**5
        trajectory = path.evaluate(parameter)
        tail = None
        if jaw is not None:
            time = clock * duration
            close_start = duration - closing_before
            # Finish opening before descent and before the closing ramp begins.
            vertical_start = np.interp(boundaries[vertical_segment], parameter, time)
            opening_duration = min(close_start, vertical_start) * 0.8
            if opening_duration < 0.15:
                duration *= 1.5
                continue
            u = np.clip(time / opening_duration, 0, 1)
            trajectory[:, 5] = path.knots[0, 5] + (opened - path.knots[0, 5]) * (10*u**3 - 15*u**4 + 6*u**5)
            elapsed = np.clip(time - close_start, 0, closing_before)
            closing = opened - peak * (elapsed / 2 - closing_before * np.sin(np.pi * elapsed / closing_before) / (2*np.pi))
            trajectory[time >= close_start, 5] = closing[time >= close_start]
            trajectory[-1, 5] = handoff
            elapsed = np.minimum(np.arange(1, int(np.ceil(closing_after * FPS)) + 1) / FPS, closing_after)
            tail = np.repeat(trajectory[-1][None, :], len(elapsed), axis=0)
            tail[:, 5] = handoff - peak * (elapsed / 2 + closing_after * np.sin(np.pi * elapsed / closing_after) / (2*np.pi))
            tail[-1, 5] = closed
        combined = trajectory if tail is None else np.vstack((trajectory, tail))
        padded = np.vstack((np.repeat(combined[:1], 3, axis=0), combined,
                            np.repeat(combined[-1:], 3, axis=0)))
        velocity = np.diff(padded, axis=0) * FPS
        acceleration = np.diff(velocity, axis=0) * FPS
        jerk = np.diff(acceleration, axis=0) * FPS
        ratio = max(np.max(abs(velocity)) / config.max_velocity,
                    np.sqrt(np.max(abs(acceleration)) / config.max_acceleration),
                    np.cbrt(np.max(abs(jerk)) / config.max_jerk))
        if ratio <= 1.001:
            if context.steps + len(combined) > context.max_steps:
                raise PlanningError(
                    f"optimizer2 needs {len(combined)} steps ({len(combined)/FPS:.1f}s); "
                    f"only {context.max_steps-context.steps} remain"
                )
            vertical_end = int(np.searchsorted(parameter, boundaries[vertical_segment + 1]))
            return Optimizer2Plan(trajectory, vertical_end, tail, peak if jaw is not None else 0.)
        # Jaw peak must also slow down when time scaling is needed.
        duration *= max(1.1, 1.05 * ratio)
        if jaw is not None:
            peak /= max(1.1, 1.05 * ratio)
            closing_before = 2 * (opened - handoff) / peak
            closing_after = 2 * (handoff - closed) / peak
            duration = max(duration, closing_before + 1.)
    raise PlanningError("optimizer2 could not satisfy sampled motion limits")


def optimizer2(points, rotations, vertical_segment, context, carrying=False, jaw=None):
    """Constrained Cartesian path search, retaining both earlier controllers.

    Vertical endpoints/tangents and endpoint orientations are fixed. Only free
    waypoints and the through-velocity of the vertical join are searched. This
    uses a NumPy search for free-space guides followed by a CasADi/IPOPT
    constrained joint-spline fit. The final path is checked with MuJoCo FK; physical
    tracking is checked during execution rather than claimed to be exact.
    """
    h, config = context.handles, context.config
    print(f"[PLAN] optimizer2 {'transfer' if carrying else 'reach'}: searching Cartesian path", flush=True)
    reference = np.asarray(points, dtype=float)
    best, tangent = reference.copy(), 0.45
    fixed = {0, len(points) - 1, vertical_segment, vertical_segment + 1}
    free = [i for i in range(len(points)) if i not in fixed]
    rng = np.random.default_rng(0)
    seed_failures = []
    def objective(candidate, slope):
        try:
            q, _ = _optimizer2_ik(candidate, rotations, vertical_segment, slope, context, 33)
            if jaw is not None:
                q[1:, 5] = jaw[0]  # conservative collision screen at open width
            collision = _planning_collisions(q, h, carrying)
            difference = np.diff(q[:, :5], axis=0)
            return (np.sum(difference**2) + 5 * np.sum(np.diff(difference, axis=0)**2)
                    + 20 * np.sum((candidate - reference)**2) + 1e4 * collision)
        except PlanningError as exc:
            if not seed_failures:
                seed_failures.append(str(exc))
            return float("inf")
    score = objective(best, tangent)
    for iteration in range(config.optimizer_iterations):
        candidate = best.copy()
        slope = np.clip(tangent + rng.normal(0, 0.08), 0.15, 0.7)
        if free:
            index = int(rng.choice(free))
            candidate[index] += rng.normal(0, 0.006, 3)
            candidate[index] = np.clip(candidate[index], reference[index] - 0.025, reference[index] + 0.025)
        value = objective(candidate, slope)
        if value < score:
            best, tangent, score = candidate, slope, value
    if not np.isfinite(score):
        detail = seed_failures[0] if seed_failures else "no finite initialization objective"
        raise PlanningError(f"optimizer2 found no usable IK initialization; first rejection: {detail}")
    path, boundaries = _optimizer2_ik(best, rotations, vertical_segment, tangent, context, 257)
    smooth_path = _smooth_optimizer2_path(path, best, rotations, vertical_segment, tangent, context)
    plan = _optimizer2_timed_path(smooth_path, context, boundaries, vertical_segment, jaw)
    if np.any(plan.trajectory < h.lower - 1e-7) or np.any(plan.trajectory > h.upper + 1e-7):
        raise PlanningError("optimizer2 trajectory exceeds joint limits")
    if _planning_collisions(plan.trajectory, h, carrying):
        raise PlanningError("optimizer2 timed path failed collision validation")
    # Joint interpolation can deviate from Cartesian targets: reject, don't
    # silently call a curved or overshooting descent 'vertical'.
    clock = np.linspace(0, 1, len(plan.trajectory))
    parameter = 10 * clock**3 - 15 * clock**4 + 6 * clock**5
    plan.vertical_start = int(np.searchsorted(parameter, boundaries[vertical_segment]))
    plan.vertical_xy = best[vertical_segment, :2].copy()
    expected, _, _ = _cartesian_curve(best, rotations, vertical_segment, parameter, tangent)
    for q, target, u in zip(plan.trajectory, expected, parameter):
        d = h.planning_data
        d.qpos[:] = h.data.qpos
        d.qpos[h.qadr] = q
        mujoco.mj_forward(h.model, d)
        actual = d.site_xpos[h.site]
        if np.linalg.norm(actual - target) > 0.003:
            raise PlanningError("optimizer2 Cartesian path error exceeds 3 mm")
        if boundaries[vertical_segment] <= u <= boundaries[vertical_segment + 1]:
            if np.linalg.norm(actual[:2] - best[vertical_segment, :2]) > 0.002:
                raise PlanningError("optimizer2 vertical segment deviates laterally by over 2 mm")
    print(f"[PLAN] optimizer2 frames={len(plan.trajectory)} "
          f"closing_peak={plan.closing_speed:.3f} rad/s", flush=True)
    return plan


def _settled_cube_grasp(context):
    """Require opposing jaw forces, settled fingers, and a closing preload.

    Contact alone is not proof of a grasp. The subsequent vertical test lift
    remains the final check that the cube is actually held.
    """
    h = context.handles
    forces = {body: 0.0 for body in h.finger_bodies}
    wrench = np.zeros(6)
    for index, contact in enumerate(h.data.contact):
        a, b = (int(h.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if a == h.cube and b in forces:
            finger = b
        elif b == h.cube and a in forces:
            finger = a
        else:
            continue
        mujoco.mj_contactForce(h.model, h.data, index, wrench)
        forces[finger] += max(0., float(wrench[0]))
    actual = h.data.qpos[h.qadr[5]]
    closing_preload = actual - context.command[5]
    finger_speed = abs(h.data.qvel[h.dof[5]])
    return (all(force > 0.05 for force in forces.values())
            and finger_speed < 0.08 and closing_preload > 0.015)


def grip_in_motion(plan, context):
    """Continue the closing ramp, handing off early when both jaws settle.

    Retain the current position target on handoff: its closing preload supplies
    pressure throughout transfer. Do not jump to the fully closed target.
    With contact_grasp disabled, execute the original full tail and settling hold.
    """
    if plan.closure_tail is None:
        raise ValueError("A reach plan with a closing tail is required")
    h = context.handles
    d = h.planning_data
    d.qpos[:] = h.data.qpos
    d.qpos[h.qadr] = plan.trajectory[-1]
    mujoco.mj_forward(h.model, d)
    if np.linalg.norm(h.data.site_xpos[h.site] - d.site_xpos[h.site]) > 0.004:
        raise PlanningError("Arm has not reached the grasp pose at closing handoff")
    validate_joint_trajectory(np.vstack((context.command, plan.closure_tail)), h, allow_cube=True)
    stable_steps = 0
    for q in plan.closure_tail:
        step_and_record(q, context, allow_cube=True)
        if context.config.contact_grasp:
            stable_steps = stable_steps + 1 if _settled_cube_grasp(context) else 0
            if stable_steps >= 4:
                print("[GRASP] Stable jaw contact; starting transfer with closing pressure held", flush=True)
                return
    # Preserve the original behavior if contact detection never becomes stable.
    for _ in range(round(0.2 * FPS)):
        step_and_record(context.command, context, allow_cube=True)


def _optimizer2_step(q, index, plan, context, carrying=False):
    """Reject physical lateral drift instead of silently accepting it as vertical."""
    step_and_record(q, context, allow_cube=carrying)
    if plan.vertical_start <= index <= plan.vertical_end:
        actual_xy = context.handles.data.site_xpos[context.handles.site, :2]
        if np.linalg.norm(actual_xy - plan.vertical_xy) > 0.004:
            raise PlanningError("Physical vertical tracking drift exceeds 4 mm")


def run_optimizer2_episode(h, config, seed, dataset=None, max_steps=1800):
    discard_episode(dataset)
    obs, _ = h.venv.reset(seed=seed)
    context = EpisodeContext(h, config, obs, h.data.qpos[h.qadr].copy(), dataset, max_steps)
    try:
        for _ in range(10):
            step_and_record(context.command, context)
        scene = read_scene_state(h)
        context.stage = "optimizer2_reach"
        approach_pos, approach_rotation = get_stage_target("approach", scene, config)
        approach_q = solve_ik(approach_pos, approach_rotation, context.command, h, config)
        selection = EpisodeContext(h, config, context.obs, approach_q.copy())
        # Select at the old known-safe opening, then size the executed opening.
        action = model_joints_to_env_action(selection.command, h)
        action[5] = config.open_gripper
        selection.command = env_action_to_model_joints(action, h)
        rotation = choose_grasp(scene, selection)
        jaw = _jaw_openings(scene, rotation, context)
        pre, _ = get_stage_target("pregrasp", scene, config)
        grasp, _ = get_stage_target("descend", scene, config)
        points = np.array([scene["gripper_pos"], approach_pos, pre, grasp])
        rotations = [scene["gripper_rotation"], approach_rotation, rotation, rotation]
        plan = optimizer2(points, rotations, 2, context, jaw=jaw)
        for index, q in enumerate(plan.trajectory[1:], 1):
            _optimizer2_step(q, index, plan, context)
            if np.linalg.norm(h.data.xpos[h.cube] - scene["cube_pos"]) > 0.005:
                raise PlanningError("Cube moved before optimizer2 closing handoff")
        context.stage = "optimizer2_grip"
        grip_in_motion(plan, context)
        before = read_scene_state(h)
        context.stage = "optimizer2_transfer"
        start = before["gripper_pos"].copy()
        lifted = start + [0, 0, 0.035]
        high = start.copy()
        high[2] = config.transfer_height
        over, over_rotation = get_stage_target("transfer", before, config)
        lower, lower_rotation = get_stage_target("lower", before, config)
        points = np.array([start, lifted, high, over, lower])
        rotations = [before["gripper_rotation"], before["gripper_rotation"],
                     transfer_rotation(high), over_rotation, lower_rotation]
        plan = optimizer2(points, rotations, 0, context, carrying=True)
        checked = False
        for index, q in enumerate(plan.trajectory[1:], 1):
            _optimizer2_step(q, index, plan, context, carrying=True)
            if not checked and index >= plan.vertical_end:
                if not check_grasp(before, read_scene_state(h)):
                    raise PlanningError("Cube failed optimizer2 vertical test lift")
                checked = True
            if checked and h.data.xpos[h.cube, 2] < 0.035:
                raise PlanningError("Cube dropped during optimizer2 transfer")
        context.stage = "release"
        # Reopen only to the same cube-sized aperture, not the old wide setting.
        release = context.command.copy()
        release[5] = jaw[0]
        execute_trajectory(generate_joint_trajectory(context.command, release, config), context, allow_cube=True)
        context.stage = "retreat"
        _move_to(*get_stage_target("retreat", read_scene_state(h), config), context, allow_cube=True)
        context.stage = "validate"
        for _ in range(20):
            step_and_record(context.command, context)
            if not check_placement(h):
                raise PlanningError("Cube is not resting fully inside the box")
        return EpisodeResult(True, seed, context.steps, context.stage)
    except PlanningError as exc:
        discard_episode(dataset)
        return EpisodeResult(False, seed, context.steps, context.stage, str(exc))
    except BaseException:
        discard_episode(dataset)
        raise


def save_successful_episode(dataset, result: EpisodeResult):
    """Commit only a validated episode; encode cameras sequentially for portability."""
    if not result.success:
        raise ValueError("Refusing to save an unsuccessful episode")
    if dataset is not None:
        dataset.save_episode(parallel_encoding=False)


def collect_demonstrations(h, config, args, dataset=None) -> list[EpisodeResult]:
    """Try consecutive reset seeds, retaining successes and accounting for failures."""
    results = []
    successes = 0
    try:
        for attempt in range(args.max_attempts):
            runner = {"scripted": run_expert_episode, "optimized": run_optimized_episode,
                      "optimizer2": run_optimizer2_episode}[args.motion_method]
            result = runner(h, config, args.seed + attempt, dataset, args.max_steps)
            if result.success:
                save_successful_episode(dataset, result)
                successes += 1
            results.append(result)
            status = "PASS" if result.success else "REJECT"
            print(f"[{status}] seed={result.seed} steps={result.steps} "
                  f"stage={result.stage} successes={successes}/{args.episodes}"
                  + (f" reason={result.reason}" if result.reason else ""), flush=True)
            if successes >= args.episodes:
                break
    finally:
        # Preserve completed attempt information even if interrupted later.
        if args.report is not None:
            report = {
                "task": TASK_DESCRIPTION,
                "config": asdict(config),
                "mujoco_version": mujoco.__version__,
                "seed_start": args.seed,
                "requested_episodes": args.episodes,
                "recording": dataset is not None,
                "motion_method": args.motion_method,
                "successes": successes,
                "attempts": len(results),
                "failures_by_stage": dict(Counter(r.stage for r in results if not r.success)),
                "results": [asdict(r) for r in results],
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[SUMMARY] successes={successes} attempts={len(results)} "
          f"rate={successes / len(results):.1%}", flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--motion-method", choices=("scripted", "optimized", "optimizer2"), default="scripted",
                        help="Original stages, joint splines, or Cartesian optimizer2 with integrated gripper timing")
    parser.add_argument("--optimizer-iterations", type=int, default=160)
    parser.add_argument("--max-jerk", type=float, default=8.0, help="Optimized motion limit, rad/s³")
    parser.add_argument("--full-grip-close", action="store_true",
                        help="optimizer2: use the previous full closure instead of contact-based handoff")
    parser.add_argument("--repo-id", default="local/so101_expert")
    parser.add_argument("--root", type=Path, help="New dataset directory; must not exist")
    parser.add_argument("--episodes", type=int, default=20, help="Number of successful demos requested")
    parser.add_argument("--seed", type=int, default=1000,
                        help="First reset seed; incremented for every attempt")
    parser.add_argument("--max-attempts", type=int, help="Default: three times --episodes")
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--no-record", action="store_true",
                        help="Run physics and cameras without writing a dataset")
    parser.add_argument("--preview", action="store_true",
                        help="Show live front-camera frames; ESC/window close stops collection")
    parser.add_argument("--report", type=Path, help="JSON attempt report; defaults inside recorded dataset")
    parser.add_argument("--max-velocity", type=float, default=0.8, help="Joint command limit, rad/s")
    parser.add_argument("--max-acceleration", type=float, default=1.6, help="Joint command limit, rad/s²")
    args = parser.parse_args(argv)
    args.max_attempts = args.max_attempts if args.max_attempts is not None else 3 * args.episodes
    if min(args.episodes, args.max_attempts, args.max_steps) <= 0:
        parser.error("Episode, attempt, and step counts must be positive")
    if (not np.isfinite(args.max_velocity) or not np.isfinite(args.max_acceleration)
            or min(args.max_velocity, args.max_acceleration) <= 0):
        parser.error("Motion limits must be finite and positive")
    if args.optimizer_iterations < 1 or not np.isfinite(args.max_jerk) or args.max_jerk <= 0:
        parser.error("Optimizer iterations and jerk limit must be positive and finite")
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    if not args.repo_id.split("/")[-1] or args.repo_id.split("/")[-1] in (".", ".."):
        parser.error("repo-id must end in a dataset name")
    if not args.no_record:
        default_root = Path(__file__).resolve().parents[1] / "datasets" / args.repo_id.split("/")[-1]
        args.root = (args.root or default_root).expanduser().resolve()
        if args.root.exists():
            parser.error(f"Dataset directory already exists: {args.root}; choose a new --root")
        args.report = args.report or args.root / "expert_report.json"
    if args.report is not None:
        args.report = args.report.expanduser().resolve()
        if args.report.exists():
            parser.error(f"Report already exists: {args.report}; choose a new --report")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = ExpertConfig(max_velocity=args.max_velocity, max_acceleration=args.max_acceleration,
                          optimizer_iterations=args.optimizer_iterations, max_jerk=args.max_jerk,
                          contact_grasp=not args.full_grip_close)
    venv, dataset, preview = None, None, None
    try:
        # Establish SDL's desktop window before MuJoCo creates/makes current its
        # offscreen EGL context. Both libraries otherwise share GL driver state.
        if args.preview:
            preview = LivePreview()
        venv = create_environment()
        obs, _ = venv.reset(seed=args.seed)
        h = inspect_environment(venv)
        if preview is not None:
            h.preview = preview
            preview.show(obs, "ready")
        if not args.no_record:
            args.root.parent.mkdir(parents=True, exist_ok=True)
            dataset = build_dataset(args.repo_id, args.root)
        print(f"[START] task=red_cube fps={FPS} recording={dataset is not None}", flush=True)
        results = collect_demonstrations(h, config, args, dataset)
        return 0 if sum(r.success for r in results) >= args.episodes else 2
    except KeyboardInterrupt:
        print("[STOP] Interrupted; pending attempt discarded", flush=True)
        return 130
    finally:
        try:
            if dataset is not None:
                try:
                    discard_episode(dataset)
                finally:
                    dataset.finalize()
        finally:
            try:
                if venv is not None:
                    venv.close()
            finally:
                if preview is not None:
                    preview.close()


if __name__ == "__main__":
    raise SystemExit(main())
