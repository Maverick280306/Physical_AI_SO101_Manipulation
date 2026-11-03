#!/usr/bin/env python
"""Teleoperate the SO-101 tabletop MuJoCo env and record a LeRobotDataset of demos.

Live control (default): opens a pygame window showing the front camera feed.
Keyboard mapping (hold/tap, deltas applied per-step in degrees, gripper in 0-100):

    q / a   shoulder_pan   -/+
    w / s   shoulder_lift  -/+
    e / d   elbow_flex     -/+
    r / f   wrist_flex     -/+
    t / g   wrist_roll     -/+
    y / h   gripper close/open
    SPACE   end episode as SUCCESS and save it
    x       discard current episode (no save), reset
    ESC     quit and finalize the dataset

Self-test mode (--scripted N): skips human input, drives the arm with a fixed
sinusoidal joint sweep for N episodes. Used to validate the env -> dataset ->
training pipeline mechanically before any real teleoperation session.

Usage:
    python teleop_collect.py --repo-id local/so101_pick_place --episodes 10
    python teleop_collect.py --repo-id local/so101_smoketest --scripted 2
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.envs.factory import make_env  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.feature_utils import (  # noqa: E402
    OBS_STR,
    ACTION,
    hw_to_dataset_features,
    build_dataset_frame,
    combine_feature_dicts,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HOME = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0], dtype=np.float32)
STEP_DEG = 2.0  # per-key-press delta; gripper uses the same units (0-100 range)
FPS = 30
TASK_DESCRIPTION = "Put all four cubes in the green box."


def build_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    obs_hw = {f"{j}.pos": float for j in JOINTS}
    obs_hw["front"] = (480, 640, 3)
    obs_hw["wrist"] = (480, 640, 3)
    action_hw = {f"{j}.pos": float for j in JOINTS}

    obs_features = hw_to_dataset_features(obs_hw, OBS_STR, use_video=True)
    action_features = hw_to_dataset_features(action_hw, ACTION, use_video=True)
    features = combine_feature_dicts(obs_features, action_features)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        image_writer_threads=4,
    )


def obs_to_values(obs) -> dict:
    """Flatten the env's batched obs dict (n_envs=1) into named scalars + frames."""
    pos = np.asarray(obs["agent_pos"])[0]
    values = {f"{j}.pos": float(pos[i]) for i, j in enumerate(JOINTS)}
    values["front"] = np.asarray(obs["pixels"]["front"])[0]
    values["wrist"] = np.asarray(obs["pixels"]["wrist"])[0]
    return values


def action_to_values(action: np.ndarray) -> dict:
    return {f"{j}.pos": float(action[i]) for i, j in enumerate(JOINTS)}


def run_scripted(venv, dataset, episodes: int, max_steps: int):
    """Deterministic joint sweep, no human input. Validates the full write path."""
    for ep in range(episodes):
        obs, _ = venv.reset(seed=ep)
        cur = HOME.copy()
        for step in range(max_steps):
            t = step / FPS
            delta = np.array(
                [8 * np.sin(2 * np.pi * 0.3 * t + i) for i in range(len(JOINTS))], dtype=np.float32
            )
            target = np.clip(cur + delta * 0.05, venv.action_space.low[0], venv.action_space.high[0])
            action = target.reshape(1, -1)

            obs_frame = build_dataset_frame(dataset.features, obs_to_values(obs), prefix=OBS_STR)
            obs, reward, terminated, truncated, info = venv.step(action)
            action_frame = build_dataset_frame(dataset.features, action_to_values(target), prefix=ACTION)

            frame = {**obs_frame, **action_frame, "task": TASK_DESCRIPTION}
            dataset.add_frame(frame)

            if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
                break
        dataset.save_episode()
        print(f"[scripted] episode {ep + 1}/{episodes} saved ({step + 1} steps)")


def run_live(venv, dataset, episodes: int, max_steps: int):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("SO-101 teleop -- SPACE=save ep, x=discard, ESC=quit")
    clock = pygame.time.Clock()

    key_deltas = {
        pygame.K_q: (0, -1), pygame.K_a: (0, +1),
        pygame.K_w: (1, -1), pygame.K_s: (1, +1),
        pygame.K_e: (2, -1), pygame.K_d: (2, +1),
        pygame.K_r: (3, -1), pygame.K_f: (3, +1),
        pygame.K_t: (4, -1), pygame.K_g: (4, +1),
        pygame.K_y: (5, -1), pygame.K_h: (5, +1),
    }

    saved = 0
    ep = 0
    while saved < episodes:
        obs, _ = venv.reset(seed=1000 + ep)
        cur = HOME.copy()
        step = 0
        discarded = False
        while step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return saved

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                pygame.quit()
                return saved
            if keys[pygame.K_x]:
                discarded = True
                break

            delta = np.zeros(len(JOINTS), dtype=np.float32)
            for key, (idx, sign) in key_deltas.items():
                if keys[key]:
                    delta[idx] += sign * STEP_DEG
            target = np.clip(cur + delta, venv.action_space.low[0], venv.action_space.high[0])
            action = target.reshape(1, -1)

            obs_frame = build_dataset_frame(dataset.features, obs_to_values(obs), prefix=OBS_STR)
            obs, reward, terminated, truncated, info = venv.step(action)
            action_frame = build_dataset_frame(dataset.features, action_to_values(target), prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": TASK_DESCRIPTION})
            cur = target

            frame_rgb = np.asarray(obs["pixels"]["front"])[0]
            surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if keys[pygame.K_SPACE]:
                break
            if bool(np.asarray(terminated)[0]):
                break
            step += 1
            clock.tick(FPS)

        if discarded:
            print(f"episode discarded ({step} steps)")
        else:
            dataset.save_episode()
            saved += 1
            print(f"episode saved: {saved}/{episodes}")
        ep += 1

    pygame.quit()
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. local/so101_pick_place")
    p.add_argument("--root", default=None, help="dataset root dir (default: dataset cache)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600, help="20s at 30fps per episode cap")
    p.add_argument("--scripted", type=int, default=0, help="N scripted episodes, skip live input")
    args = p.parse_args()

    root = Path(args.root) if args.root else Path.home() / "manipulation-vertical" / "datasets" / args.repo_id.split("/")[-1]
    root.parent.mkdir(parents=True, exist_ok=True)

    envs = make_env("lerobot/so101-tabletop-mujoco", n_envs=1, trust_remote_code=True)
    venv = envs["so101_tabletop"][0]

    dataset = build_dataset(args.repo_id, root)

    try:
        if args.scripted:
            run_scripted(venv, dataset, args.scripted, args.max_steps)
        else:
            n = run_live(venv, dataset, args.episodes, args.max_steps)
            print(f"Saved {n} episodes.")
    finally:
        dataset.finalize()
        venv.close()
        print(f"Dataset written to: {root}")


if __name__ == "__main__":
    main()
