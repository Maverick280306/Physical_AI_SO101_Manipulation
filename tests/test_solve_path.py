"""Run with the LeRobot Python: python -m unittest discover -s tests -v.

Uses the cached Hub environment and real MuJoCo physics; no dataset is written
except by the explicit temporary-dataset integration test.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import solve_path as sp


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.venv = sp.create_environment()
        cls.venv.reset(seed=1000)
        cls.h = sp.inspect_environment(cls.venv)

    @classmethod
    def tearDownClass(cls):
        cls.venv.close()

    def setUp(self):
        self.obs, _ = self.venv.reset(seed=1000)

    def test_coordinate_mapping(self):
        h = self.h
        q = h.data.qpos[h.qadr].copy()
        action = sp.model_joints_to_env_action(q, h)
        np.testing.assert_allclose(action, self.obs["agent_pos"][0], atol=1e-5)
        np.testing.assert_allclose(sp.env_action_to_model_joints(action, h), q, atol=1e-7)
        for fraction in (0, 0.25, 0.5, 1):
            q = h.lower + fraction * (h.upper - h.lower)
            action = sp.model_joints_to_env_action(q, h)
            np.testing.assert_allclose(h.world.to_ctrl(action), q, atol=2e-7)
        with self.assertRaises(ValueError):
            sp.env_action_to_model_joints(np.zeros(5), h)

    def test_scene_snapshot(self):
        scene = sp.read_scene_state(self.h)
        self.assertEqual(scene["cube_rotation"].shape, (3, 3))
        self.assertAlmostEqual(scene["cube_pos"][2], 0.015)
        self.assertFalse(np.shares_memory(scene["joints"], self.h.data.qpos))

    def test_ik_reachable_and_no_live_mutation(self):
        h = self.h
        before = h.data.qpos.copy()
        d = h.planning_data
        q = h.data.qpos[h.qadr].copy()
        q[:5] += [0.08, 0.1, -0.12, 0.08, 0.15]
        d.qpos[:] = h.data.qpos
        d.qpos[h.qadr] = q
        sp.mujoco.mj_forward(h.model, d)
        pos = d.site_xpos[h.site].copy()
        rot = d.site_xmat[h.site].reshape(3, 3).copy()
        solved = sp.solve_ik(pos, rot, before[h.qadr], h, sp.ExpertConfig())
        d.qpos[h.qadr] = solved
        sp.mujoco.mj_forward(h.model, d)
        self.assertLess(np.linalg.norm(d.site_xpos[h.site] - pos), 0.0015)
        np.testing.assert_array_equal(h.data.qpos, before)
        self.assertTrue(np.all(solved >= h.lower) and np.all(solved <= h.upper))

    def test_unreachable_ik(self):
        with self.assertRaises(sp.PlanningError):
            sp.solve_ik([2., 0., 1.], sp.top_down_rotation(0),
                        self.h.data.qpos[self.h.qadr], self.h, sp.ExpertConfig())

    def test_trajectory_bounds(self):
        config = sp.ExpertConfig()
        start = self.h.data.qpos[self.h.qadr].copy()
        goal = start + [0.5, 0.2, -0.3, 0.1, 0.7, 0.8]
        trajectory = sp.generate_joint_trajectory(start, goal, config)
        np.testing.assert_allclose(trajectory[[0, -1]], [start, goal], atol=1e-12)
        velocity = np.diff(trajectory, axis=0) * sp.FPS
        acceleration = np.diff(velocity, axis=0) * sp.FPS
        self.assertLessEqual(np.max(np.abs(velocity)), config.max_velocity)
        self.assertLessEqual(np.max(np.abs(acceleration)), config.max_acceleration)
        stationary = sp.generate_joint_trajectory(start, start, config)
        np.testing.assert_allclose(stationary, np.broadcast_to(start, stationary.shape))

    def test_trajectory_validation(self):
        h = self.h
        start = h.data.qpos[h.qadr].copy()
        goal = start.copy()
        goal[0] += 0.05
        before = h.data.qpos.copy()
        sp.validate_joint_trajectory(sp.generate_joint_trajectory(start, goal, sp.ExpertConfig()), h)
        np.testing.assert_array_equal(h.data.qpos, before)
        with self.assertRaises(sp.PlanningError):
            sp.validate_joint_trajectory(np.array([h.upper + 0.1]), h)
        # Folded well below the table: a real geometry collision, not a mock.
        q = start.copy()
        q[1:4] = [1.5, 0.5, 0.5]
        with self.assertRaisesRegex(sp.PlanningError, "collision"):
            sp.validate_joint_trajectory(np.array([q]), h)

    def test_execution_and_step_limit(self):
        h = self.h
        start = h.data.qpos[h.qadr].copy()
        context = sp.EpisodeContext(h, sp.ExpertConfig(), self.obs, command=start)
        goal = start.copy()
        goal[0] += 0.04
        sp.execute_trajectory(sp.generate_joint_trajectory(start, goal, context.config), context)
        self.assertGreater(context.steps, 10)
        self.assertLess(abs(h.data.qpos[h.qadr[0]] - goal[0]), 0.03)
        context.max_steps = context.steps
        with self.assertRaisesRegex(sp.PlanningError, "step limit"):
            sp.step_and_record(goal, context)

    def test_environment_truncation(self):
        raw = self.venv.envs[0].unwrapped
        original = raw._max_episode_steps
        raw._max_episode_steps = 1
        try:
            q = self.h.data.qpos[self.h.qadr].copy()
            context = sp.EpisodeContext(self.h, sp.ExpertConfig(), self.obs, command=q)
            with self.assertRaisesRegex(sp.PlanningError, "truncated"):
                sp.step_and_record(q, context)
        finally:
            raw._max_episode_steps = original

    def test_expert_physics(self):
        # Controller/contact regression with real physics; omit RGB rendering here.
        # Camera rendering and saved video decoding are checked separately.
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(self.h.world, "render", return_value=pixels):
            result = sp.run_expert_episode(self.h, sp.ExpertConfig(), seed=1000)
        self.assertTrue(result.success, result)
        self.assertTrue(sp.check_placement(self.h))

    def test_grasp_rejects_empty_lift(self):
        before = {"cube_pos": np.array([.2, 0, .015]), "gripper_pos": np.array([.2, 0, .015])}
        empty = {"cube_pos": before["cube_pos"], "gripper_pos": np.array([.2, 0, .05])}
        self.assertFalse(sp.check_grasp(before, empty))
        held = {"cube_pos": np.array([.2, 0, .05]), "gripper_pos": np.array([.2, 0, .05])}
        self.assertTrue(sp.check_grasp(before, held))

    def test_record_discard_save_and_reload(self):
        import datasets.config
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory(prefix="solve-path-test-") as tmp:
            root = Path(tmp) / "dataset"
            dataset = sp.build_dataset("local/expert_test", root)
            try:
                # A timed-out attempt must leave no frames or episode behind.
                result = sp.run_expert_episode(self.h, sp.ExpertConfig(), 1000, dataset, max_steps=2)
                self.assertFalse(result.success)
                self.assertEqual(dataset.num_episodes, 0)
                self.assertFalse(dataset.has_pending_frames())
                self.assertFalse(list(root.rglob("*.png")))

                obs, _ = self.venv.reset(seed=1000)
                q = self.h.data.qpos[self.h.qadr].copy()
                context = sp.EpisodeContext(self.h, sp.ExpertConfig(), obs, q, dataset)
                initial_state = obs["agent_pos"][0].copy()
                initial_image = obs["pixels"]["front"][0].copy()
                for _ in range(12):
                    sp.step_and_record(q, context)
                with self.assertRaises(ValueError):
                    sp.save_successful_episode(dataset, result)
                # Exercise saving with a synthetic result only in this pipeline test.
                sp.save_successful_episode(dataset, sp.EpisodeResult(True, 1000, 12, "test"))
            finally:
                dataset.finalize()

            # Keep the reader's Arrow cache inside the temporary test directory.
            with patch.object(datasets.config, "HF_DATASETS_CACHE", Path(tmp) / "hf_cache"):
                loaded = LeRobotDataset("local/expert_test", root=root, video_backend="pyav")
            self.assertEqual(loaded.num_episodes, 1)
            self.assertEqual(loaded.num_frames, 12)
            self.assertEqual(loaded.fps, 30)
            frame = loaded[0]
            np.testing.assert_allclose(frame["observation.state"].numpy(), initial_state, atol=1e-5)
            np.testing.assert_allclose(frame["action"].numpy(),
                                       sp.model_joints_to_env_action(q, self.h), atol=1e-5)
            self.assertEqual(frame["task"], sp.TASK_DESCRIPTION)
            for camera in ("front", "wrist"):
                self.assertEqual(tuple(frame[f"observation.images.{camera}"].shape), (3, 480, 640))
            decoded = frame["observation.images.front"].numpy().transpose(1, 2, 0) * 255
            self.assertLess(np.abs(decoded - initial_image.astype(float)).mean(), 8)

    def test_cli_validation(self):
        args = sp.parse_args(["--no-record", "--episodes", "2"])
        self.assertEqual(args.max_attempts, 6)
        for options in (["--episodes", "0"], ["--max-velocity", "nan"],
                        ["--root", "/tmp"], ["--repo-id", "local/.."]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sp.parse_args(options)

    def test_attempt_report_and_partial_exit(self):
        with tempfile.TemporaryDirectory(prefix="solve-path-report-") as tmp:
            path = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = sp.main(["--no-record", "--episodes", "1", "--max-attempts", "2",
                                  "--max-steps", "2", "--report", str(path)])
            self.assertEqual(status, 2)
            report = json.loads(path.read_text())
            self.assertEqual(report["successes"], 0)
            self.assertEqual(report["attempts"], 2)
            self.assertEqual([r["seed"] for r in report["results"]], [1000, 1001])
            self.assertFalse(report["recording"])


if __name__ == "__main__":
    unittest.main()
