# SO-101 pick-and-place demonstrations for ACT

This repository collects simulation demonstrations of an SO-101 arm picking up
**the red cube and placing it in the green box**. The other three cubes are
obstacles. Demonstrations are saved locally as LeRobot datasets for ACT training.

## Environment and dependencies

Use the project's Conda environment for every command. On the current machine:

```bash
conda activate lerobot
cd ~/manipulation-vertical
```

The scripts require Python 3.12, a compatible LeRobot checkout, NumPy, MuJoCo,
CasADi (with IPOPT), and pygame for optional preview. The current setup uses an
editable LeRobot checkout at `~/lerobot`. Dataset writing uses its
`LeRobotDataset.create` API, writer/image-writer helpers, and feature utilities.
LeRobot API compatibility matters; the older README's version pin has not been
validated against the current collector.

To install dependencies into an activated project environment when setting up
another machine:

```bash
cd ~/lerobot
python -m pip install -e ".[training,dataset]"
python -m pip install "mujoco>=3,<4" "casadi>=3.7,<4" pygame
```

The collector loads `lerobot/so101-tabletop-mujoco` through LeRobot's Hub loader
with `trust_remote_code=True`. Initial use needs the environment code and assets
available from Hugging Face; subsequent runs can use the local cache. It expects
the SO-101 model's named joints, gripper site, jaw geometry, cube, and box.

MuJoCo defaults to offscreen EGL rendering (`MUJOCO_GL=egl`). A live preview uses
pygame to display the front camera frames; it does not replace EGL with an
interactive MuJoCo viewer.

## Collect demonstrations

The single controller in `scripts/solve_path.py` is `optimizer`; no method
selection flag is needed.

Test without writing a dataset:

```bash
python scripts/solve_path.py --no-record --episodes 1 --seed 1000
```

Display the live front camera:

```bash
python scripts/solve_path.py --no-record --episodes 1 --seed 1000 --preview
```

On desktops where SDL selects an incompatible Wayland backend, explicitly select
X11 (requires a working X11/XWayland display):

```bash
SDL_VIDEODRIVER=x11 SDL_RENDER_DRIVER=software SDL_FRAMEBUFFER_ACCELERATION=0 \
python scripts/solve_path.py --no-record --episodes 1 --preview
```

Record four successful demonstrations:

```bash
python scripts/solve_path.py --repo-id local/act_batch_01 --episodes 4 --seed 1050
```

The default output directory is `datasets/act_batch_01/`, relative to this
repository. `--root` selects another directory. **The directory must not already
exist:** the collector does not append to or resume existing datasets. Nothing
is uploaded to Hugging Face.

`--episodes` counts successes. Each attempt increments the seed, including
failures; each invocation starts at the supplied seed (default 1000). By default,
the collector allows three attempts per requested success. Use `--max-attempts`
to change that limit. It returns exit code 2 if it cannot collect the requested
number. The default per-attempt budget is 1800 steps, or 60 simulated seconds.
Rendering and planning can make wall-clock collection substantially longer.

Press **Ctrl+C** to stop. With preview enabled, **ESC** or window close also stops
collection. Pending attempts are discarded; completed saved episodes are retained.
Native solver calls may take time to return before Python handles an interrupt.

Record smaller batches using different dataset names and seed ranges, then merge
them with LeRobot's dataset tools before training. Copying dataset files together
manually will not update their episode indices or video references correctly.

## Controller and data format

The optimizer searches Cartesian guide paths, initializes them with bounded
approximate IK, then uses CasADi/IPOPT to fit a continuous joint spline. Descent
and initial lift have vertical Cartesian constraints. Timing accounts for joint
velocity, acceleration, and jerk. MuJoCo checks the resulting path for collisions;
the carried cube is approximated as rigidly held during planning, while actual
execution uses physical finger contact.

The gripper opens during reach to a cube-sized aperture. Closure continues
through a peak-speed handoff and can finish early after stable opposing jaw
contact. The lift verifies that the cube is held. Placement is accepted only after
the released cube rests inside the box. `--full-grip-close` disables the early
contact-based handoff. These checks do not guarantee success for every reset;
constraints are sampled, and unsuccessful attempts are reported and discarded.

Use `python scripts/solve_path.py --help` for motion limits and other options.
The preview can remain stationary while planning: simulation does not advance
while a path is being solved.

Each saved timestep contains:

- `observation.state`: five arm angles in degrees and gripper opening in 0–100.
- `observation.images.front` and `.wrist`: 480 × 640 RGB camera frames.
- `action`: absolute joint targets in the same units as the state.
- LeRobot timestamps, frame/episode indices, and the red-cube task description.

Sampling is 30 simulated FPS. Numeric data is stored in Parquet, camera streams
in MP4, and schema/statistics/episode information under `meta/`.
`expert_report.json` records controller settings, completed attempt seeds,
successes, and failure reasons. `--no-record` writes no dataset; an explicit
`--report` can still save the attempt report.

`datasets/` and training `outputs/` are gitignored local artifacts.

## Repository structure

- `scripts/solve_path.py`: optimizer, preview, execution, and collection CLI.
- `scripts/teleop_collect.py`: original keyboard collector and shared recording
  helpers imported by `solve_path.py`.
- `tests/test_solve_path.py`: coordinate, IK, motion-limit, recording, and CLI checks.

`teleop_collect.py` remains available separately. Its original task label and
keyboard workflow differ from the optimizer's red-cube controller.

## Train and evaluate ACT

For a compatible LeRobot installation, an initial training command is:

```bash
lerobot-train \
  --dataset.repo_id=local/act_training \
  --dataset.root="$PWD/datasets/act_training" \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --output_dir="$PWD/outputs/act_pick_place" \
  --steps=20000 --batch_size=8 --num_workers=0 --wandb.enable=false
```

Use the actual collected or merged dataset name and directory. CUDA can be
selected with `--policy.device=cuda` when supported by the environment. A complete
ACT training/evaluation run has not been validated by this repository's tests.

Evaluate on reset seeds excluded from collection and report the red-cube task's
success rate and failure types. The Hub environment's built-in success condition
requires **all four cubes**, so it is not the single-red-cube evaluation criterion.
There is currently no dedicated trained-policy evaluation script in this repo.

## Checks

Run checks in the project environment:

```bash
python -B -m unittest discover -s tests -v
```

These require the cached Hub environment and offscreen rendering. Recording
checks create temporary datasets and decode their videos; they do not execute a
full optimizer demonstration or establish policy success rates.
