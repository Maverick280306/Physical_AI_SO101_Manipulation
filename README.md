# Physical AI — SO-101 Manipulation Vertical

This is the simulation-first track for our manipulation vertical: validate a full
imitation-learning pipeline on LeRobot's SO-101 arm in MuJoCo *before* the real
SO-101 hardware arrives, so we hit the ground running once it does. This repo
holds the scripts and docs; the actual datasets and model checkpoints you
produce are **local to your own machine** (`datasets/`, `outputs/` are
gitignored — they're large binary artifacts, don't try to commit them).

If anything here is unclear or breaks on your setup, say so in an issue/message
rather than guessing — this was smoke-tested on one Linux laptop (CPU-only, no
GPU) and your environment may differ.

## 1. Setup

You need two things cloned side by side: upstream `lerobot` itself, and this
repo for the scripts.

```bash
# 1. Clone upstream lerobot and pin to the tested version
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot && git checkout v0.6.1

# 2. Create an isolated Python 3.12 env (lerobot main requires >=3.12)
conda create -y -n lerobot312 python=3.12
conda activate lerobot312

# 3. Install lerobot + the extras we need (training, dataset tools, viz, gamepad)
pip install -e ".[training,dataset,viz,gamepad]"

# 4. MuJoCo isn't bundled -- install it separately
pip install "mujoco>=3.0.0,<4.0.0"

# 5. Force offscreen rendering -- a native/onscreen GL context will crash
#    (segfault / Trace trap) on most headless or driver-less laptops.
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl   # also set it now for this shell

# 6. Clone THIS repo next to it
git clone <this-repo-url> ~/manipulation-vertical
```

**Sanity check** — this should print the SO-101 env's observation/action spaces
with no errors:

```bash
cd ~/manipulation-vertical
python -c "
from lerobot.envs.factory import make_env
envs = make_env('lerobot/so101-tabletop-mujoco', n_envs=1, trust_remote_code=True)
venv = envs['so101_tabletop'][0]
obs, _ = venv.reset(seed=0)
print('OK:', venv.observation_space, venv.action_space)
"
```

If it crashes with a native crash / core dump instead of an error message,
you forgot step 5 — `export MUJOCO_GL=egl` and retry.

If `pip install -e lerobot` complains about the Python version, you're on an
old conda env — make a fresh one with `python=3.12` as above, don't try to
reuse an existing 3.10/3.11 env.

## 2. The environment

`lerobot/so101-tabletop-mujoco` ([model card](https://huggingface.co/lerobot/so101-tabletop-mujoco)):
a 6-DOF SO-101 arm, four colored cubes (red/green/blue/yellow), one open box,
on a tabletop. Observations are `agent_pos` (6 joint angles, degrees) plus
`front` and `wrist` camera RGB frames (480×640). Actions are absolute joint
position targets at 30 fps. Reward is the fraction of cubes in the box;
episode succeeds when all four are in.

**For this task, ignore the "all four cubes" framing** — see the task below.

## 3. Collect demonstrations

Use `scripts/teleop_collect.py`. It opens a window showing the live front
camera and lets you drive the arm with the keyboard:

| Key | Joint | Key | Joint |
|---|---|---|---|
| `q` / `a` | shoulder_pan ∓ | `r` / `f` | wrist_flex ∓ |
| `w` / `s` | shoulder_lift ∓ | `t` / `g` | wrist_roll ∓ |
| `e` / `d` | elbow_flex ∓ | `y` / `h` | gripper close/open |

- **SPACE** — end the current episode as a success and save it
- **x** — discard the current episode (mistake / knocked something over) and retry
- **ESC** — finalize the dataset and quit

```bash
conda activate lerobot312
cd ~/manipulation-vertical/scripts
python teleop_collect.py --repo-id local/so101_pick_place --episodes 20
```

This writes a dataset to `~/manipulation-vertical/datasets/so101_pick_place/`.

**Collection notes:**
- Go slowly and deliberately — a jerky, inconsistent demo teaches the policy
  a jerky, inconsistent habit. Twenty clean, similar demos beat fifty messy,
  wildly-varying ones.
- Keep the cube's *starting position* varied a bit between episodes (don't
  reset to the exact same spot every time) — a policy trained on one fixed
  start position learns to reproduce a trajectory, not to actually look at
  where the cube is.
- If you discard more than ~1 in 4 episodes, slow down further before
  continuing.

## 4. Train ACT on your demos

```bash
lerobot-train \
  --dataset.repo_id=local/so101_pick_place \
  --dataset.root=~/manipulation-vertical/datasets/so101_pick_place \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --output_dir=~/manipulation-vertical/outputs/act_pick_place \
  --steps=20000 --batch_size=8 --num_workers=0 --wandb.enable=false
```

On CPU, expect roughly **5–8 seconds per step** (measured on one laptop with
no GPU) — 20k steps is a multi-hour-to-overnight run. If you've got a GPU,
`--policy.device=cuda` will be dramatically faster; say so and we'll route
heavier training through Advait's GPU-accelerated-RL setup instead of your
own laptop.

Checkpoints land in `outputs/act_pick_place/checkpoints/`.

## 5. Evaluate what you trained

```bash
lerobot-eval \
  --policy.path=~/manipulation-vertical/outputs/act_pick_place/checkpoints/last/pretrained_model \
  --env.type=... # TODO once eval config for hub envs is confirmed -- flag if this doesn't work out of the box
  --eval.n_episodes=20
```

(`lerobot-eval`'s support for Hub-hosted envs hasn't been exercised yet on
this repo — if it doesn't accept `lerobot/so101-tabletop-mujoco` directly,
write a short eval loop yourself using the same `make_env(...)` call from
step 1's sanity check, run N episodes with the trained policy, and report
success rate. Don't silently skip evaluation — a trained checkpoint nobody
measured isn't a result.)

## 6. The task

**Single-task pick-and-place: pick up the red cube and place it in the box.**
Not all four cubes — that's a much longer-horizon task and a bad first target.
Ignore the other three cubes entirely; don't worry if they're in the way,
just don't let starting state vary so much that the red cube is sometimes
blocked.

## 7. What we're actually trying to learn (the end goal)

The deliverable isn't "a policy exists." It's an honest answer to: **does ACT,
trained on ~20 teleop demos, reach a real single-task success rate in sim, and
what does it take to get there?**

Concretely, report back:

1. **Success rate** over 20 held-out eval episodes (different seeds than any
   training demo), not just "it worked when I watched it once."
2. **Where it fails**, when it does — stalls short of the cube, grasps but
   drops it, knocks it past the box, never approaches at all. This matters
   more than the raw number: it tells us whether the problem is more demos,
   cleaner demos, or something structural (action scaling, chunk size).
3. **How many demos it actually took** to cross some reasonable bar (say,
   50%+ success) — if 20 isn't enough, say so and try 40; we need the real
   data-efficiency curve, not a guess.

This result feeds directly into the next phase (chaining vs. single
task-conditioned policies, then sim-to-real transfer once hardware arrives) —
so a clearly-reported failure is just as useful to us as a clean success.
Don't polish the number; report what actually happened.

## Questions / blockers

If something in this README doesn't match what you see on your machine, that's
expected to happen at least once — this was only tested on one setup. Report
the exact error rather than working around it silently; a wrong assumption
baked in quietly now costs more to find later.
