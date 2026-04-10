# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/tienkung_lite/npz/{motion_name}.npz --headless --log_project_name
#若不想使用wandb，可以去掉 --logger wandb
#继续训练 --resume {load_run_name}
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--motion_file",type=str,default=None,help="Path to the motion file (e.g., .npz) to load.")
parser.add_argument("--motion_files", type=str, nargs="+", default=None,
                    help="Paths to multiple motion files for multi-motion training.")
parser.add_argument("--motion_selector", type=str, default="uniform",
                    choices=["uniform", "adaptive", "curriculum"],
                    help="Motion selection strategy for multi-motion training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Prefer the package sources from this repository over any stale editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_ROOT = REPO_ROOT / "source" / "whole_body_tracking"
if LOCAL_PACKAGE_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from glob import glob

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _as_dict(obj):
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {}


def _to_plain_dict(data):
    if isinstance(data, dict):
        return {k: _to_plain_dict(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_plain_dict(v) for v in data]
    if hasattr(data, "to_dict"):
        return _to_plain_dict(data.to_dict())
    return data


def _latest_checkpoint(run_dir: str) -> str | None:
    patterns = ["model_*.pt", "model.pt"]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob(os.path.join(run_dir, pattern)))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def _apply_motion_settings(env_cfg, args_cli, motion_files: list[str] | None, motion_selector: str | None) -> list[str] | None:
    """Apply motion source in priority: stage_cfg > CLI > single-motion CLI."""
    selected_files = motion_files if motion_files is not None else args_cli.motion_files
    selector = motion_selector if motion_selector is not None else args_cli.motion_selector

    # Default single-motion override when explicitly provided.
    if selected_files is None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file if args_cli.motion_file else None
        return None

    from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommandCfg

    if isinstance(env_cfg.commands.motion, MultiMotionCommandCfg):
        env_cfg.commands.motion.motion_files = selected_files
        env_cfg.commands.motion.motion_selector_type = selector
        print(
            f"[INFO] Multi-motion training with {len(selected_files)} motions "
            f"(selector: {selector})"
        )
    else:
        env_cfg.commands.motion.motion_file = selected_files[0]
        print(
            f"[WARN] motion_files provided but env uses single-motion config. "
            f"Using first file: {selected_files[0]}"
        )
    return selected_files


def _build_log_dir(agent_cfg, stage_name: str | None = None) -> tuple[str, str]:
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if stage_name:
        log_dir += f"_{stage_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    return log_root_path, log_dir


def _run_one_stage(
    *,
    args_cli,
    env_cfg,
    agent_cfg,
    stage_name: str,
    stage_cfg: dict,
    teacher_checkpoint: str | None,
) -> tuple[str, str | None]:
    """Run one staged training segment and return (log_dir, latest_checkpoint)."""

    stage_env_cfg = deepcopy(env_cfg)
    stage_agent_cfg = deepcopy(agent_cfg)

    if stage_cfg.get("max_iterations") is not None:
        stage_agent_cfg.max_iterations = int(stage_cfg["max_iterations"])

    if stage_cfg.get("run_name_suffix"):
        base_name = stage_agent_cfg.run_name or ""
        stage_agent_cfg.run_name = (base_name + stage_cfg["run_name_suffix"]).strip("_")

    stage_files = stage_cfg.get("motion_files")
    stage_selector = stage_cfg.get("motion_selector", args_cli.motion_selector)
    _apply_motion_settings(stage_env_cfg, args_cli, stage_files, stage_selector)

    stage_env_cfg.seed = stage_agent_cfg.seed
    stage_env_cfg.sim.device = args_cli.device if args_cli.device is not None else stage_env_cfg.sim.device

    log_root_path, log_dir = _build_log_dir(stage_agent_cfg, stage_name=stage_name)

    env = gym.make(args_cli.task, cfg=stage_env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO][{stage_name}] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    stage_context = {
        "stage_name": stage_name,
        "enable_amp": bool(stage_cfg.get("enable_amp", False)),
        "enable_distill": bool(stage_cfg.get("enable_distill", False)),
        "teacher_checkpoint": teacher_checkpoint,
        "motion_selector": stage_selector,
        "num_motion_files": len(stage_files) if stage_files else 0,
    }

    runner = OnPolicyRunner(
        env,
        stage_agent_cfg.to_dict(),
        log_dir=log_dir,
        device=stage_agent_cfg.device,
        stage_context=stage_context,
    )
    runner.add_git_repo_to_log(__file__)
    runner.emit_stage_summary()

    resume_checkpoint = stage_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        print(f"[INFO][{stage_name}] Loading configured checkpoint: {resume_checkpoint}")
        runner.load(resume_checkpoint)
    elif teacher_checkpoint:
        print(f"[INFO][{stage_name}] Loading previous-stage checkpoint: {teacher_checkpoint}")
        runner.load(teacher_checkpoint)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), stage_env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), stage_agent_cfg)

    # Keep amp/distill metadata in params for reproducibility.
    dump_yaml(os.path.join(log_dir, "params", "stage.yaml"), _to_plain_dict(stage_context))

    runner.learn(num_learning_iterations=stage_agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()

    latest_ckpt = _latest_checkpoint(log_dir)
    if latest_ckpt is None:
        print(f"[WARN][{stage_name}] No checkpoint file found in {log_dir}")
    else:
        print(f"[INFO][{stage_name}] Latest checkpoint: {latest_ckpt}")

    return log_dir, latest_ckpt


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    _apply_motion_settings(env_cfg, args_cli, motion_files=None, motion_selector=None)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load the motion file from the wandb registry (only when --registry_name is provided)
    registry_name = args_cli.registry_name
    if registry_name is not None:
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # staged mode: stage1 (AMP) -> stage2 (distill)
    staged_cfg = _as_dict(getattr(agent_cfg, "staged_training", {}))
    if staged_cfg.get("enabled", False):
        stage1_cfg = _as_dict(staged_cfg.get("stage1", {}))
        stage2_cfg = _as_dict(staged_cfg.get("stage2", {}))

        print("[STAGED] Running two-stage training pipeline")
        stage1_dir, stage1_ckpt = _run_one_stage(
            args_cli=args_cli,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            stage_name="stage1",
            stage_cfg=stage1_cfg,
            teacher_checkpoint=None,
        )

        if stage1_ckpt is None:
            raise RuntimeError(
                f"Stage1 did not produce a checkpoint under {stage1_dir}. "
                "Cannot start stage2 distillation."
            )

        teacher_strategy = stage2_cfg.get("teacher_source", "stage1")
        teacher_checkpoint = stage1_ckpt if teacher_strategy == "stage1" else stage2_cfg.get("teacher_checkpoint")
        _run_one_stage(
            args_cli=args_cli,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            stage_name="stage2",
            stage_cfg=stage2_cfg,
            teacher_checkpoint=teacher_checkpoint,
        )

        return

    log_root_path, log_dir = _build_log_dir(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        stage_context={"stage_name": "single", "enable_amp": False, "enable_distill": False},
    )
    runner.emit_stage_summary()
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
