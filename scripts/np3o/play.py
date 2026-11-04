# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a checkpoint trained with NP3O."""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Play a checkpoint trained with NP3O.')
parser.add_argument('--task', type=str, required=True, help='Gym task ID.')
parser.add_argument('--num_envs', type=int, default=50, help='Number of envs for playback.')
parser.add_argument('--checkpoint', type=str, default=None, help='Absolute checkpoint path (overrides auto-resolve).')
parser.add_argument('--load_run', type=str, default='.*', help='Run dir regex when --checkpoint is omitted.')
parser.add_argument('--load_checkpoint', type=str, default=r'model_.*\.pt', help='Checkpoint filename regex.')
parser.add_argument(
    '--export_policy', action='store_true',
    help='Export the loaded policy as TorchScript + ONNX next to the checkpoint and exit.',
)
parser.add_argument(
    '--export_dir', type=str, default=None,
    help='Override export directory (defaults to <checkpoint_dir>/exported).',
)
parser.add_argument('--keyboard', action='store_true', help='Window keyboard: W/S vx, A/D yaw, Q/E lower/raise height, Space stop, R reset.')
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.keyboard and args_cli.headless:
    parser.error('--keyboard requires a window; remove --headless.')

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ddt_lab.tasks  # noqa: F401  -- registers tasks
import gymnasium as gym
import torch
from isaaclab_tasks.utils import get_checkpoint_path

from ddt_lab.algorithms.np3o import IsaacLabNP3OWrapper, OnConstraintPolicyRunner
from keyboard_controller import KeyboardController


def _resolve_runner_cfg(entry_point: str) -> dict:
    module_name, attr = entry_point.split(':')
    obj = getattr(importlib.import_module(module_name), attr)
    cfg = obj() if callable(obj) else obj
    if not isinstance(cfg, dict):
        raise TypeError(f"NP3O cfg entry point '{entry_point}' must return a dict")
    return cfg


def main():
    spec = gym.spec(args_cli.task)
    runner_cfg = _resolve_runner_cfg(spec.kwargs['np3o_cfg_entry_point'])
    env_cfg_entry = spec.kwargs['env_cfg_entry_point']
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        # 键盘模式从默认高度启动；保持训练范围，后续由控制器接管目标。
        height_cfg = getattr(env_cfg.commands, 'base_height', None)
        if height_cfg is not None:
            height_cfg.resampling_time_range = (1.0e9, 1.0e9)  # 避免键盘输入被定时重采样覆盖

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabNP3OWrapper(env, device=args_cli.device or 'cuda:0')

    runner = OnConstraintPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device or 'cuda:0')

    if args_cli.checkpoint is not None:
        ckpt = args_cli.checkpoint
    else:
        log_root = os.path.abspath(os.path.join('logs', 'np3o', runner_cfg['runner']['experiment_name']))
        ckpt = get_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)
    print(f'[INFO] loading checkpoint: {ckpt}')
    runner.load(ckpt, load_optimizer=False)

    # Always export the JIT/ONNX policy next to the checkpoint (matches the
    # pre-NP3O scripts/rsl_rl/play.py behavior). Skip on --export_policy off
    # would be a one-line guard; we keep it on by default since it's cheap.
    export_dir = args_cli.export_dir or os.path.join(os.path.dirname(ckpt), 'exported')
    runner.alg.actor_critic.save_torch_jit_policy(export_dir, args_cli.device or 'cuda:0')
    if args_cli.export_policy:
        # --export_policy explicitly requested: skip rollout, just export.
        env.env.close()
        return

    policy = runner.get_inference_policy(args_cli.device or 'cuda:0')
    obs = env.get_observations()
    controller = None
    try:
        if args_cli.keyboard:
            # Configure limits, increments and timeout through constructor arguments.
            controller = KeyboardController(env)
            print('[INFO] Focus the Isaac Sim window: W/S vx, A/D yaw, Space stop, Q/E lower/raise height (0.01 m), Space stop (keep height), R reset (height 0.35 m).')
        while simulation_app.is_running():
            with torch.inference_mode():
                if controller is not None:
                    obs = controller.apply(obs)
                actions = policy(obs)
                obs, _, _, _, dones, _ = env.step(actions)
                if controller is not None:
                    obs = controller.apply(obs, advance_history=True, reset_mask=dones)
    finally:
        if controller is not None:
            controller.close()
        env.env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
