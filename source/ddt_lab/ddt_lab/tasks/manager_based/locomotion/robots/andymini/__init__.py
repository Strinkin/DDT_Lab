# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments. All Tita tasks are wired to NP3O (BarlowTwins-PPO);
# stock PPO is no longer supported because the policy obs is now 3D.
##

gym.register(
    id="Height-Flat-Andymini",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AndyminiFlatEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:andymini_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="Height-Flat-Andymini-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AndyminiFlatEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:andymini_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="Height-Rough-Andymini",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AndyminiRoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:andymini_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="Height-Rough-Andymini-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AndyminiRoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:andymini_rough_np3o_runner_cfg",
    },
)
