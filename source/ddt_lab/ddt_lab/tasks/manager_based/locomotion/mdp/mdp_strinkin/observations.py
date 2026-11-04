"""项目自定义观测函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_relative(
    env: ManagerBasedRLEnv,  # 当前批量仿真环境
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要测量高度的机器人
) -> torch.Tensor:
    """返回机身根原点相对环境原点的高度，形状为 (num_envs, 1)，单位为米。

    平地上可作为离地高度；粗糙地形下不代表机身相对局部地面的高度。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 保留最后一维，便于和 Critic 的其他观测按最后一维拼接。
    return asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
