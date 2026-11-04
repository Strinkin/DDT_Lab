"""项目自定义奖励函数；迁移目录后保留原有计算逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .observations import base_height_relative

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核奖励对 x 轴线速度指令的跟踪。"""
    # 获取机器人对象并标注类型。
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算 x 轴线速度跟踪误差的平方。
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0]
        - asset.data.root_lin_vel_b[:, 0]
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def auxiliary_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回当前帧接触力模长大于 1 N 的辅助轮比例。"""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > 1.0).float().mean(dim=1)


# 旧高度平方惩罚实现，保留供对照；当前不启用。
# def track_base_height_l2(
#     env: ManagerBasedRLEnv,  # 当前批量仿真环境
#     command_name: str,  # 提供目标高度的一维指令名称
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要跟踪高度的机器人
# ) -> torch.Tensor:
#     """计算动态目标高度的平方误差，返回 (num_envs,)；配合负权重使用。
#
#     实际高度复用 Critic 的环境原点相对高度定义，仅适用于平地高度跟踪。
#     """
#     # 读取未经过观测缩放的目标高度，和实际高度统一使用米。
#     target_height = env.command_manager.get_command(command_name)[:, 0]
#     actual_height = base_height_relative(env, asset_cfg)[:, 0]
#     penalty = torch.square(actual_height - target_height)
#     # 沿用旧高度奖励的直立程度系数：严重倾斜时减弱高度惩罚。
#     asset: RigidObject = env.scene[asset_cfg.name]
#     upright_factor = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return penalty * upright_factor
#
#
def base_height_error(
    env: ManagerBasedRLEnv,  # 当前批量仿真环境
    command_name: str,  # 目标高度指令名称
    tolerance: float = 0.01,  # 对称容差带，单位 m
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要测量高度的机器人
) -> torch.Tensor:
    """返回超出高度容差带的绝对误差，形状为 (num_envs,)。"""
    if tolerance < 0:
        raise ValueError("tolerance 必须为非负数")
    target = env.command_manager.get_command(command_name)[:, 0]
    actual = base_height_relative(env, asset_cfg)[:, 0]
    # 容差带内误差为零，带外按超出部分线性增长。
    return (torch.abs(actual - target) - tolerance).clamp_min(0.0)


def tracking_base_height_exp(
    env: ManagerBasedRLEnv,  # 当前批量仿真环境
    command_name: str,  # 目标高度指令名称
    sigma: float = 0.07**2,  # 指数分母，单位 m²
    tolerance: float = 0.01,  # 对称容差带，单位 m
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要测量高度的机器人
) -> torch.Tensor:
    """容差带内返回 1，带外按平方误差指数衰减；权重由奖励管理器应用。"""
    if sigma <= 0:
        raise ValueError("sigma 必须大于零")
    error = base_height_error(env, command_name, tolerance, asset_cfg)
    return torch.exp(-error.square() / sigma)
