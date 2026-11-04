"""项目自定义 MDP 组件。"""

from .height_command import UniformHeightCommand, UniformHeightCommandCfg
from .observations import base_height_relative
from .rewards import (
    auxiliary_contact, base_height_error,  # track_base_height_l2 已停用
    track_lin_vel_x_exp, tracking_base_height_exp,
)

__all__ = [
    "UniformHeightCommand",
    "UniformHeightCommandCfg",
    "base_height_relative",
    "auxiliary_contact",
    "track_lin_vel_x_exp",
    # "track_base_height_l2",  # 旧平方惩罚已停用
    "base_height_error",
    "tracking_base_height_exp",
]
