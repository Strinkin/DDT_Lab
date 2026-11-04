"""一维目标机身离地高度指令；这里只生成目标，不执行高度控制。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .observations import base_height_relative

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformHeightCommand(CommandTerm):
    """为每个环境独立生成目标高度，输出形状为 (num_envs, 1)，单位为米。"""

    cfg: UniformHeightCommandCfg  # 当前高度指令使用的配置

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        lower, upper = cfg.height_range
        if not all(math.isfinite(v) for v in (lower, upper)) or not 0 <= lower <= upper:
            raise ValueError("height_range 必须是有限、非负且按下限到上限排列的范围。")
        t_min, t_max = cfg.resampling_time_range
        if not all(math.isfinite(v) for v in (t_min, t_max)) or not 0 < t_min <= t_max:
            raise ValueError("resampling_time_range 必须是有限、正数且按下限到上限排列的范围。")
        # 在首次环境重置前也提供有效值；之后由基类管理重采样计时。
        self.height_command = torch.full((self.num_envs, 1), float(lower), device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """目标机身离地高度；地面测量与高度奖励由后续独立实现。"""
        return self.height_command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # 显式展开全环境索引，保证基类重采样能够使用 len(env_ids)。
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # 只更新指定环境；先生成独立随机样本再赋值，避免高级索引副本不回写。
        lower, upper = self.cfg.height_range
        samples = torch.empty(len(env_ids), device=self.device).uniform_(lower, upper)
        self.height_command[env_ids, 0] = samples

    def _update_command(self):
        # 初版采用分段常值目标，在两次重采样之间保持高度不变。
        pass

    def _update_metrics(self):
        # 与速度误差指标一致，按最大指令周期对应的步数归一化后累计。
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        actual_height = base_height_relative(self._env)[:, 0]
        self.metrics["error_height"] += torch.abs(self.height_command[:, 0] - actual_height) / max_command_step


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """高度指令配置，复用基类的重采样周期和指令管理接口。"""

    class_type: type = UniformHeightCommand  # 对应的高度指令生成类
    # 初版保持原来的 0.35 m；确认可达范围后再开放随机高度。
    height_range: tuple[float, float] = (0.35, 0.35)  # 目标高度范围（下限、上限），单位 m
