# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O training configs for andymini.

Built on top of [agents/np3o_cfg.py](../../agents/np3o_cfg.py) base; only
``experiment_name`` / ``max_iterations`` differ. Numbers mirror
``LocomotionWithNP3O/configs/andymini/andymini_flat_config.py``.
"""

from __future__ import annotations

from ddt_lab.tasks.manager_based.locomotion.agents.np3o_cfg import base_np3o_runner_cfg


def andymini_flat_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andymini_flat"
    cfg["runner"]["max_iterations"] = 3000
    return cfg


def andymini_rough_np3o_runner_cfg() -> dict:
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "andymini_rough"
    cfg["runner"]["max_iterations"] = 5000
    return cfg
