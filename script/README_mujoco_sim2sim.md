# MuJoCo sim2sim：高度与键盘控制

脚本位置为 `script/mujoco_sim2sim.py`。当前远端版本以本机已修正版本为基础，保留机身速度坐标转换、持续可视化窗口和 R 完整重置。

## 运行

```bash
conda activate isaaclab
cd /home/sjq/ddt_lab
python script/mujoco_sim2sim.py \
  --mjcf /path/to/robot_or_scene.xml \
  --checkpoint /path/to/model.pt \
  --task Height-Flat-Andymini \
  --height 0.35
```

本机运行时，将项目路径替换为 `/home/strinkin/ddt_lab`。默认打开窗口，直到手动关闭。无窗口测试须指定 `--headless --duration 30`。`--duration` 也可用于限制窗口运行时间。

脚本从 checkpoint 同级 `params/env.yaml`、`params/agent.yaml` 加载训练配置，读取当前项目的网络实现；不会用当前任务配置悄悄替换旧模型训练参数。旧的速度模型没有高度指令，省略 `--height` 后仍可加载；不能向旧模型强行增加高度输入。

## 键盘

点击 MuJoCo 窗口使其获得焦点：

- W/S：前进速度增加/减少 0.1 m/s。
- A/D：转向角速度增加/减少 0.1 rad/s。
- Q/E：目标高度降低/升高 0.01 m。
- 空格：速度清零，保留高度。
- R：恢复物理初始状态、控制目标、零速度指令、默认高度和观测历史。

高度默认是训练范围内最接近 0.35 m 的值；运动和高度指令均受保存的训练范围限制。当前新训练配置的高度范围为 `(0.35, 0.35)`，因此 Q/E 暂时不会产生高度变化。必须使用具有相应训练范围的策略才能评估动态升降能力。

MuJoCo viewer 回调只有按键事件，运动输入超时按最后一次按键/重复事件计算：首次键盘运动输入后，若 1 秒没有新的运动事件则停车，Q/E 不刷新速度超时。启动参数 `--command VX VY WZ` 在首次运动键输入前保持固定。

指令在策略观测中按 YAML 配置缩放和拼接。键盘只刷新最新帧的指令，不把新指令写入过去的历史帧。

## 模型要求与当前限制

MJCF 必须包含一个自由基座、完整机身惯量，以及动作关节对应的单位 gain/gear 力矩 motor。无地面时在同目录寻找 `<模型名>_scene.xml`。

2026-10-27 检查发现，项目当前 `andymini.xml` 是重新生成的固定基座版本，缺少原自由基座、base_link 机身惯量和 motor 定义。使用这份文件会在模型检查阶段报错。脚本适配不会自动猜测并补齐这些物理结构；该项目 XML 尚未在本轮修改。

当前脚本保留初版执行器近似：ImplicitActuator 使用显式 PD 力矩，尚未等效实现驱动轮物理速度限制。改变训练执行器限值后，也应检查 MJCF 的 actuatorfrcrange 和 ctrlrange。

## 本轮验证

环境：远端 isaaclab，MuJoCo 3.3.7。

使用 `/tmp/ddt_mujoco_check_assets` 下临时补齐自由基座、机身惯量和电机的测试模型，未修改项目 MJCF：

- 旧 `model_20000.pt`：27 维单帧观测，历史长度 10，成功加载及步进。
- 新高度模型 `2026-10-27_14-04-06/model_1300.pt`：28 维单帧观测，历史长度 10，成功加载及步进。
- Q/E 高度增量、上下限、观测最新帧和历史保护、W/A 速度、停车/超时保留高度、R 恢复：通过程序化事件测试。

这些检查证明脚本控制与推理链路可运行，不代表项目当前 MJCF 可直接运行，也不是新训练策略升降效果验收。

验证日志：`/tmp/ddt_mujoco_height_check.log`。
