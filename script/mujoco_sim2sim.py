#!/usr/bin/env python3
"""MuJoCo sim2sim for project NP3O checkpoints and their saved params/*.yaml."""
import argparse
import ast
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import re
import sys
import time
from queue import SimpleQueue
import types
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/ddt_lab/ddt_lab'


def load_config(path):
    # Read Python-tagged Isaac Lab YAML without executing YAML constructors.
    def convert(x):
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        if x in ('null', '~', ''):
            return None
        if x in ('true', 'false'):
            return x == 'true'
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x
    return convert(yaml.load(path.read_text(), Loader=yaml.BaseLoader))


def check_task(task, agent):
    for path in (SOURCE / 'tasks').rglob('__init__.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            args = {k.arg: k.value for k in node.keywords}
            if not isinstance(args.get('id'), ast.Constant) or args['id'].value != task:
                continue
            kw = args.get('kwargs')
            fields = {k.value: v for k, v in zip(kw.keys, kw.values)}
            entry = fields.get('np3o_cfg_entry_point')
            if isinstance(entry, ast.JoinedStr):
                suffix = entry.values[-1].value
            elif isinstance(entry, ast.Constant):
                suffix = entry.value
            else:
                raise ValueError('Task must register a supported NP3O entry point')
            module, func = suffix.split(':')
            cfg_path = path.parent / 'agents' / (module.rsplit('.', 1)[-1] + '.py')
            for definition in ast.parse(cfg_path.read_text()).body:
                if isinstance(definition, ast.FunctionDef) and definition.name == func:
                    for stmt in ast.walk(definition):
                        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                            for target in stmt.targets:
                                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == 'experiment_name':
                                    if stmt.value.value != agent['runner']['experiment_name']:
                                        raise ValueError('Task/checkpoint experiment_name mismatch')
                                    print(f'[INFO] Task: {task}; registration: {path.relative_to(ROOT)}', flush=True)
                                    return
            raise ValueError(f'Cannot verify experiment_name in {cfg_path}')
    raise ValueError(f'Unregistered task: {task}')


def select(patterns, names, preserve=True):
    if patterns is None:
        raise ValueError('Initial version requires configured joint_names')
    patterns = [patterns] if isinstance(patterns, str) else patterns
    groups = [[i for i, n in enumerate(names) if re.fullmatch(p, n)] for p in patterns]
    if any(not g for g in groups):
        raise ValueError(f'Unmatched joint pattern: {patterns}; available: {names}')
    indices = [i for group in groups for i in group]
    if len(set(indices)) != len(indices):
        raise ValueError(f'Overlapping joint patterns: {patterns}')
    if not preserve and len(indices) > 1:
        raise ValueError('Cannot infer Isaac joint order with preserve_order=False')
    return np.array(indices, dtype=int)


def parameter(value, name, default=0.0):
    if value is None:
        return default
    if isinstance(value, dict):
        hits = [v for p, v in value.items() if re.fullmatch(p, name)]
        if len(hits) > 1:
            raise ValueError(f'Overlapping parameter patterns for {name}')
        return hits[0] if hits else default
    return value


def load_actor(agent, checkpoint, nprop, naction, history_len):
    # Bypass package __init__, which imports Isaac Sim; reuse pure torch modules.
    package = types.ModuleType('_sim2sim_np3o')
    package.__path__ = [str(SOURCE / 'algorithms/np3o')]
    sys.modules[package.__name__] = package
    module = importlib.import_module(package.__name__ + '.actor_critic')
    if agent['runner']['policy_class_name'] != 'ActorCriticBarlowTwins':
        raise ValueError('Only ActorCriticBarlowTwins is supported in this version')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    cfg = agent['policy']
    actor = module.MlpBarlowTwinsActor(nprop, cfg['bt_window'], naction,
        cfg['bt_mlp_encoder_dims'], cfg['bt_latent_encoder_dims'],
        cfg['actor_hidden_dims'], cfg['bt_projector_dims'], module.get_activation(cfg['activation']))
    prefix = 'actor_teacher_backbone.'
    actor.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    actor.eval()
    window = cfg['bt_window']
    if history_len < window + 1:
        raise ValueError('Insufficient history length')
    def infer(history):
        with torch.inference_mode():
            obs = torch.from_numpy(history.astype(np.float32)).unsqueeze(0)
            return actor(obs[:, -1], obs[:, -window-1:-1]).squeeze(0).numpy()
    return infer


class Simulation:
    def __init__(self, args, cfg):
        spec = mujoco.MjSpec.from_file(str(args.mjcf))
        if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
            scene = args.mjcf.with_name(args.mjcf.stem + '_scene.xml')
            if not scene.is_file():
                raise ValueError(f'MJCF has no ground. Supply a scene with --mjcf or create {scene}')
            includes = ET.parse(scene).getroot().findall('include')
            if not any((scene.parent / node.attrib['file']).resolve() == args.mjcf for node in includes):
                raise ValueError(f'{scene} does not include the requested MJCF')
            spec = mujoco.MjSpec.from_file(str(scene))
            if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
                raise ValueError(f'{scene} has no plane ground')
            print(f'[INFO] Ground scene: {scene}; original MJCF unchanged.', flush=True)
        self.model = m = spec.compile()
        m.opt.timestep = float(cfg['sim']['dt'])
        m.opt.gravity[:] = cfg['sim']['gravity']
        self.decimation = int(cfg['decimation'])
        if m.opt.timestep <= 0 or self.decimation < 1:
            raise ValueError('Invalid training control period')
        self.data = d = mujoco.MjData(m)
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError('Exactly one free root joint is required')
        self.base = int(m.jnt_bodyid[free[0]])
        self.root_q = int(m.jnt_qposadr[free[0]])
        ids = [i for i in range(m.njnt) if i != free[0]]
        if any(int(m.jnt_type[i]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) for i in ids):
            raise ValueError('Only hinge/slide robot joints are supported')
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in ids]
        self.qids = m.jnt_qposadr[ids]
        self.vids = m.jnt_dofadr[ids]
        robot = cfg['scene']['robot']
        init = robot['init_state']
        self.q0 = np.array([parameter(init['joint_pos'], n) for n in self.names])
        self.v0 = np.array([parameter(init['joint_vel'], n) for n in self.names])
        d.qpos[self.root_q:self.root_q+3] = init['pos']
        d.qpos[self.root_q+3:self.root_q+7] = init['rot']
        d.qpos[self.qids] = self.q0
        d.qvel[self.vids] = self.v0
        if np.any(init['lin_vel']) or np.any(init['ang_vel']):
            raise ValueError('Initial version requires zero base initial velocity')
        self.motors = {}
        for aid in range(m.nu):
            jid = int(m.actuator_trnid[aid, 0])
            if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT or jid not in ids:
                raise ValueError('Only direct joint motors are supported')
            if (m.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED or
                m.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE or
                m.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE or
                not np.isclose(m.actuator_gainprm[aid, 0], 1) or
                not np.isclose(m.actuator_gear[aid, 0], 1)):
                raise ValueError('MJCF must use unit-gain, unit-gear torque motors')
            ix = ids.index(jid)
            if ix in self.motors:
                raise ValueError('Multiple motors on one joint are unsupported')
            self.motors[ix] = aid
        self.actions = []
        for term in cfg['actions'].values():
            if term is None:
                continue
            kind = term['class_type'].split(':')[-1]
            if kind not in ('JointPositionAction', 'JointVelocityAction') or term['asset_name'] != 'robot':
                raise ValueError(f'Unsupported action: {kind}')
            for ix in select(term['joint_names'], self.names, term.get('preserve_order', False)):
                if ix not in self.motors:
                    raise ValueError(f'Missing torque motor for {self.names[ix]}')
                name = self.names[ix]
                offset = parameter(term.get('offset'), name)
                if term.get('use_default_offset'):
                    offset = self.q0[ix] if kind == 'JointPositionAction' else self.v0[ix]
                self.actions.append((ix, kind, parameter(term['scale'], name, 1), offset,
                                     parameter(term.get('clip'), name, [-np.inf, np.inf])))
        if len({a[0] for a in self.actions}) != len(self.actions):
            raise ValueError('Multiple action terms control the same joint')
        self.last_action = np.zeros(len(self.actions))
        self.target_q, self.target_v = self.q0.copy(), self.v0.copy()
        self.actuators = {}
        for actuator in robot['actuators'].values():
            kind = actuator['class_type'].split(':')[-1]
            if kind not in ('DCMotor', 'IdealPDActuator', 'ImplicitActuator'):
                raise ValueError(f'Unsupported actuator model: {kind}')
            for ix in select(actuator['joint_names_expr'], self.names):
                if ix in self.actuators:
                    raise ValueError('Overlapping actuator groups')
                name = self.names[ix]
                kp = parameter(actuator.get('stiffness'), name)
                kd = parameter(actuator.get('damping'), name)
                limit = parameter(actuator.get('effort_limit_sim') if kind == 'ImplicitActuator' else actuator.get('effort_limit'), name, np.inf)
                vel = parameter(actuator.get('velocity_limit'), name, np.inf)
                stall = parameter(actuator.get('saturation_effort'), name, limit)
                self.actuators[ix] = (kind, kp, kd, limit, vel, stall)
                m.dof_damping[self.vids[ix]] = 0  # PD damping is applied below.
                m.dof_frictionloss[self.vids[ix]] = parameter(actuator.get('friction'), name)
                if actuator.get('armature') is not None:
                    m.dof_armature[self.vids[ix]] = parameter(actuator['armature'], name)
        if set(self.actuators) != set(range(len(ids))):
            raise ValueError('Every MJCF joint needs a training actuator definition')
        self.initial_qpos = d.qpos.copy()
        self.initial_qvel = d.qvel.copy()
        self.command = np.array(args.command, dtype=float)
        # 高度及其允许范围来自 checkpoint 保存的配置，不猜测旧策略的输入。
        height_cfg = cfg.get('commands', {}).get('base_height')
        self.height_range = None
        self.default_height = self.height = None
        requested_height = getattr(args, 'height', None)
        if height_cfg is not None:
            lower, upper = map(float, height_cfg['height_range'])
            if not np.isfinite([lower, upper]).all() or not 0 <= lower <= upper:
                raise ValueError('Invalid saved height_range')
            self.height_range = (lower, upper)
            self.default_height = float(np.clip(0.35, lower, upper))
            self.height = self.default_height if requested_height is None else requested_height
            if not np.isfinite(self.height) or not lower <= self.height <= upper:
                raise ValueError(f'--height must be within training range {self.height_range}')
        elif requested_height is not None:
            raise ValueError('This checkpoint has no base_height command; --height is unsupported')
        group = cfg['observations']['policy']
        self.terms = [(k, v) for k, v in group.items() if isinstance(v, dict) and 'func' in v]
        if not group.get('concatenate_terms') or group.get('flatten_history_dim'):
            raise ValueError('NP3O requires concatenated, unflattened history')
        self.history_len = int(group['history_length'])
        mujoco.mj_forward(m, d)
        obs = self.observe()
        self.history = np.repeat(obs[None], self.history_len, axis=0)
        self.command_observations = []
        offset = 0
        widths = {'base_ang_vel': 3, 'base_lin_vel': 3, 'projected_gravity': 3,
                  'last_action': len(self.actions)}
        for _, term in self.terms:
            func = term['func'].split(':')[-1]
            params = term.get('params', {})
            if func == 'generated_commands':
                width = 1 if params['command_name'] == 'base_height' else 3
                self.command_observations.append((slice(offset, offset + width), term))
            elif func in ('joint_pos_rel', 'joint_pos_rel_without_wheel', 'joint_vel_rel'):
                asset = params.get('asset_cfg', {})
                width = len(select(asset.get('joint_names'), self.names, asset.get('preserve_order', False)))
            else:
                width = widths[func]
            offset += width
        print(f'[INFO] Height command: {self.height}; training range: {self.height_range}', flush=True)
        print(f'[INFO] MJCF joints: {self.names}', flush=True)
        print(f'[INFO] Action order: {[self.names[a[0]] for a in self.actions]}', flush=True)
        print(f'[INFO] Observation order: {[k for k, _ in self.terms]}', flush=True)
        print(f'[INFO] Observation/history/action: {len(obs)}/{self.history_len}/{len(self.actions)}; control dt={m.opt.timestep*self.decimation:g}s', flush=True)

    def reset(self):
        """Restore physics, controller targets and policy observation history."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.last_action.fill(0)
        self.target_q[:] = self.q0
        self.target_v[:] = self.v0
        mujoco.mj_forward(self.model, self.data)
        self.history[:] = self.observe()[None, :]

    def observe(self):
        d, m = self.data, self.model
        rotation = d.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY local coordinates follow the inertial frame, which can differ
        # from the root/body axes expected by the policy. Rotate world velocity.
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        parts = []
        for name, term in self.terms:
            func = term['func'].split(':')[-1]
            params = term.get('params', {})
            if term.get('modifiers') or term.get('history_length', 0):
                raise ValueError(f'Unsupported modifiers/per-term history: {name}')
            if func == 'base_ang_vel':
                value = rotation.T @ velocity[:3]
            elif func == 'base_lin_vel':
                value = rotation.T @ velocity[3:]
            elif func == 'projected_gravity':
                value = rotation.T @ (m.opt.gravity / np.linalg.norm(m.opt.gravity))
            elif func == 'generated_commands':
                command_name = params.get('command_name')
                if command_name == 'base_velocity':
                    value = self.command.copy()
                elif command_name == 'base_height' and self.height is not None:
                    value = np.array([self.height])  # 单位 m，之后按观测配置缩放
                else:
                    raise ValueError(f'Unsupported command: {params}')
            elif func in ('joint_pos_rel', 'joint_pos_rel_without_wheel', 'joint_vel_rel'):
                asset = params.get('asset_cfg', {})
                if asset.get('name', 'robot') != 'robot':
                    raise ValueError('Only robot observations are supported')
                indices = select(asset.get('joint_names'), self.names, asset.get('preserve_order', False))
                value = (d.qvel[self.vids] - self.v0) if func == 'joint_vel_rel' else (d.qpos[self.qids] - self.q0)
                if func == 'joint_pos_rel_without_wheel':
                    value[select(params['wheel_asset_cfg']['joint_names'], self.names)] = 0
                value = value[indices]
            elif func == 'last_action':
                value = self.last_action.copy()
            else:
                raise ValueError(f'Unsupported observation: {term["func"]}; an adapter is required')
            if term.get('clip') is not None:
                value = np.clip(value, *term['clip'])
            if term.get('scale') is not None:
                value = value * np.array(term['scale'])
            parts.append(value)
        return np.concatenate(parts).astype(np.float32)

    def refresh_command_observations(self):
        """只刷新最新帧的指令，避免键盘事件造成额外的历史推进。"""
        for target_slice, term in self.command_observations:
            name = term['params']['command_name']
            value = self.command.copy() if name == 'base_velocity' else np.array([self.height])
            if term.get('clip') is not None:
                value = np.clip(value, *term['clip'])
            if term.get('scale') is not None:
                value = value * np.array(term['scale'])
            self.history[-1, target_slice] = value

    def set_action(self, action):
        if action.shape != self.last_action.shape or not np.isfinite(action).all():
            raise ValueError(f'Invalid policy output: {action}')
        self.last_action[:] = action
        for raw, (ix, kind, scale, offset, clip) in zip(action, self.actions):
            value = np.clip(raw * scale + offset, *clip)
            if kind == 'JointPositionAction':
                self.target_q[ix] = value
            else:
                self.target_v[ix] = value

    def step(self):
        m, d = self.model, self.data
        d.ctrl[:] = 0
        d.qfrc_applied[:] = 0
        for ix, (kind, kp, kd, limit, vmax, stall) in self.actuators.items():
            q, v = d.qpos[self.qids[ix]], d.qvel[self.vids[ix]]
            effort = kp * (self.target_q[ix] - q) + kd * (self.target_v[ix] - v)
            lo, hi = -limit, limit
            if kind == 'DCMotor':
                speed = np.clip(v, -vmax*(1+limit/stall), vmax*(1+limit/stall))
                lo, hi = max(-limit, stall*(-1-speed/vmax)), min(limit, stall*(1-speed/vmax))
            effort = np.clip(effort, lo, hi)
            if ix in self.motors:
                aid = self.motors[ix]
                if m.actuator_ctrllimited[aid]:
                    effort = np.clip(effort, *m.actuator_ctrlrange[aid])
                d.ctrl[aid] = effort
            else:
                d.qfrc_applied[self.vids[ix]] = effort
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError(f'Non-finite simulation state at t={d.time}')
        if any(w.number for w in d.warning):
            raise RuntimeError(f'MuJoCo simulation warning at t={d.time}')


class KeyboardController:
    """MuJoCo 按键指令；回调仅排队，由仿真线程更新指令和物理状态。"""

    def __init__(self, sim, cfg):
        self.sim = sim
        ranges = cfg['commands']['base_velocity']['ranges']
        self.vx_range = ranges['lin_vel_x']
        self.wz_range = ranges['ang_vel_z']
        self.velocity_step = 0.1  # 线速度/转向角速度每次增量，分别为 m/s、rad/s
        self.height_step = 0.01  # 每次高度增量，单位 m
        self.input_timeout = 1.0  # 最后一次运动键事件后停车的时间，单位 s
        self.last_motion_input = None  # 固定 --command 在首次运动键输入前保持有效
        self.events = SimpleQueue()

    def on_key(self, keycode):
        if keycode in (32, ord('W'), ord('S'), ord('A'), ord('D'), ord('Q'), ord('E'), ord('R'),
                       ord('w'), ord('s'), ord('a'), ord('d'), ord('q'), ord('e'), ord('r')):
            self.events.put(keycode)

    def update(self):
        """处理已排队按键，返回是否需要重置整个仿真。"""
        reset_requested = False
        now = time.monotonic()
        while not self.events.empty():
            key = chr(self.events.get()).upper()
            if key in ('W', 'S'):
                self.sim.command[0] = np.clip(self.sim.command[0] + self.velocity_step * (1 if key == 'W' else -1), *self.vx_range)
                self.last_motion_input = now
            elif key in ('A', 'D'):
                self.sim.command[2] = np.clip(self.sim.command[2] + self.velocity_step * (1 if key == 'A' else -1), *self.wz_range)
                self.last_motion_input = now
            elif key in ('Q', 'E') and self.sim.height is not None:
                self.sim.height = float(np.clip(self.sim.height + self.height_step * (1 if key == 'E' else -1), *self.sim.height_range))
            elif key == ' ':
                self.sim.command.fill(0)  # 停车只清零速度，不改变高度
                self.last_motion_input = None
            elif key == 'R':
                self.sim.command.fill(0)
                self.sim.height = self.sim.default_height
                self.last_motion_input = None
                reset_requested = True
        # MuJoCo viewer 回调只有按键事件；以最后一次按键/重复事件计时。
        if self.last_motion_input is not None and now - self.last_motion_input >= self.input_timeout:
            self.sim.command.fill(0)
            self.last_motion_input = None
        return reset_requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--height', type=float, default=None, help='Initial target height in metres; default 0.35 clamped to saved training range')
    parser.add_argument('--duration', type=float, default=None, help='Optional simulation seconds; default: run until the viewer is closed')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--command', type=float, nargs=3, default=[0, 0, 0], metavar=('VX', 'VY', 'WZ'))
    args = parser.parse_args()
    if (args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0)) or not np.isfinite(args.command).all():
        parser.error('duration must be finite and positive; command must be finite')
    if args.headless and args.duration is None:
        parser.error('--headless requires --duration')
    if not args.headless and not os.environ.get('DISPLAY'):
        parser.error('A graphical display is required for the viewer; set DISPLAY or use --headless --duration SECONDS')
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.mjcf = args.mjcf.resolve(strict=True)
    params = args.checkpoint.parent / 'params'
    cfg, agent = load_config(params / 'env.yaml'), load_config(params / 'agent.yaml')
    check_task(args.task, agent)
    print(f'[INFO] Training configuration: {params}', flush=True)
    torch.set_num_threads(1)
    sim = Simulation(args, cfg)
    infer = load_actor(agent, args.checkpoint, sim.history.shape[-1], len(sim.actions), sim.history_len)
    headless = args.headless
    print(f'[INFO] Mode: {"headless" if headless else "viewer"}; command={args.command}; duration={args.duration if args.duration is not None else "unlimited"}', flush=True)
    keyboard = KeyboardController(sim, cfg)
    if not headless:
        print('[INFO] W/S forward/back, A/D turn, Q/E lower/raise height (0.01 m), Space stop (keep height), R reset.', flush=True)

    viewer = nullcontext(None)
    if not headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(sim.model, sim.data, key_callback=keyboard.on_key)
    steps = None if args.duration is None else int(np.ceil(args.duration / sim.model.opt.timestep))
    controls = completed = cycle_step = 0
    start = time.monotonic()
    min_height, max_action = float('inf'), 0.0
    with viewer as view:
        if view is not None:
            view.cam.lookat[:] = sim.data.xpos[sim.base]
            view.cam.distance = 2.5
        while steps is None or completed < steps:
            if view is not None and not view.is_running():
                break
            if keyboard.update():
                with view.lock():
                    sim.reset()
                cycle_step = 0
                start = time.monotonic() - completed * sim.model.opt.timestep
                view.cam.lookat[:] = sim.data.xpos[sim.base]
                view.sync()
                print('[INFO] R reset: initial state, zero velocity command, default height and policy history restored.', flush=True)
            step = cycle_step
            if step % sim.decimation == 0:
                if step:
                    sim.history[:-1] = sim.history[1:].copy()
                    sim.history[-1] = sim.observe()
                sim.refresh_command_observations()
                action = infer(sim.history)
                sim.set_action(action)
                controls += 1
                max_action = max(max_action, float(np.max(np.abs(action))))
            sim.step()
            completed += 1
            cycle_step += 1
            min_height = min(min_height, float(sim.data.qpos[sim.root_q+2]))
            if view is not None:
                if step % sim.decimation == 0:
                    view.cam.lookat[:] = sim.data.xpos[sim.base]
                    view.sync()
                time.sleep(max(0, start+(step+1)*sim.model.opt.timestep-time.monotonic()))
    print(f'[RESULT] Completed {completed}/{steps} physics steps, {controls} policy steps, simulated {sim.data.time:.3f}s; elapsed {time.monotonic()-start:.2f}s', flush=True)
    print(f'[RESULT] Base height: min={min_height:.4f}m, final={sim.data.qpos[sim.root_q+2]:.4f}m; max |action|={max_action:.4f}', flush=True)
    if steps is not None and completed == steps:
        print('[SUCCESS] Policy inference and MuJoCo simulation completed with finite states and no simulation warnings.', flush=True)
    else:
        print('[INFO] Viewer closed; simulation stopped.', flush=True)


if __name__ == '__main__':
    main()
