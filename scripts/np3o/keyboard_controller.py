"""窗口键盘控制：W/S 前后、A/D 转向、Q/E 降低/升高、空格停车、R 重置。"""
import math
import time
import torch


class KeyboardController:
    def __init__(self, env, *, linear_velocity_min=-1.0, linear_velocity_max=1.0,
                 linear_velocity_step=0.1, angular_velocity_min=-1.0,
                 angular_velocity_max=1.0, angular_velocity_step=0.1,
                 height_step=0.01,
                 input_timeout=1.0):
        import carb.input
        import omni.appwindow
        self._env = env.unwrapped
        # 高度范围直接使用任务配置；没有高度指令的任务保留原速度控制。
        self._height_term = (
            self._env.command_manager.get_term('base_height')
            if 'base_height' in self._env.command_manager.active_terms else None
        )
        height_min, height_max = (
            self._height_term.cfg.height_range if self._height_term is not None else (0.35, 0.35)
        )
        for lower, upper, step in (
            (linear_velocity_min, linear_velocity_max, linear_velocity_step),
            (angular_velocity_min, angular_velocity_max, angular_velocity_step),
            (height_min, height_max, height_step),
        ):
            if not all(math.isfinite(v) for v in (lower, upper, step)) or lower > upper or step <= 0:
                raise ValueError('Limits must be finite and ordered; steps must be positive.')
        if not linear_velocity_min <= 0 <= linear_velocity_max or not angular_velocity_min <= 0 <= angular_velocity_max:
            raise ValueError('Velocity limits must include zero.')
        if not math.isfinite(input_timeout) or input_timeout <= 0:
            raise ValueError('input_timeout must be positive and finite.')
        self.linear_velocity_min = linear_velocity_min
        self.linear_velocity_max = linear_velocity_max
        self.linear_velocity_step = linear_velocity_step
        self.angular_velocity_min = angular_velocity_min
        self.angular_velocity_max = angular_velocity_max
        self.angular_velocity_step = angular_velocity_step
        # Q/E 不刷新速度输入超时；停车或速度超时后保留当前目标高度。
        self.height_min = height_min
        self.height_max = height_max
        self.height_step = height_step
        self.default_height = min(max(0.35, height_min), height_max)  # 默认/重置高度限制在训练范围内
        self.height = self.default_height
        self.input_timeout = input_timeout
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self._pressed = set()
        self._last_input = time.monotonic()
        self._last_printed_command = (self.linear_velocity, self.angular_velocity, self.height)
        self._histories = {}
        self._wrapper = env
        self._reset_requested = False
        self._term = self._env.command_manager.get_term('base_velocity')
        if not hasattr(self._term, 'vel_command_b') or self._term.command.shape[-1] != 3:
            raise ValueError('Expected a [vx, vy, yaw_rate] velocity command.')
        manager = self._env.observation_manager
        names = manager.active_terms['policy']
        index = names.index('velocity_commands')
        dims = manager.group_obs_term_dim['policy']
        widths = [d if isinstance(d, int) else d[-1] for d in dims]
        if widths[index] != 3:
            raise ValueError('velocity_commands must have width 3.')
        start = sum(widths[:index])
        self._slice = slice(start, start + 3)
        scale = manager._group_obs_term_cfgs['policy'][index].scale
        self._scale = torch.as_tensor(1.0 if scale is None else scale, device=self._env.device)
        self._height_slice = None
        if self._height_term is not None:
            if self._height_term.command.shape[-1] != 1:
                raise ValueError('base_height must be a one-dimensional command.')
            index = names.index('height_commands')
            if widths[index] != 1:
                raise ValueError('height_commands must have width 1.')
            start = sum(widths[:index])
            self._height_slice = slice(start, start + 1)
            scale = manager._group_obs_term_cfgs['policy'][index].scale
            self._height_scale = torch.as_tensor(1.0 if scale is None else scale, device=self._env.device)
        self._event_type = carb.input.KeyboardEventType
        self._input = carb.input.acquire_input_interface()
        window = omni.appwindow.get_default_app_window()
        if window is None or window.get_keyboard() is None:
            raise RuntimeError('Keyboard control requires an Isaac Sim window.')
        self._keyboard = window.get_keyboard()
        # Run before ordinary device subscribers and consume our control keys.
        self._subscription = self._input.subscribe_to_input_events(
            self._on_input_event, device=self._keyboard, order=0,
        )

    def _on_input_event(self, event):
        return self._on_event(event.event)

    def _on_event(self, event):
        raw_key = getattr(event.input, 'name', event.input)
        if not isinstance(raw_key, str):
            return True
        key = raw_key.upper()
        if key in {' ', 'SPACEBAR'}:
            key = 'SPACE'
        if key not in {'W', 'S', 'A', 'D', 'Q', 'E', 'SPACE', 'R'}:
            return True
        if event.type == self._event_type.KEY_RELEASE:
            self._pressed.discard(key)
            if key in {'W', 'S', 'A', 'D'}:
                self._last_input = time.monotonic()
        elif event.type == self._event_type.KEY_PRESS:
            if key in self._pressed:
                return False
            self._pressed.add(key)
            if key == 'R':
                self._reset_requested = True
            elif key == 'SPACE':
                self.linear_velocity = self.angular_velocity = 0.0
                self._last_input = time.monotonic()
            elif key in {'Q', 'E'} and self._height_term is not None:
                delta = self.height_step * (1 if key == 'E' else -1)  # E 升高，Q 降低
                self.height = min(self.height_max, max(self.height_min, self.height + delta))
            elif key in {'W', 'S', 'A', 'D'}:
                self._last_input = time.monotonic()
                if key in {'W', 'S'}:
                    delta = self.linear_velocity_step * (1 if key == 'W' else -1)
                    self.linear_velocity = min(self.linear_velocity_max, max(self.linear_velocity_min, self.linear_velocity + delta))
                else:
                    delta = self.angular_velocity_step * (1 if key == 'A' else -1)
                    self.angular_velocity = min(self.angular_velocity_max, max(self.angular_velocity_min, self.angular_velocity + delta))
        # False consumes press/repeat/release/character events for control keys.
        return False

    def apply(self, obs, *, advance_history=False, reset_mask=None):
        if self._reset_requested:
            # Reset outside the input callback, before computing the next action.
            self._reset_requested = False
            obs = self._wrapper.reset()
            self.linear_velocity = self.angular_velocity = 0.0
            self._last_input = time.monotonic()
            self._histories.clear()
            self.height = self.default_height  # R 恢复 0.35 m，或训练范围内最近的合法值
            reset_mask = None
        now = time.monotonic()
        if self._pressed & {'W', 'S', 'A', 'D'}:
            self._last_input = now
        if 'SPACE' in self._pressed or now - self._last_input >= self.input_timeout:
            self.linear_velocity = self.angular_velocity = 0.0
        current_command = (self.linear_velocity, self.angular_velocity, self.height)
        if current_command != self._last_printed_command:
            print(
                f'[Keyboard] vx={self.linear_velocity:.2f} m/s, '
                f'wz={self.angular_velocity:.2f} rad/s, '
                f'height={self.height:.2f} m',
                flush=True,
            )
            self._last_printed_command = current_command
        command = self._term.command
        command[:, 0] = self.linear_velocity
        command[:, 1] = 0.0
        command[:, 2] = self.angular_velocity
        self._term.is_heading_env[:] = False
        self._term.is_standing_env[:] = False
        # Prevent periodic resampling; reset commands are overridden by apply.
        self._term.time_left[:] = 1.0e9
        scaled = command.to(obs.device) * self._scale.to(obs.device)
        self._write_command_observation(obs, scaled, self._slice, 'velocity', advance_history, reset_mask)
        if self._height_term is not None:
            # 同时更新环境指令与 Actor 输入，避免奖励和策略收到不同高度。
            height_command = self._height_term.command
            height_command[:] = self.height
            self._height_term.time_left[:] = 1.0e9  # 键盘接管后不允许随机采样覆盖
            scaled_height = height_command.to(obs.device) * self._height_scale.to(obs.device)
            self._write_command_observation(obs, scaled_height, self._height_slice, 'height', advance_history, reset_mask)
        return obs

    def _write_command_observation(self, obs, scaled, target_slice, name, advance_history, reset_mask):
        """分别维护速度和高度历史，每个环境步只推进一次缓存。"""
        if obs.dim() == 2:
            obs[:, target_slice] = scaled
        elif obs.dim() == 3:
            shape = (obs.shape[0], obs.shape[1], scaled.shape[-1])
            history = self._histories.get(name)
            if history is None or tuple(history.shape) != shape:
                history = scaled[:, None, :].expand(shape).clone()
            else:
                if advance_history:
                    history = torch.roll(history, shifts=-1, dims=1)
                history[:, -1, :] = scaled
            if reset_mask is not None:
                mask = reset_mask.to(device=obs.device, dtype=torch.bool).reshape(-1)
                history[mask] = scaled[mask, None, :]
            self._histories[name] = history
            obs[:, :, target_slice] = history
        else:
            raise ValueError(f'Unsupported policy observation shape: {tuple(obs.shape)}')

    def close(self):
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None
