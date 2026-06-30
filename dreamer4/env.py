"""Walker Walk DMC environment (minimal dm_control wrapper, no embodied)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_SUPPORTED_TASK = "walker_walk"


def _ensure_mujoco_gl() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


def _free_mujoco_render_context(physics) -> None:
    """Release EGL/OSMesa render context before process exit."""
    if physics is None:
        return
    ctx = getattr(physics, "context", None)
    if ctx is None:
        return
    try:
        ctx.free()
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        physics.context = None
    except Exception:
        pass


class WalkerEnv:
    """``reset()`` / ``step(action)`` with uint8 ``obs['image']`` (H, W, C)."""

    def __init__(
        self,
        *,
        repeat: int = 1,
        image_size: int = 64,
        camera_id: int = 0,
        max_episode_steps: int | None = 1000,
    ):
        _ensure_mujoco_gl()
        from dm_control import suite

        self._repeat = max(1, int(repeat))
        self._size = (int(image_size), int(image_size))
        self._camera_id = int(camera_id)
        self._max_episode_steps = max_episode_steps
        self._env = suite.load("walker", "walk")
        self._physics = self._env.physics
        self.action_dim = int(np.prod(self._env.action_spec().shape))
        self._step_count = 0

    def _render(self) -> np.ndarray:
        return self._physics.render(*self._size, camera_id=self._camera_id)

    def reset(self) -> dict[str, Any]:
        self._step_count = 0
        self._env.reset()
        return {
            "image": self._render().copy(),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
        }

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32)
        reward = 0.0
        last = False
        for _ in range(self._repeat):
            ts = self._env.step(action)
            reward += float(ts.reward or 0.0)
            last = bool(ts.last())
            if last:
                break

        self._step_count += 1
        is_last = last
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            is_last = True

        return {
            "image": self._render().copy(),
            "reward": reward,
            "is_first": False,
            "is_last": is_last,
        }

    def close(self) -> None:
        if self._env is None:
            return
        try:
            _free_mujoco_render_context(self._physics)
        except Exception:
            pass
        self._env = None
        self._physics = None


def make_dmc_env(
    task: str = _SUPPORTED_TASK,
    *,
    repeat: int = 1,
    image_size: int = 64,
    proprio: bool = False,
    image: bool = True,
    camera: int = -1,
    max_episode_steps: int | None = 1000,
) -> WalkerEnv:
    """Build Walker Walk env for online eval."""
    if task != _SUPPORTED_TASK:
        raise ValueError(f"only {_SUPPORTED_TASK!r} is supported, got {task!r}")
    if not image:
        raise ValueError("image=False is not supported in the minimal walker env")
    if proprio:
        raise ValueError("proprio=True is not supported in the minimal walker env")
    camera_id = 0 if camera < 0 else int(camera)
    return WalkerEnv(
        repeat=repeat,
        image_size=image_size,
        camera_id=camera_id,
        max_episode_steps=max_episode_steps,
    )
