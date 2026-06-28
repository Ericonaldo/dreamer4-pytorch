"""DMC environment wrapper (embodied-style API).

Adapted from danijar/embodied embodied/envs/dmc.py:
https://github.com/danijar/embodied/blob/main/embodied/envs/dmc.py

Task names use underscore form, e.g. ``walker_walk`` -> domain ``walker``, task ``walk``.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import elements
import numpy as np
from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

# Vendored embodied (ref/embodied) when the package is not installed.
_REPO_EMBODIED = Path(__file__).resolve().parents[1] / "ref" / "embodied"
if _REPO_EMBODIED.is_dir() and str(_REPO_EMBODIED) not in sys.path:
    sys.path.insert(0, str(_REPO_EMBODIED))


def _load_embodied(name: str, rel_path: str):
    import importlib.util

    path = _REPO_EMBODIED / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


if "embodied" not in sys.modules:
    import types

    _emb = types.ModuleType("embodied")
    _core = types.ModuleType("embodied.core")
    _base = _load_embodied("embodied.core.base", "embodied/core/base.py")
    _emb.Env = _base.Env
    _core.Env = _base.Env
    _emb.core = _core
    sys.modules["embodied"] = _emb
    sys.modules["embodied.core"] = _core
    sys.modules["embodied.core.base"] = _base

ActionRepeat = _load_embodied("embodied.core.wrappers", "embodied/core/wrappers.py").ActionRepeat
FromDM = _load_embodied("embodied.envs.from_dm", "embodied/envs/from_dm.py").FromDM
Env = sys.modules["embodied.core.base"].Env


class DMC(Env):
    DEFAULT_CAMERAS = dict(
        quadruped=2,
        rodent=4,
    )

    def __init__(
        self,
        env,
        repeat=1,
        size=(64, 64),
        proprio=True,
        image=True,
        camera=-1,
    ):
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if isinstance(env, str):
            domain, task = env.split("_", 1)
            if camera == -1:
                camera = self.DEFAULT_CAMERAS.get(domain, 0)
            if domain == "cup":
                domain = "ball_in_cup"
            if domain == "manip":
                env = manipulation.load(task + "_vision")
            elif domain == "rodent":
                env = getattr(basic_rodent_2020, task)()
            else:
                env = suite.load(domain, task)
        self._dmenv = env
        self._env = FromDM(self._dmenv)
        self._env = ActionRepeat(self._env, repeat)
        self._size = size
        self._proprio = proprio
        self._image = image
        self._camera = camera

    @functools.cached_property
    def obs_space(self):
        basic = ("is_first", "is_last", "is_terminal", "reward")
        spaces = self._env.obs_space.copy()
        if not self._proprio:
            spaces = {k: spaces[k] for k in basic}
        key = "image" if self._image else "log/image"
        spaces[key] = elements.Space(np.uint8, self._size + (3,))
        return spaces

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        basic = ("is_first", "is_last", "is_terminal", "reward")
        if not self._proprio:
            obs = {k: obs[k] for k in basic}
        key = "image" if self._image else "log/image"
        obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
        for key, space in self.obs_space.items():
            if np.issubdtype(space.dtype, np.floating):
                assert np.isfinite(obs[key]).all(), (key, obs[key])
        return obs

    def close(self) -> None:
        if getattr(self, "_dmenv", None) is None:
            return
        try:
            _free_mujoco_render_context(self._dmenv.physics)
        except Exception:
            pass
        self._dmenv = None
        self._env = None


def _free_mujoco_render_context(physics) -> None:
    """Release EGL/OSMesa render context before process exit (avoids MjrContext __del__ errors)."""
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


class DMCEnvAdapter:
    """Thin adapter: ``reset()`` / ``step(np.ndarray)`` for policy eval scripts."""

    def __init__(self, env: DMC, *, max_episode_steps: int | None = None):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._action_key = "action"
        space = env.act_space[self._action_key]
        self.action_dim = int(np.prod(space.shape))
        self._zero_action = np.zeros(space.shape, dtype=np.float32)

    def reset(self) -> dict:
        self._step_count = 0
        return self._env.step({"reset": True, self._action_key: self._zero_action.copy()})

    def step(self, action: np.ndarray) -> dict:
        obs = self._env.step(
            {"reset": False, self._action_key: np.asarray(action, dtype=np.float32)}
        )
        if not obs["is_first"]:
            self._step_count += 1
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            obs["is_last"] = True
        return obs

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def make_dmc_env(
    task: str = "walker_walk",
    *,
    repeat: int = 2,
    image_size: int = 64,
    proprio: bool = False,
    image: bool = True,
    camera: int = -1,
    max_episode_steps: int | None = 1000,
) -> DMCEnvAdapter:
    """Build DMC env for online eval. ``task`` is e.g. ``walker_walk``."""
    core = DMC(
        task,
        repeat=repeat,
        size=(image_size, image_size),
        proprio=proprio,
        image=image,
        camera=camera,
    )
    return DMCEnvAdapter(core, max_episode_steps=max_episode_steps)
