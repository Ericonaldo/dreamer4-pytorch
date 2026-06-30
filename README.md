# Dreamer 4 (PyTorch)

Minimal PyTorch Lightning reimplementation of [Dreamer 4](https://arxiv.org/abs/2509.24527) for DMC Walker Walk.

References: [Dreamer v4 Paper (arXiv)](https://arxiv.org/pdf/2509.24527), [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)(only tokenzier + dynamics training), [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax) (no imagine rl), [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4).

## Training stages

Three-stage pipeline (paper-aligned); configs under `configs/walker_walk/`:

| Stage | Name | What trains | Config (this repo) |
|-------|------|-------------|-------------------|
| **1** | **Tokenizer** | Causal patch **encoder + decoder**; block-causal transformer with **MAE** random patch masking → latent bottleneck `z_t` + recon loss | `tokenizer.yaml` |
| **2** | **Pretraining** | **BC + dynamics** on frozen tokenizer encode: joint **flow matching** (`wm_dynamics`) and **BC** action/reward MTP (`wm_agent`) on one backbone | `bc_dynamics.yaml` |
| **3** | **Posttraining** | **RL** in latent imagination: rollout with learned dynamics + policy, value head on TD(λ) returns (not BC MTP) | `policy_imagination_pmpo.yaml` |

Stage 1 decoder is dropped at inference for downstream stages (encode-only). Stage 3 uses imagined trajectories, not dataset BC labels.

## Network structure

Symbols: `B` batch, `T` aligned timesteps, `A=6` action dim, `L=8` MTP horizon, `D` dynamics `embed_dim`, `L_z=16` tokenizer latents, `D_z=32` latent dim, `k=2` packing factor, `S_sp=L_z/k=8` spatial slots, `D_sp=D_z·k=64`, `R=4` register tokens, `N=1` agent token, `P=(H/patch)²` image patches.

**Time alignment** (`GranularEpisodeDataset` → `align_dynamics_batch` in `data.py`):

| Stage | `window_mode` | Raw window from dataset | Model batch |
|-------|---------------|-------------------------|-------------|
| Tokenizer | `frame` | `seq_len` consecutive frames | `image (B,T,H,W,C)` — one frame per step is typical |
| bc_dynamics / rl | `transition` | obs `[s0..sT]` (**T+1** frames), actions `[a1..aT]`, rewards `[r1..rT]` (**T** transitions) | see table below |

Dreamer transition: `s_{t-1} -- a_t --> s_t`, reward `r_t` labels the step into `s_t`.  
`align_dynamics_batch` prepends **NULL** action/reward at `t=0` so every row shares the same index as the observation:

| time `t` | observation | action input `a_t` | reward `r_t` | latent |
|----------|-------------|--------------------|--------------|--------|
| 0 | s₀ | **0** (NULL) | **0** | z̄₀ ← encode(s₀) |
| ≥1 | s_t | a_t (action that led to s_t) | r_t | z̄_t ← encode(s_t) |

Aligned shapes passed to dynamics / BC: `image (B,T,H,W,C)`, `action (B,T,A)`, `reward (B,T)` with **`T = seq_len + 1`**.

**Tokenizer** — **encoder** frozen after stage 1; encodes observations for dynamics / BC. **Decoder** used only in tokenizer training and when decoding latents back to pixels (recon viz, rollout eval); not used on the BC env loop.

Encoder (frozen encode → dynamics / BC):
```
s_t  (B,T,H,W,C)
  → patches (B,T,P,patch_dim)
  → per-timestep tokens [latent×L_z | patch×P]  (B,T,L_z+P,D_tok)
  → z_t  (B,T,L_z,D_z)
  → pack → z̄_t  (B,T,S_sp,D_sp)     # group k latents per spatial slot
```

Decoder (tokenizer train + decode video only):
```
z_t  (B,T,L_z,D_z)
  → up-proj latents + patch queries → decoder transformer
  → patch logits → ŝ_t  (B,T,H,W,C)   MAE recon loss at train; rollout / viz decode
```

**Dynamics** (`DynamicsModel.forward`):
```
Per timestep t, build spatial token sequence (dim S = 2+S_sp+R+N = 15 by default):

  x_t = [ a_t | σ_t | z̄_t | reg | agent_t ]   each slot → (D,)
        (1)   (1)  (S_sp) (R)   (N)

  a_t:      actions (B,T,A) → ActionEncoder → (B,T,1,D)
  σ_t:      flow noise level (B,T) → MLP → (B,T,1,D); 0 at BC inference
  z̄_t:      packed latents (B,T,S_sp,D_sp) → Linear → (B,T,S_sp,D)
  reg:      learned register (B,T,R,D)
  agent_t:  learned (B,T,N,D); flow uses wm_dynamics mask, BC uses wm_agent mask

  x = cat over slots → (B,T,S,D) → block-causal transformer →
    spatial out → x̂₁ (B,T,S_sp,D_sp)   flow head, predict clean z̄
    agent out   → h_t (B,T,N,D)         BC readout slot
```

**BC heads** (`PolicyModel` + `AgentHeads`, trained jointly in stage 2):
```
h = mean(h_t, dim agent)  (B,T,D)
AgentHeads(h):
  policy  → squashed Gaussian  (B,T,L,A)   MTP slot ℓ predicts a_{t+ℓ}, NLL
  reward  → symexp twohot MLP  (B,T,L)     MTP slot ℓ predicts r_{t+ℓ}, twohot CE
```

**BC + dynamics** (`bc_dynamics` stage, `BCDynamicsModule` — config `bc_dynamics.yaml`):
```
Same shared backbone (PolicyModel.dynamics + learned agent_tokens + AgentHeads).
Frozen tokenizer encode once → packed_z, action, reward.

Two forwards per step, different space attention masks (space_modes: [wm_dynamics, wm_agent]):

  1) flow (space_mode=wm_dynamics)
     σ_t ~ Uniform(0,1) per (B,T); agent_t present but isolated (world ignores agent keys)
     → flow head x̂₁ vs clean z̄_t
     loss_flow = MSE(x̂₁, z̄_t)

  2) BC (space_mode=wm_agent)
     σ_t = 0; agent_t attends world (spatial/register/action/noise)
     → h_t → AgentHeads → bc_loss (action NLL + reward twohot CE)

  loss = flow_weight · loss_flow + bc_loss
```

## TODO

### Transformer (vs paper §architecture)

Paper baseline: *pre-layer RMSNorm, RoPE, SwiGLU, QKNorm, attention logit soft capping*.

| Component | Paper | This repo | Location |
|-----------|-------|-----------|----------|
| Pre-layer RMSNorm | ✓ | ✓ | `models/transformer_blocks.py` — `norm1/2/3` before space/time attn & MLP |
| SwiGLU FFN | ✓ | ✓ | `MLP`: `u * silu(v)` gated FFN |
| RoPE | ✓ | ✗ | Uses additive sinusoidal positions on token embeddings (`add_sinusoidal_positions`), not rotary Q/K |
| QKNorm | ✓ | ✗ | `MultiheadSelfAttention` — no norm on Q/K heads |
| Attention logit soft capping | ✓ | ✗ | Standard `scaled_dot_product_attention`, no logit cap |

- [ ] **RoPE** — apply rotary embeddings to Q/K in space & time attention (replace or pair with current sinusoidal token bias)
- [ ] **QKNorm** — RMSNorm (or equivalent) on Q and K per head before dot product
- [ ] **Attention logit soft capping** — e.g. `softcap * tanh(logits / softcap)` before softmax (tune cap hyperparameter)

### Tokenizer

- [x] Block-causal MAE encoder–decoder
- [x] Train / val loop, step-based eval, reconstruction viz, wandb
- [x] Multi-GPU DDP
- [x] Latent temporal collapse — tune `embed_dim` / `latent_dim` (default configs: `embed_dim=64`, `latent_dim=32`, `patch_size=8`, `n_latents=16`; large 512-dim runs collapsed); monitor `tokenizer/z_temporal_std`
- [x] Robust checkpoint pruning (`KeepLastCheckpoints` handles Lightning `-v1` suffixes, rank-0 only)
- [x] Resume from Lightning checkpoint (`train.resume_ckpt`)
- [x] Standalone tokenizer eval script (recon metrics + panels from checkpoint) — see `eval/tokenizer.py`
- [ ] Ablation: `scale_pos_embeds` off (paper notes it can help)

### Pretraining (stage 2 — BC + dynamics)

- [x] `DynamicsModel` — action + noise-conditioned flow on packed tokenizer latents
- [x] Simple flow-matching loss (empirical MSE on clean latents, no shortcut/bootstrap)
- [x] `BCDynamicsModule` — joint flow + BC on shared backbone (`bc_dynamics.yaml`)
- [ ] Shortcut forcing + bootstrap self-consistency loss (ref `dynamics_pretrain_loss` self branch); Discrete noise schedule / `k_max` grid (ref uses finest-step flow grid)
- [x] Agent token slot — `n_agent=1`, learned `agent_tokens`, dual space masks (`wm_dynamics`, `wm_agent`)
- [x] Action-conditioned rollout eval — dataset actions, autoregressive latent sampling, decode, `val/rollout_mse` / PSNR vs floor, wandb viz
- [x] Standalone rollout eval script — panels, return-band sweeps, long mp4 rollouts (`eval/dynamics_rollout.py`)
- [x] `PolicyModel` + `AgentHeads` — L-step action NLL + reward symexp twohot MTP (no value head in pretrain)
- [ ] ~`TaskEmbedder` for multi-task BC — skipped for now: only Walker Walk is implemented, so a single learned `agent_tokens` parameter is enough and avoids an extra embedding table with no conditioning signal; add when training multiple tasks on shared weights~
- [ ] **Reward MTP readout switch (config)** — selectable via YAML, not implemented yet:
  - `symexp_twohot` (current): `SymExpTwoHotHead` + twohot CE; raw rewards from dataloader, **no norm**
  - `mse`: simple MLP → scalar `(B,T,L)` + MSE on L-step future rewards; **normalize rewards in dataloader** (e.g. running mean/std or fixed scale in `data` / `train` config); denorm for metrics only
  - Config sketch: `model.reward_head: symexp_twohot | mse`; when `mse`, add `data.reward_norm` (or `train.reward_norm`) for dataloader stats / clip range

### Imagination RL (stage 3)

- [x] `imagine_latent_rollout` in latent space (`modules/rl.py`)
- [x] `RLModule` — imagination RL on top of BC (`modules/rl.py`); value head with TD(λ) return targets
- [x] Config `configs/walker_walk/policy_imagination_pmpo.yaml`
- [x] RL post-training stability: `init_value_head_from_reward_head`, policy warmup (value-only), `context_len_min`, `32-true` precision, PMPO log-prob recompute on fixed actions (see `analysis/imagination_rl.md`)

### Data & infra

- [ ] `data.obs_mode=proprio` / `both` paths through dynamics & policy (tokenizer is image-only today)
- [x] DMC online env eval (`env.py` + `agent.py`; CLI `dreamer4-eval`, training `AsyncPolicyEval`)

## Setup

```bash
uv sync
```

Dataset uses the [granular](https://github.com/danijar/granular) format:

```python
import granular

dataset = granular.ShardedDatasetReader("dmc_walker_walk", granular.decoders)
elem = dataset[0]  # {'data': {...}, 'length': T}
```

Place the dataset at `data/dmc_walker_walk/`.

## Training pipeline

See **Training stages** above. Runnable configs:

| Stage | Config | Notes |
|-------|--------|-------|
| 1 Tokenizer | `tokenizer.yaml` | MAE encoder–decoder |
| 2 Pretraining | `bc_dynamics.yaml` | Joint flow + BC on one backbone |
| 3 Posttraining | `policy_imagination_pmpo.yaml` | Imagination RL (PMPO) + async env eval |

```bash
# Full model
uv run dreamer4-train configs/walker_walk/tokenizer.yaml

# Override any config field
uv run dreamer4-train configs/walker_walk/tokenizer.yaml data.obs_mode=image log.wandb=true
```

### Multi-GPU (DDP)

Lightning spawns workers when `train.devices > 1` (`strategy=ddp` in `train.py`). Run a **single** process; do not combine with `torchrun`.

`train.batch_size` is **per GPU**. Global batch = `batch_size × devices` (default `16 × 8 = 128`).

```bash
# 8 GPUs (local or server)
uv run dreamer4-train configs/walker_walk/tokenizer.yaml \
  train.devices=8 \
  data.num_workers=4 \
  log.wandb=true
```

Tune `train.batch_size` if OOM; scale `data.num_workers` per GPU (e.g. 2–4).

### Imagination RL (PMPO)

Stage `rl` on frozen tokenizer + dynamics + BC reward; trains value + policy in latent imagination. **Stability details (advantage bias, warmup, context, precision, PMPO log-prob):** see local `analysis/imagination_rl.md` (gitignored).

We previously supported an optional PPO path but removed it: clipped PPO in latent imagination tended to explode easily (ratio blow-ups, unstable policy updates, env return collapse) despite stabilization attempts; this repo now trains PMPO only.

**Prerequisites:** `tokenizer_ckpt` and `bc_ckpt` in the yaml (default: `bc_dynamics` run / `last.ckpt`). **Start from a bc_dynamics checkpoint, not a corrupted RL checkpoint.**

**RL post-training stability (defaults in yaml + code):**

| Measure | Config / code | Why |
|---------|---------------|-----|
| Value init from BC reward | `init_value_head_from_reward_head()` in `RLModule` | Avoids \(V(s)\approx 0\) while imagined returns are ~5–15 → advantages almost all positive → PMPO destroys BC |
| Value / policy warmup | `imagination.policy_warmup_steps: 500` | First N steps train **value only** (`train_policy=False`, policy `lr=0`); saves `warmup_end.ckpt` for resume |
| Bounded imagination context | `imagination.context_len_min: 8` (+ `data.seq_len: 16`) | Rollout context suffix sampled in `[context_len_min, seq_len]` (not from length 1); set `context_len_min == seq_len` for fixed context |
| Full fp32 RL training | `train.precision: 32-true` | BC/dynamics use `bf16-mixed`; PMPO `exp` / `atanh` / `Normal` / KL are numerically sensitive in mixed precision |

| Config | Default warmup | Notes |
|--------|----------------|-------|
| `policy_imagination_pmpo.yaml` | 500 steps | `policy_lr: 3e-5` |

```bash
# PMPO (Dreamer4 paper default)
uv run dreamer4-train configs/walker_walk/policy_imagination_pmpo.yaml \
  log.wandb=true

# 8 GPUs (server; headless MuJoCo)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  uv run dreamer4-train configs/walker_walk/policy_imagination_pmpo.yaml \
  train.devices=8 train.batch_size=64 \
  log.run_name=walker_walk/rl_pmpo log.wandb=true

# Smoke (1 GPU, no wandb)
uv run dreamer4-train configs/walker_walk/policy_imagination_pmpo.yaml \
  train.max_steps=100 train.devices=1 train.batch_size=32 \
  train.val_every=100 log.wandb=false \
  log.run_name=walker_walk/rl_pmpo_smoke eval.env_eval=false
```

`val/env_return_mean` from async env eval (`eval.env_eval: true`, every `train.val_every` steps). Override checkpoints:

```bash
uv run dreamer4-train configs/walker_walk/policy_imagination_pmpo.yaml \
  bc_ckpt=logs/walker_walk/bc_dynamics_10m/checkpoints/last.ckpt
```

### Resume training

```bash
# Continue to train.max_steps (global_step restored from ckpt)
uv run dreamer4-train configs/walker_walk/tokenizer.yaml \
  train.max_steps=20000 \
  train.resume_ckpt=logs/walker_walk/tokenizer/checkpoints/step-step=10000.ckpt

# Rebuild cosine on current max_steps at global_step (not ckpt scheduler / floor lr)
uv run dreamer4-train configs/walker_walk/tokenizer.yaml \
  train.max_steps=20000 \
  train.resume_ckpt=logs/walker_walk/tokenizer/checkpoints/step-step=10000.ckpt \
  train.resume_reset_scheduler=true
```

`train.resume_reset_scheduler=true` keeps model, optimizer, and `global_step`, then sets `CosineAnnealingLR.T_max` to current `max_steps` and `last_epoch=global_step` (checkpoint state restores the old `T_max`, so this override is required when extending `max_steps`).

### Tokenizer validation

Episode-level hold-out via `data.val_fraction` (default 5%). Metrics: `val/loss_mae` (masked MSE, same as train), `val/loss_full` (all-patch MSE, stable), and `val/z_temporal_std` (latent diversity across time; should stay well above ~1e-3). Reconstruction panels (`target | masked | recon_masked | recon_full`) go to `logs/<run_name>/viz/` and wandb (`tokenizer/viz`) when `log.wandb=true`. Tune `train.val_every`, `train.val_max_batches`, `log.viz_max_items`.

### Dynamics rollout eval (standalone)

Script: `eval/dynamics_rollout.py`. Loads frozen tokenizer + dynamics checkpoint, replays **dataset actions** (open-loop) via `dreamer4.eval_utils.dynamics_rollout_eval` / `dynamics_rollout_video`, decodes latents, and compares to GT.

**Panel eval** (default): autoregressive rollout with context lengths `ctx=1..rollout_ctx`; GT frames in context, predicted frames after. Outputs `rollout_panel_all.png`, `rollout_traj_{i:02d}.png`, and `metrics.json` (`rollout_mse`, `rollout_psnr`, repeat-last-frame floor). Horizon comes from `train.rollout_ctx`, `train.rollout_horizon`, `train.rollout_flow_steps`.

**Long rollout videos** (`--rollout-video`): single GT frame `obs[0]` as context; for step `g` predicting frame `g`, attend to all past latents while `g <= L`, then the previous `L` only (`L` = `--attn-window`, default `train.rollout_ctx`). Default rollout length 64 (`--rollout-length`). Writes `videos/rollout_video_*.mp4`, `videos/rollout_video_*_gt_pred.mp4` (GT over pred), and `video_metrics.json`.

**Episode selection**: `--split train|val|all`; `--min-episode-return` / `--max-episode-return`; `--by-reward-bands` for Walker Walk return strata (writes `summary.json` + per-band dirs).

```bash
# Panels + metrics on val split
uv run python -m eval.dynamics_rollout configs/walker_walk/bc_dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/bc_dynamics_10m/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_eval \
  --split val --max-items 4

# 64-step rollout mp4s (context obs[0], attn window 8)
uv run python -m eval.dynamics_rollout configs/walker_walk/bc_dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/bc_dynamics_10m/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_videos \
  --rollout-video --rollout-length 64 --attn-window 8 --max-items 2

# Rollout stratified by cumulative return
uv run python -m eval.dynamics_rollout configs/walker_walk/bc_dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/bc_dynamics_10m/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_bands \
  --by-reward-bands --split val
```

### BC policy eval (online DMC)

CLI: `dreamer4-eval` (`eval/policy.py`) — thin wrapper over `dreamer4/agent.py` and `dreamer4/eval_utils.py` (`DreamerAgent`, `run_policy_env_eval`, `AsyncPolicyEval`). When the training YAML lives in `configs/walker_walk/`, merges `policy_eval.yaml` underneath it for heavier offline eval defaults (`episodes: 50`, `num_envs: 8`; training yaml `eval.episodes` is typically smaller). `eval.action_horizon` (default **1**) controls open-loop eval: **1** = closed-loop (replan + forward every env step); **L>1** = forward once then execute MTP slots `1..L` without re-forwarding, but still **encode every env frame** and commit actions into `(z, a)` history (capped by `model.action_horizon - 1`). Regenerate horizon sweep charts with `python -m dreamer4.plot_action_horizon_eval`.

**Multi-GPU**: `--gpus 8` splits 50 episodes across 8 GPUs; each GPU runs `eval.num_envs` parallel envs (8 in `policy_eval.yaml`).

**Policy video** (`--video-out`): one online episode mp4; `--annotate-video` overlays `step 0, 1, …` (top-right) and episode return on the last frame.

```bash
CKPT=logs/walker_walk/bc_dynamics_10m/checkpoints/step-step=50000.ckpt
CFG=configs/walker_walk/bc_dynamics.yaml

# 50 episodes, 8 GPUs × 8 envs per GPU
uv run dreamer4-eval $CFG --bc-ckpt $CKPT \
  --episodes 50 --gpus 8 \
  --out logs/walker_walk/bc_dynamics_10m/eval/env_step_50000.json

# Annotated policy eval video
uv run dreamer4-eval $CFG --bc-ckpt $CKPT \
  --video-out logs/walker_walk/bc_dynamics_10m/eval/policy_video_step50000.mp4 \
  --annotate-video --video-fps 20
```

### Dynamics rollout videos (expert vs weak, 64-step)

Pick episodes by **full-episode cumulative return**, then extract a **64-step** transition window (`--rollout-length 64` sets dataset `seq_len=64`). Open-loop rollout from `obs[0]` with dataset actions; `--annotate-steps` labels each frame `step 0, 1, …` (top-right).

```bash
CKPT=logs/walker_walk/bc_dynamics_10m/checkpoints/step-step=50000.ckpt
CFG=configs/walker_walk/bc_dynamics.yaml

# Expert trajectory (return >= 900)
uv run python -m eval.dynamics_rollout $CFG \
  --dynamics-ckpt $CKPT \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_expert64_step50000 \
  --split all --max-items 1 \
  --min-episode-return 900 \
  --rollout-video --rollout-length 64 --attn-window 8 \
  --skip-panel --annotate-steps --video-fps 15

# Weak trajectory (return < 200)
uv run python -m eval.dynamics_rollout $CFG \
  --dynamics-ckpt $CKPT \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_weak64_step50000 \
  --split all --max-items 1 \
  --max-episode-return 200 \
  --rollout-video --rollout-length 64 --attn-window 8 \
  --skip-panel --annotate-steps --video-fps 15
```

Outputs: `videos/rollout_video_ep*_ret*.mp4` (predicted frames) and `videos/rollout_video_ep*_ret*_gt_pred.mp4` (GT over pred), plus `video_metrics.json`.

## Observation modes

Set `data.obs_mode` in config:

- `image` — pixels only (tokenizer training)
- `proprio` — proprioception only
- `both` — image + proprioception

## Project layout

```
dreamer4/
  config.py           # YAML config (OmegaConf)
  checkpoint.py       # load_state for Lightning ckpts
  data.py             # Granular dataset + batch collation
  models/             # NN modules
    transformer_blocks.py
    tokenizer.py
    dynamics.py
    policy.py         # AgentHeads, PolicyModel, bc_loss, SymExpTwoHot readouts
  modules/            # Lightning modules per stage
    base.py
    tokenizer.py
    bc_dynamics.py
    rl.py
  agent.py            # DreamerAgent, load_policy_modules (online inference)
  env.py              # DMC environment
  eval_utils.py       # Eval rollouts: dynamics (offline) + policy (online DMC)
  callbacks.py        # Lightning callbacks, AsyncPolicyEval, val panel logging
  train.py            # Trainer, dataloaders, dreamer4-train entry
eval/                 # Standalone eval scripts + viz
  common.py
  smoke_test.py       # Checkpoint compatibility baseline
  dynamics_rollout.py
  tokenizer.py
  imagination.py
  policy.py           # dreamer4-eval CLI
  viz/                # Panel / frame annotations
configs/walker_walk/
  tokenizer.yaml
  bc_dynamics.yaml
  policy_eval.yaml
  policy_imagination_pmpo.yaml   # imagination RL (PMPO)
```
