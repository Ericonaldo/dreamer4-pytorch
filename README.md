# Dreamer 4 (PyTorch)

Minimal PyTorch Lightning reimplementation of [Dreamer 4](https://arxiv.org/abs/2509.24527) for DMC Walker Walk.

References: [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax), [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4).

## Training stages

Three-stage pipeline (paper-aligned); configs under `configs/walker_walk/`:

| Stage | Name | What trains | Config (this repo) |
|-------|------|-------------|-------------------|
| **1** | **Tokenizer** | Causal patch **encoder + decoder**; block-causal transformer with **MAE** random patch masking → latent bottleneck `z_t` + recon loss | `tokenizer.yaml`, `tokenizer_5m.yaml` |
| **2** | **Pretraining** | **BC + dynamics** on frozen tokenizer encode: joint **flow matching** (`wm_dynamics`) and **BC** action/reward MTP (`wm_agent`) on one backbone; optional `dynamics.yaml` warm start | `bc_dynamics.yaml`, `bc_dynamics_10m.yaml` (also `bc.yaml` for BC-only finetune) |
| **3** | **Posttraining** | **RL** in latent imagination: rollout with learned dynamics + policy, value head on TD(λ) returns (not BC MTP) | `policy.yaml` *(planned)* |

Stage 1 decoder is dropped at inference for downstream stages (encode-only). Stage 3 uses imagined trajectories, not dataset BC labels.

## Network structure

Symbols: `B` batch, `T` aligned timesteps, `A=6` action dim, `L=8` MTP horizon, `D` dynamics `embed_dim`, `L_z=16` tokenizer latents, `D_z=32` latent dim, `k=2` packing factor, `S_sp=L_z/k=8` spatial slots, `D_sp=D_z·k=64`, `R=4` register tokens, `N=1` agent token, `P=(H/patch)²` image patches.

**Time alignment** (`GranularEpisodeDataset` → `align_dynamics_batch` in `data.py`):

| Stage | `window_mode` | Raw window from dataset | Model batch |
|-------|---------------|-------------------------|-------------|
| Tokenizer | `frame` | `seq_len` consecutive frames | `image (B,T,H,W,C)` — one frame per step is typical |
| Dynamics / BC / bc_dynamics | `transition` | obs `[s0..sT]` (**T+1** frames), actions `[a1..aT]`, rewards `[r1..rT]` (**T** transitions) | see table below |

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
  agent_t:  zeros at dynamics pretrain; learned (B,T,N,D) at BC

  x = cat over slots → (B,T,S,D) → block-causal transformer →
    spatial out → x̂₁ (B,T,S_sp,D_sp)   flow head, predict clean z̄
    agent out   → h_t (B,T,N,D)         BC readout slot
```

**BC** (`BCModel`, stage `bc` or heads-only finetune):
```
h = mean(h_t, dim agent)  (B,T,D)
AgentHeads(h):
  policy  → squashed Gaussian  (B,T,L,A)   MTP slot ℓ predicts a_{t+ℓ}, NLL
  reward  → symexp twohot MLP  (B,T,L)     MTP slot ℓ predicts r_{t+ℓ}, twohot CE
```

**BC + dynamics** (`bc_dynamics` stage, `BCDynamicsModule` — config `bc_dynamics.yaml`):
```
Same shared backbone (BCModel.dynamics + learned agent_tokens + AgentHeads).
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

BC-only stage (`bc`) uses wm_agent only; dynamics-only pretrain uses wm_dynamics with agent_t = 0.
Optional warm start: dynamics_ckpt from stage 2, then joint train flow + BC heads together.
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
- [x] Optional LPIPS loss (`train.use_lpips`, `tokenizer_w_lpips.yaml`)
- [x] Multi-GPU DDP
- [x] Latent temporal collapse — tune `embed_dim` / `latent_dim` (default configs: `embed_dim=64`, `latent_dim=32`, `patch_size=8`, `n_latents=16`; large 512-dim runs collapsed); monitor `tokenizer/z_temporal_std`
- [x] Robust checkpoint pruning (`KeepLastCheckpoints` handles Lightning `-v1` suffixes, rank-0 only)
- [x] Resume from Lightning checkpoint (`train.resume_ckpt`)
- [ ] Standalone tokenizer eval script (recon metrics + panels from checkpoint) — see `eval_tokenizer.py`
- [ ] Ablation: `scale_pos_embeds` off (paper notes it can help)

### Dynamics (stage 2)

- [x] `DynamicsModel` — action + noise-conditioned flow on packed tokenizer latents
- [x] Simple flow-matching loss (empirical MSE on clean latents, no shortcut/bootstrap)
- [x] `DynamicsModule` training with frozen tokenizer encode
- [ ] Shortcut forcing + bootstrap self-consistency loss (ref `dynamics_pretrain_loss` self branch); Discrete noise schedule / `k_max` grid (ref uses finest-step flow grid)
- [x] Agent token slot — `n_agent=1`, zero agent at pretrain, `wm_dynamics` (ref `wm_agent_isolated`); `forward` returns `h_t` for BC
- [x] `wm_agent` space mask — agent attends world; world/action ignore agent keys
- [x] Action-conditioned rollout eval — dataset actions, autoregressive latent sampling, decode, `val/rollout_mse` / PSNR vs floor, wandb viz
- [x] Standalone rollout eval script — panels, return-band sweeps, long mp4 rollouts (`eval_dynamics_rollout.py`)
- [x] Dynamics config aligned with tokenizer ckpt (`tokenizer_ckpt`, arch in `dynamics.yaml`)

### BC (stage 3)

- [x] `BCModel` — dynamics init (`dynamics_ckpt`), `wm_agent`, learned agent tokens, `AgentHeads` (L-step action NLL + reward symexp twohot MTP; no value head in BC)
- [x] `BCModule` training with frozen tokenizer encode
- [x] `BCDynamicsModule` — joint flow + BC on shared backbone (`bc_dynamics.yaml`)
- [x] Config `configs/walker_walk/bc.yaml`, `bc_dynamics.yaml`
- [ ] ~`TaskEmbedder` for multi-task BC — skipped for now: only Walker Walk is implemented, so a single learned `agent_tokens` parameter is enough and avoids an extra embedding table with no conditioning signal; add when training multiple tasks on shared weights~
- [ ] **Reward MTP readout switch (config)** — selectable via YAML, not implemented yet:
  - `symexp_twohot` (current): `SymExpTwoHotHead` + twohot CE; raw rewards from dataloader, **no norm**
  - `mse`: simple MLP → scalar `(B,T,L)` + MSE on L-step future rewards; **normalize rewards in dataloader** (e.g. running mean/std or fixed scale in `data` / `train` config); denorm for metrics only
  - Config sketch: `model.reward_head: symexp_twohot | mse`; when `mse`, add `data.reward_norm` (or `train.reward_norm`) for dataloader stats / clip range

### Policy / imagination (stage 4)

- [ ] `imagine_rollout` in latent space (`imagination.py` stub)
- [ ] `PolicyModule` — imagination RL on top of BC (`modules/policy.py`); value head with TD(λ) return targets
- [ ] Config `configs/walker_walk/policy.yaml`

### Data & infra

- [ ] `data.obs_mode=proprio` / `both` paths through dynamics & policy (tokenizer is image-only today)
- [x] DMC online env eval (`env.py` + `policy_agent.py`; CLI `dreamer4-eval`, training `AsyncBCEval`)

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
| 1 Tokenizer | `tokenizer.yaml`, `tokenizer_5m.yaml` | MAE encoder–decoder |
| 2 Pretraining | `bc_dynamics.yaml`, `bc_dynamics_10m.yaml` | Joint flow + BC; optional `dynamics.yaml` init |
| 3 Posttraining | `policy.yaml` | Imagination RL *(planned)* |

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

# Same on embo (after uv sync)
cd ~/mhliu/dreamer4-pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python -m dreamer4.cli configs/walker_walk/tokenizer.yaml \
  train.devices=8 \
  data.num_workers=4 \
  log.wandb=true
```

Tune `train.batch_size` if OOM; scale `data.num_workers` per GPU (e.g. 2–4).

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

Script: `dreamer4/eval_dynamics_rollout.py`. Loads frozen tokenizer + dynamics checkpoint, replays **dataset actions** (open-loop), decodes latents, and compares to GT.

**Panel eval** (default): autoregressive rollout with context lengths `ctx=1..rollout_ctx`; GT frames in context, predicted frames after. Outputs `rollout_panel_all.png`, `rollout_traj_{i:02d}.png`, and `metrics.json` (`rollout_mse`, `rollout_psnr`, repeat-last-frame floor). Horizon comes from `train.rollout_ctx`, `train.rollout_horizon`, `train.rollout_flow_steps`.

**Long rollout videos** (`--rollout-video`): single GT frame `obs[0]` as context; for step `g` predicting frame `g`, attend to all past latents while `g <= L`, then the previous `L` only (`L` = `--attn-window`, default `train.rollout_ctx`). Default rollout length 64 (`--rollout-length`). Writes `videos/rollout_video_*.mp4`, `videos/rollout_video_*_gt_pred.mp4` (GT over pred), and `video_metrics.json`.

**Episode selection**: `--split train|val|all`; `--min-episode-return` / `--max-episode-return`; `--by-reward-bands` for Walker Walk return strata (writes `summary.json` + per-band dirs).

```bash
# Panels + metrics on val split
uv run python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/dynamics/rollout_eval \
  --split val --max-items 4

# 64-step rollout mp4s (context obs[0], attn window 8)
uv run python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/dynamics/rollout_videos \
  --rollout-video --rollout-length 64 --attn-window 8 --max-items 2

# Rollout stratified by cumulative return
uv run python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \
  --out-dir logs/walker_walk/dynamics/rollout_bands \
  --by-reward-bands --split val
```

### BC policy eval (online DMC)

CLI: `dreamer4-eval` (`dreamer4/eval_policy.py`) — thin wrapper over `dreamer4/policy_agent.py` (`BCPolicy`, `run_bc_env_eval`, `AsyncBCEval`). Merges `configs/walker_walk/policy_eval.yaml` when present (`max_history: 16`, `episodes: 50`, `num_envs: 8`). `eval.action_horizon` (default **1**) controls open-loop eval: **1** = closed-loop (replan from current obs each step, MTP slot 1); **L>1** = execute MTP slots `1..L` without re-encoding before the next replan (capped by `model.action_horizon - 1`).

**Multi-GPU**: `--gpus 8` splits 50 episodes across 8 GPUs; each GPU runs `eval.num_envs` parallel envs (8 in `bc_dynamics_10m.yaml`).

**Policy video** (`--video-out`): one online episode mp4; `--annotate-video` overlays `step 0, 1, …` (top-right) and episode return on the last frame.

```bash
CKPT=logs/walker_walk/bc_dynamics_10m/checkpoints/step-step=50000.ckpt
CFG=configs/walker_walk/bc_dynamics_10m.yaml

# 50 episodes, 8 GPUs × 8 envs per GPU
uv run dreamer4-eval $CFG --policy bc --bc-ckpt $CKPT \
  --episodes 50 --gpus 8 \
  --out logs/walker_walk/bc_dynamics_10m/eval/env_step_50000.json

# Annotated policy eval video
uv run dreamer4-eval $CFG --policy bc --bc-ckpt $CKPT \
  --video-out logs/walker_walk/bc_dynamics_10m/eval/policy_video_step50000.mp4 \
  --annotate-video --video-fps 20
```

### Dynamics rollout videos (expert vs weak, 64-step)

Pick episodes by **full-episode cumulative return**, then extract a **64-step** transition window (`--rollout-length 64` sets dataset `seq_len=64`). Open-loop rollout from `obs[0]` with dataset actions; `--annotate-steps` labels each frame `step 0, 1, …` (top-right).

```bash
CKPT=logs/walker_walk/bc_dynamics_10m/checkpoints/step-step=50000.ckpt
CFG=configs/walker_walk/bc_dynamics_10m.yaml

# Expert trajectory (return >= 900)
uv run python -m dreamer4.eval_dynamics_rollout $CFG \
  --dynamics-ckpt $CKPT \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_expert64_step50000 \
  --split all --max-items 1 \
  --min-episode-return 900 \
  --rollout-video --rollout-length 64 --attn-window 8 \
  --skip-panel --annotate-steps --video-fps 15

# Weak trajectory (return < 200)
uv run python -m dreamer4.eval_dynamics_rollout $CFG \
  --dynamics-ckpt $CKPT \
  --out-dir logs/walker_walk/bc_dynamics_10m/rollout_weak64_step50000 \
  --split all --max-items 1 \
  --max-episode-return 200 \
  --rollout-video --rollout-length 64 --attn-window 8 \
  --skip-panel --annotate-steps --video-fps 15
```

Outputs: `videos/rollout_video_ep*_ret*.mp4` (predicted frames) and `videos/rollout_video_ep*_ret*_gt_pred.mp4` (GT over pred), plus `video_metrics.json`.

On `embo`, use `.venv/bin/dreamer4-eval` and `.venv/bin/python` if `uv` is unavailable; data symlink: `data/dmc_walker_walk`.

## Remote training

Code is developed locally; experiments run on `ssh embo` at `~/mhliu/dreamer4-pytorch`.

### Dataset (Google Drive, shared with your account)

`rclone` is installed on `embo`. From your Mac (browser OAuth once):

```bash
brew install rclone   # if needed
./scripts/rclone_setup_gdrive.sh      # log in with your Google account
./scripts/rclone_download_walker.sh   # ~15GB, background on server
./scripts/rclone_extract_walker.sh    # tar -> data/dmc_walker_walk
```

Monitor: `ssh embo tail -f ~/mhliu/rclone_download.log`


```bash
# On server (first time)
cd ~/mhliu/dreamer4-pytorch
uv sync
uv run python -c "import torch; print(torch.cuda.device_count())"
```

## Observation modes

Set `data.obs_mode` in config:

- `image` — pixels only (tokenizer training)
- `proprio` — proprioception only
- `both` — image + proprioception

## Project layout

```
dreamer4/
  config.py           # YAML config (OmegaConf)
  data.py             # Granular dataset + batch collation
  models/             # NN modules
    transformer_blocks.py
    tokenizer.py
    dynamics.py
    policy.py         # AgentHeads, BCModel, bc_loss, SymExpTwoHot readouts
  modules/            # Lightning modules per stage
    base.py
    tokenizer.py
    dynamics.py
    bc.py
    bc_dynamics.py
    policy.py
  imagination.py      # Latent rollouts (stub)
  train.py            # Trainer, callbacks, dataloaders
  cli.py
  policy_agent.py         # BCPolicy, online eval, AsyncBCEval (used by eval_policy + BC modules)
  eval_dynamics_rollout.py  # Standalone dynamics rollout panels + optional mp4
  eval_policy.py            # CLI: dreamer4-eval (random / BC metrics / policy video)
  eval_tokenizer.py         # Standalone tokenizer recon eval
  video_utils.py            # Frame annotation for eval videos
configs/walker_walk/
  tokenizer.yaml
  tokenizer_5m.yaml
  tokenizer_w_lpips.yaml
  dynamics.yaml
  bc.yaml
  bc_dynamics.yaml
  bc_dynamics_10m.yaml
  policy_eval.yaml
  policy.yaml
```
