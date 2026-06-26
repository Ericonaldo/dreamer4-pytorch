# Dreamer 4 (PyTorch)

Minimal PyTorch Lightning reimplementation of [Dreamer 4](https://arxiv.org/abs/2509.24527) for DMC Walker Walk.

References: [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax), [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4).

Network structure
```
Tokenizer (frozen) → packed z_t
Dynamics:
  inputs: a_t, z_t, agent_t
  output: h_t
AgentHeads(h_t):
  - policy: MLP → (B,T,L,A)  L-step action, MSE
  - reward: MLP → (B,T,L)
  - value:  MLP → (B,T)
```

Data token order

**Tokenizer** (`window_mode=frame`):
```
t:  frame s_t
```

**Dynamics / BC** (`window_mode=transition`, after `align_dynamics_batch`):
```
t=0:  frame s0,  action 0 (NULL), reward 0
t≥1:  frame s_t, action a_t,      reward r_t
```

Dataset raw (transition): `obs [s0..sT]`, `actions [a1..aT]`, `rewards [r1..rT]`.

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
- [ ] Resume from Lightning checkpoint (`train.resume_ckpt`)
- [ ] Standalone tokenizer eval script (recon metrics + panels from checkpoint)
- [ ] Ablation: `scale_pos_embeds` off (paper notes it can help)

### Dynamics (stage 2)

- [x] `DynamicsModel` — action + noise-conditioned flow on packed tokenizer latents
- [x] Simple flow-matching loss (empirical MSE on clean latents, no shortcut/bootstrap)
- [x] `DynamicsModule` training with frozen tokenizer encode
- [ ] Shortcut forcing + bootstrap self-consistency loss (ref `dynamics_pretrain_loss` self branch)
- [ ] Discrete noise schedule / `k_max` grid (ref uses finest-step flow grid)
- [x] Agent token slot — `n_agent=1`, zero agent at pretrain, `wm_dynamics` (ref `wm_agent_isolated`); `forward` returns `h_t` for BC
- [x] `wm_agent` space mask — agent attends world; world/action ignore agent keys
- [x] Action-conditioned rollout eval — dataset actions, autoregressive latent sampling, decode, `val/rollout_mse` / PSNR vs floor, wandb viz
- [x] Dynamics config aligned with tokenizer ckpt (`tokenizer_ckpt`, arch in `dynamics.yaml`)

### BC (stage 3)

- [x] `BCModel` — dynamics init (`dynamics_ckpt`), `wm_agent`, learned agent tokens, `AgentHeads` (L-step action / reward + value, MSE)
- [x] `BCModule` training with frozen tokenizer encode
- [x] Config `configs/walker_walk/bc.yaml`, smoke `bc_debug.yaml`
- [ ] `TaskEmbedder` for multi-task BC — skipped for now: only Walker Walk is implemented, so a single learned `agent_tokens` parameter is enough and avoids an extra embedding table with no conditioning signal; add when training multiple tasks on shared weights
- [ ] Closed-loop L-step rollout BC (policy-fed actions into dynamics)
- [ ] Config tuning; reward/value targets (returns vs raw reward)

### Policy / imagination (stage 4)

- [ ] `imagine_rollout` in latent space (`imagination.py` stub)
- [ ] `PolicyModule` — imagination RL on top of BC (`modules/policy.py`)
- [ ] Config `configs/walker_walk/policy.yaml`

### Data & infra

- [ ] `data.obs_mode=proprio` / `both` paths through dynamics & policy (tokenizer is image-only today)
- [ ] DMC online env integration (`env.py`) for policy eval
- [ ] Git initial commit & CI smoke test

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

Four stages, each driven by a YAML config under `configs/walker_walk/`:

| Stage | Config | Description |
|-------|--------|-------------|
| 1 | `tokenizer.yaml` | Causal patch tokenizer |
| 2 | `dynamics.yaml` | Interactive dynamics model |
| 3 | `bc.yaml` | BC policy + reward heads |
| 4 | `policy.yaml` | Imagination RL on top of BC |

```bash
# Full model
uv run dreamer4-train configs/walker_walk/tokenizer.yaml

# Tiny smoke test (small model, fewer steps)
uv run dreamer4-train configs/walker_walk/tokenizer_debug.yaml

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

### Tokenizer validation

Episode-level hold-out via `data.val_fraction` (default 5%). Metrics: `val/loss_mae` (masked MSE, same as train), `val/loss_full` (all-patch MSE, stable), and `val/z_temporal_std` (latent diversity across time; should stay well above ~1e-3). Reconstruction panels (`target | masked | recon_masked | recon_full`) go to `logs/<run_name>/viz/` and wandb (`tokenizer/viz`) when `log.wandb=true`. Tune `train.val_every`, `train.val_max_batches`, `log.viz_max_items`.

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
    policy.py         # AgentHeads
  modules/            # Lightning modules per stage
    base.py
    tokenizer.py
    dynamics.py
    bc.py
    policy.py
  imagination.py      # Latent rollouts (stub)
  train.py            # Trainer, callbacks, dataloaders
  cli.py
configs/walker_walk/
  tokenizer.yaml
  tokenizer_w_lpips.yaml
  tokenizer_debug.yaml
  dynamics.yaml
  bc.yaml
  policy.yaml
```
