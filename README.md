# Dreamer 4 (PyTorch)

Minimal PyTorch Lightning reimplementation of [Dreamer 4](https://arxiv.org/abs/2509.24527) for DMC Walker Walk.

References: [Dreamer v4 Paper (arXiv)](https://arxiv.org/pdf/2509.24527), [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)(only tokenizer + dynamics training), [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax) (no imagine rl), [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4).

## Training stages

Three-stage pipeline (paper-aligned); configs under `configs/walker_walk/`:

| Stage | Name | What trains | Config (this repo) |
|-------|------|-------------|-------------------|
| **1** | **Tokenizer** | Causal patch **encoder + decoder**; block-causal transformer with **MAE** random patch masking → latent bottleneck `z_t` + recon loss | `tokenizer.yaml` |
| **2** | **Pretraining** | **BC + dynamics** on frozen tokenizer encode: joint **shortcut forcing** (`wm_dynamics`) and **BC** action/reward MTP (`wm_agent`) on one backbone | `bc_dynamics.yaml` |
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
Per timestep t, build spatial token sequence (dim S = 3+S_sp+R+N = 16 by default):

  x_t = [ a_t | τ_t | d_t | z̄_t | reg | agent_t ]   each slot → (D,)
        (1)   (1)   (1)  (S_sp) (R)   (N)

  a_t:      actions (B,T,A) → ActionEncoder → (B,T,1,D)
  τ_t:      signal level index (B,T) → Embedding → (B,T,1,D)
  d_t:      step size index (B,T) → Embedding → (B,T,1,D)
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
  reward  → SymlogHead MLP  (B,T,L)     MTP slot ℓ predicts symlog(r_{t+ℓ}), MSE in symlog space; decode with symexp
```

**BC + dynamics** (`bc_dynamics` stage, `BCDynamicsModule` — config `bc_dynamics.yaml`):
```
Same shared backbone (PolicyModel.dynamics + learned agent_tokens + AgentHeads).
Frozen tokenizer encode once → packed_z, action, reward.

Two forwards per step, different space attention masks (space_modes: [wm_dynamics, wm_agent]):

  1) shortcut flow (space_mode=wm_dynamics)
     Sample step size d and signal level τ on discrete grid (k_max); corrupt z̄_t
     → flow head x̂₁ vs clean z̄_t (finest step) + bootstrap self-consistency (coarser steps)
     loss_flow = shortcut_forcing_loss (empirical + bootstrap branches, ramp weight w(τ)=0.9τ+0.1)

  2) BC (space_mode=wm_agent)
     τ = clean, d = finest; agent_t attends world (spatial/register/action/signal/step)
     → h_t → AgentHeads → bc_loss (action NLL + reward symlog MSE)

  loss = flow_weight · loss_flow + bc_loss
```

### Shortcut model (dynamics)

Paper-style [shortcut forcing](https://arxiv.org/abs/2509.24527): x-prediction on packed latents with discrete signal level **τ** and step size **d**. Implemented in `shortcut_forcing_loss` (`models/dynamics.py`); aligned with [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) `dynamics_pretrain_loss`.

**Training** (each batch, `space_mode=wm_dynamics`):

| Branch | Rows | step `d` | Loss |
|--------|------|----------|------|
| Flow (empirical) | ~75% | finest `d_min = 1/k_max` | MSE(x̂₁, z̄) with ramp `w(τ)=0.9τ+0.1` |
| Bootstrap (self) | ~25% after step 5000 | coarser `d ∈ {1/2, 1/4, …}` | large-step velocity vs avg of two half-step velocities (target detached) |

τ and `d` are sampled on a power-of-two grid; `k_max` is the finest training grid (default **64**). Bootstrap starts at `train.shortcut_bootstrap_start` (default **5000**).

**Inference** (rollout / imagination): generate each latent frame with **K forward passes** (paper default **K=4**). `flow_steps` must be a power of two and divide `k_max`.

| Config key | Stage | Default |
|------------|-------|---------|
| `model.dynamics.k_max` | train + infer | `64` |
| `train.rollout_flow_steps` | bc_dynamics val / `eval/dynamics_rollout.py` | `4` |
| `imagination.flow_steps` | RL latent rollout | `4` |
| `train.shortcut_self_fraction` | bootstrap row fraction | `0.25` |
| `train.shortcut_bootstrap_start` | global step before bootstrap | `5000` |

**Note:** shortcut checkpoints are **not compatible** with pre-shortcut `bc_dynamics` weights (`noise_mlp` → τ/d embeddings). Retrain stage 2 from a tokenizer ckpt.

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

Run the three stages in order:

```bash
# 1. Tokenizer
uv run dreamer4-train configs/walker_walk/tokenizer.yaml

# 2. BC + dynamics pretraining
uv run dreamer4-train configs/walker_walk/bc_dynamics.yaml

# 3. Imagination RL posttraining
uv run dreamer4-train configs/walker_walk/policy_imagination_pmpo.yaml
```

Override any config field from the command line:

```bash
uv run dreamer4-train configs/walker_walk/tokenizer.yaml data.obs_mode=image log.wandb=true
```

For multi-GPU training, set `train.devices > 1`; Lightning uses DDP internally, so do not launch with torchrun:

```bash
uv run dreamer4-train configs/walker_walk/tokenizer.yaml \
  train.devices=8 \
  data.num_workers=4 \
  log.wandb=true
```

`train.batch_size` is per GPU.

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

Hold-out via `data.val_fraction` (default 5%). Metrics: `val/loss_mae`, `val/loss_full`, `val/z_temporal_std`.

### Eval

| Script | What | Checkpoint |
|--------|------|------------|
| `dreamer4-eval` (`eval/policy.py`) | Online DMC env rollout (BC or RL policy) | `--policy-ckpt` (bc_dynamics or rl Lightning ckpt with `model.*`) |
| `eval/dynamics_rollout.py` | Open-loop dynamics on dataset actions | `--dynamics-ckpt` (uses `train.rollout_flow_steps`, default 4) |
| `eval/imagination.py` | RL latent imagination panels / videos (not env eval) | `--rl-ckpt` (uses `imagination.flow_steps`, default 4) |

Walker configs merge `policy_eval.yaml` under `dreamer4-eval`. See each CLI `--help` for panels, videos, episode filters, and `--gpus`.

```bash
uv run python -m eval.dynamics_rollout configs/walker_walk/bc_dynamics.yaml \
  --dynamics-ckpt logs/walker_walk/bc_dynamics_10m/checkpoints/last.ckpt --split val

uv run dreamer4-eval configs/walker_walk/bc_dynamics.yaml \
  --policy-ckpt logs/walker_walk/bc_dynamics/checkpoints/last.ckpt

# Same CLI for RL: pass the rl checkpoint (loads model.* policy weights; value_head unused in env).
uv run dreamer4-eval configs/walker_walk/policy_imagination_pmpo.yaml \
  --policy-ckpt logs/walker_walk/rl_pmpo/checkpoints/last.ckpt \
  --gpus 8 --episodes 64 --out logs/walker_walk/rl_pmpo/eval_env.json

uv run python -m eval.imagination configs/walker_walk/policy_imagination_pmpo.yaml \
  --rl-ckpt logs/walker_walk/rl_pmpo/checkpoints/last.ckpt
```

## Observation modes

Set `data.obs_mode` in config:

- `image` — pixels only (tokenizer training; required for dynamics / policy today)
- `proprio` — proprioception only (loaded by `GranularEpisodeDataset`; not wired through tokenizer / dynamics training)
- `both` — image + proprioception (same limitation as `proprio` for stages 2–3)

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
    policy.py         # AgentHeads, PolicyModel, bc_loss, SymlogHead (reward + RL value)
  trainers/           # Lightning modules per stage
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
  data_stats.py       # Episode return / band helpers for eval scripts
  dynamics_rollout.py
  tokenizer.py
  imagination.py      # RL policy latent rollout videos (--rl-ckpt)
  policy.py           # dreamer4-eval CLI
  viz/                # Panel / frame annotations
configs/walker_walk/
  tokenizer.yaml
  bc_dynamics.yaml
  policy_eval.yaml
  policy_imagination_pmpo.yaml   # imagination RL (PMPO)
```
