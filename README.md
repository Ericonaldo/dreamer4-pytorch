# Dreamer 4 (PyTorch)

Minimal PyTorch Lightning reimplementation of [Dreamer 4](https://arxiv.org/abs/2509.24527) for DMC Walker Walk.

References: [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax).

## Setup

```bash
uv sync --extra data --extra log
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

Episode-level hold-out via `data.val_fraction` (default 5%). Metrics: `val/loss_mae` (masked MSE, same as train) and `val/loss_full` (all-patch MSE, stable). Reconstruction panels (`target | masked | recon_masked | recon_full`) go to `logs/<run_name>/viz/` and wandb (`tokenizer/viz`) when `log.wandb=true`. Tune `train.val_every`, `train.val_max_batches`, `log.viz_max_items`.

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
uv sync --extra data --extra log
uv run python -c "import torch; print(torch.cuda.device_count())"
```

## Project layout

```
dreamer4/
  config.py       # YAML config loading (OmegaConf)
  data.py         # Granular dataset + batch collation
  models.py       # Tokenizer, dynamics, agent heads
  imagination.py  # Latent-space rollouts
  train.py        # Lightning modules + trainer setup
  cli.py          # Entry point
configs/walker_walk/
  tokenizer.yaml
  dynamics.yaml
  bc.yaml
  policy.yaml
```

## Observation modes

Set `data.obs_mode` in config:

- `image` — pixels only
- `proprio` — proprioception only
- `both` — image + proprioception

## Status

Scaffold is in place. Tokenizer training is implemented; dynamics / BC / policy are still stubs.
