# PlantSeg Linux Server Runbook

## Scope

This repository is adapted for Linux server execution on a single RTX 4090 with CUDA 11.8 and Python 3.10.
The supported plantseg path is ViT-B-16 only.

## Expected layout

- Repository root: `ETRIS`
- Dataset root: sibling directory `../plantseg`
- Required dataset files:
  - `../plantseg/main.json`
  - `../plantseg/images/`
  - `../plantseg/ann/`

## Environment setup

Run `run_scripts/setup_server_env.sh`.

This script:

- creates or updates the conda environment from `environment.server.yml`
- downloads `pretrain/ViT-B-16.pt` from a Hugging Face mirror
- verifies the downloaded weight with SHA256
- runs a minimal import check for the locked runtime dependencies

## Dataset preparation

Build LMDB splits from `../plantseg/main.json` by running:

```bash
python tools/plantseg_to_lmdb.py --data-root ../plantseg --overwrite
```

The converter uses zero-based `caption[3]` as the only text prompt for every sample and writes:

- `../plantseg/lmdb/train.lmdb`
- `../plantseg/lmdb/val.lmdb`
- `../plantseg/lmdb/test.lmdb`

## Training

Use the locked plantseg ViT config:

```bash
bash run_scripts/train_plantseg.sh
```

Default behavior:

- single GPU only
- `config/plantseg/bridge_v16.yaml`
- outputs under `exp/plantseg/plantseg_v16_64_8_512_3`

## Post-training validation

Run:

```bash
bash run_scripts/test_plantseg.sh
```

The checkpoint path is resolved automatically from the config and experiment name.

## Reported metrics

Validation and final test report these five metrics:

- `IoU`: foreground IoU
- `Dice`: foreground Dice
- `Recall`: foreground recall
- `mIoU`: average of foreground and background IoU
- `mACC`: average of foreground and background recall

The prediction threshold is fixed at `0.35`.
