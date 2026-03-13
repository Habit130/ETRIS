# ETRIS

This repository is adapted from the official PyTorch implementation of
[Bridging Vision and Language Encoders: Parameter-Efficient Tuning for Referring Image Segmentation](https://arxiv.org/abs/2307.11545)
for a custom JSON-based referring segmentation dataset.

## What Changed

- The project no longer uses RefCOCO / RefCOCO+ / RefCOCOg or LMDB inputs.
- Training no longer contains any validation logic.
- Testing now reports only 5 final metrics:
  - `iou`
  - `dice`
  - `recall`
  - `miou`
  - `macc`
- Dataset captions use a single configurable caption index, default `caption[2]`.
- Masks are binarized with `mask > 0` by default.

## Dataset Layout

The default dataset root is the sibling directory `../dataset` relative to the
repository root. The expected layout is:

```text
Segmentation/
├─ ETRIS/
└─ dataset/
   ├─ train.json
   ├─ test.json
   ├─ train/
   │  ├─ img/
   │  └─ lbl/
   └─ test/
      ├─ img/
      └─ lbl/
```

Each JSON item must contain:

```json
{
  "id": "sample_id",
  "image": "train/img/sample.jpg",
  "mask": "train/lbl/sample.png",
  "caption": [
    "caption 0",
    "caption 1",
    "caption 2"
  ]
}
```

Paths in JSON are resolved relative to `DATA.dataset_root`.

## Environment

For Linux with RTX 4090, use the provided conda environment:

```bash
bash run_scripts/setup_linux.sh
conda activate etris
```

If you prefer pip on top of an existing PyTorch environment:

```bash
pip install -r requirement.txt
```

## Pretrained Weights

Place the CLIP weights under `pretrain/`:

```text
pretrain/
├─ RN50.pt
├─ RN101.pt
└─ ViT-B-16.pt
```

## Configs

Custom dataset configs are under `config/custom/`:

- `bridge_r50.yaml`
- `bridge_r101.yaml`
- `bridge_v16.yaml`

Key dataset fields:

- `DATA.dataset_root`
- `DATA.train_json`
- `DATA.test_json`
- `DATA.caption_index`
- `DATA.mask_foreground_threshold`
- `TEST.pred_threshold`

## Train

```bash
bash run_scripts/train.sh
```

Or directly:

```bash
torchrun --nproc_per_node=2 train.py --config config/custom/bridge_r101.yaml
```

## Test

Testing loads `last_model.pth` by default and only prints the final 5 metrics.

```bash
bash run_scripts/test.sh
```

## Server Run Steps

Assume the server layout is:

```text
workdir/
├─ ETRIS/
└─ dataset/
```

Then run:

```bash
cd workdir/ETRIS
bash run_scripts/setup_linux.sh
conda activate etris
bash run_scripts/train.sh
bash run_scripts/test.sh
```

If your dataset directory is not `../dataset`, override it at runtime:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/custom/bridge_r101.yaml \
  --opts DATA.dataset_root /absolute/path/to/dataset
```

Or directly:

```bash
python test.py --config config/custom/bridge_r101.yaml
```

## Notes

- Training keeps the original epoch and milestone defaults: `epochs=50`,
  `milestones=[35]`.
- No validation dataset is constructed at any stage.
- Large datasets, checkpoints, weights, and experiment outputs remain external
  to Git tracking.
