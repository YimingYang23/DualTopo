# DualTopo — rebuilt PyTorch implementation

This is the standalone **lane-segment** implementation of DualTopo, rebuilt in PyTorch. It uses the original released SubsetA checkpoint. The [MMCV RATopo project](../ratopo/) is installed and run separately.

| Item | This project |
| --- | --- |
| Environment | Python 3.11+, PyTorch 2.x, torchvision, Transformers 4.57.3; no MMCV/MMDetection/MMDetection3D |
| Task | Lane segments, lane boundaries/types, pedestrian areas, directed lane-segment topology |
| Data | OpenLane-V2 SubsetA, **LaneSegNet** `data_dict_subset_A_val_lanesegnet.pkl` |
| Geometry | Seven cameras; 10 XYZ points per center/left/right line |
| Checkpoint | `dualtopo_remake_scale025_subset_a.pth`, 1,023 tensors, SHA256 `5393ca1974be702b3408f037e561fb0a87cfe148f1f563d1169cd0eae1a97ba5` |
| Branches | `init`, `temporal`; both use the same weights |

## Installation

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch
git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install -e . --no-deps
python tools/verify_release.py
```

The PyTorch/CUDA wheel must match your NVIDIA driver. No local MMCV or custom CUDA-extension build is required. [Installation details](docs/installation.md) include CPU compatibility work. The validated extraction/fixture scope is recorded in `validation.json`; a fresh-environment installation and a complete new GPU benchmark are separate checks.

## Dataset

Follow [LaneSegNet's official preparation](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset) and [these dataset details](docs/data.md). Its converter reads SubsetA lane-segment annotations and writes `data_dict_subset_A_val_lanesegnet.pkl`. Do not use the native RATopo centerline pickle or the separate Map Element Bucket `*_ls.pkl` format.

`--data-root` must make `data_root / sensor[camera]['image_path']` point at the actual image. For `OpenLane-V2/val/...` metadata use the parent `data/` directory; for `val/...` use the OpenLane-V2 image root. The checker verifies schema, complete frame count, camera count, line shape and chronological order; `--check-images` also checks every image path.

```bash
python tools/check_data.py --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl --check-images
```

## Complete reproduction

```bash
python tools/reproduce.py --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

This downloads and verifies the checkpoint, runs **all 4,806 frames** in each branch, then executes the vendored native OpenLane-V2 **lane-segment** evaluator on CPU. It writes `init.pkl`, `temporal.pkl`, metrics JSONs and `reproduction.json`. Saved outputs are reused only when their receipts, branch, annotation checksum, checkpoint checksum and frame count match. A partial debug run cannot satisfy this workflow.

For individual steps:

```bash
python tools/download_weights.py --directory weights
python tools/verify_release.py --weights weights/dualtopo_remake_scale025_subset_a.pth
python tools/infer.py --checkpoint weights/dualtopo_remake_scale025_subset_a.pth \
  --data-root /path/to/image_root --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --branch temporal --output outputs/temporal.pkl
python tools/evaluate.py --predictions outputs/temporal.pkl \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl --output outputs/temporal_metrics.json
```

Use `--branch init` for the other pathway. **Init is not a per-frame cold-start ablation:** both pathways update temporal memories. See [inference semantics](docs/inference.md). The released lane-only checkpoint has **no traffic-element detector**; native traffic-element and lane-to-traffic outputs are empty. Its task/metrics must not be substituted with the RATopo centerline metrics.

The scorer follows native LaneSegNet GT formatting for pedestrian-area boundaries, retaining the original annotations in memory-safe copies and all validation frames. The official lane-segment evaluator and used distance formulas are vendored without mathematical edits, with source checksums in `evaluator.json`. No centerline-task bbox normalization is applied. Load only trusted annotation/prediction pickle files.

## Weights and validation

[Hugging Face checkpoint](https://huggingface.co/y352184741/DualTopo/blob/main/dualtopo_remake_scale025_subset_a.pth). The existing weight bytes and learned model source are preserved. `validation.json` records exact checkpoint loading and bounded first/history/scene-reset CPU fixtures. This is inference/evaluation code; no training entrypoint is included. See `LICENSE` and `NOTICE` for attribution.
