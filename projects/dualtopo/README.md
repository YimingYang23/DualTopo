# DualTopo

Inference and evaluation for OpenLane-V2 SubsetA lane segments.

## Installation

Linux, an NVIDIA GPU with a CUDA 12.4-compatible driver, and Python 3.11.
Use a separate environment from the MMCV RATopo project.

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install -e . --no-deps
```

## Data preparation

Download [OpenLane-V2 SubsetA](https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md#download), using the **Map Element Bucket info** and **Image (val)** downloads, plus `data_dict_subset_A.json`. Then run:

```bash
python tools/prepare_data.py \
  --data-root /path/to/OpenLane-V2 \
  --output data/data_dict_subset_A_val_lanesegnet.pkl
```

Set the inference `--data-root` to the root of the image paths in this pickle. See [data preparation](docs/data.md) for the directory layout and file check.

## Evaluation

First run a three-frame check of both branches:

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations data/data_dict_subset_A_val_lanesegnet.pkl \
  --limit 3 --output-dir outputs/check_subset_a
```

Run all 4,806 validation frames:

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations data/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

Each command downloads the checkpoint and runs GPU inference followed by CPU scoring for `init` and `temporal`. The three-frame check tests setup; use the full run for the results below.

Outputs are `init.pkl`, `temporal.pkl`, `init_metrics.json` and `temporal_metrics.json`. Metric values are percentages.

Add `--branches temporal` or `--branches init` to select one branch, or `--device cuda:1` to select a GPU. See [weight download](docs/weights.md) and [separate inference/scoring commands](docs/inference.md).

## Results

SubsetA validation results (%).

| Branch | mAP | AP_ls | AP_ped | TOP_lsls | mAP_type |
| --- | ---: | ---: | ---: | ---: | ---: |
| Init | 36.94 | 35.18 | 38.69 | 28.45 | 50.49 |
| Temporal | 40.47 | 39.53 | 41.42 | 32.11 | 53.85 |

[License](LICENSE) · [Attribution](NOTICE)
