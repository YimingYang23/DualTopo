# DualTopo

PyTorch inference and evaluation for OpenLane-V2 SubsetA lane segments.

## Installation

Python 3.11, PyTorch 2.6 and Transformers 4.57.3.

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install -e . --no-deps
```

See [installation checks](docs/installation.md#check-the-environment) to confirm the environment.

## Dataset

Prepare SubsetA with [LaneSegNet's data converter](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset):

- **Annotations:** `data_dict_subset_A_val_lanesegnet.pkl` (4,806 frames).
- **Images:** seven cameras per frame; set `--data-root` to the image root referenced by the annotations.

See [data preparation](docs/data.md) for image paths and conversion details.

## Evaluation

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

The command downloads the [SubsetA checkpoint](https://huggingface.co/y352184741/DualTopo/blob/main/dualtopo_remake_scale025_subset_a.pth), evaluates all 4,806 frames through the `init` and `temporal` pathways, and saves predictions and official metrics in `outputs/subset_a`.

Use `--branches temporal` or `--branches init` to evaluate one branch. See [manual weight download](docs/weights.md) to use a local checkpoint, or [individual inference and scoring commands](docs/inference.md) to run each step separately.

## Results

SubsetA validation results (%).

| Branch | mAP | AP_ls | AP_ped | TOP_lsls | mAP_type |
| --- | ---: | ---: | ---: | ---: | ---: |
| Init | 36.94 | 35.18 | 38.69 | 28.45 | 50.49 |
| Temporal | 40.47 | 39.53 | 41.42 | 32.11 | 53.85 |

[License](LICENSE) · [Attribution](NOTICE)
