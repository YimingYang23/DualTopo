# DualTopo on RATopo

Inference and evaluation for OpenLane-V2 SubsetA 3D and SubsetB 2D centerlines.

## Installation

Linux, an NVIDIA GPU, Python 3.8 and the PyTorch CUDA 11.1 wheels. Use a separate environment from the PyTorch DualTopo project.

```bash
conda create -n dualtopo-ratopo python=3.8 -y
conda activate dualtopo-ratopo

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/ratopo

python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
python -m pip install -r requirements.txt
python -m pip check
```

## Data preparation

Download the **Image (val)** files for your subset from [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md#download).

For **SubsetA**, download the preprocessed validation annotations matching the checkpoint:

```bash
python tools/prepare_data.py --subset A \
  --output data/data_dict_subset_A_val.pkl
```

These [validation annotations](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_a/annotations) use the [OpenLane-V2 data license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

For **SubsetB**, also download **OpenLane-V2 info** and `data_dict_subset_B.json`, extract them alongside the images, and convert:

```text
OpenLane-V2/
├── data_dict_subset_B.json
└── val/
    └── <segment_id>/
        ├── info/<timestamp>.json
        └── image/<camera>/<timestamp>.jpg
```

```bash
python tools/prepare_data.py --subset B \
  --data-root /path/to/subset_b/OpenLane-V2 \
  --output data/data_dict_subset_B_val.pkl
```

For inference, `--data-root` is joined with each image path stored in the pickle:

| Image path in the pickle | Inference `--data-root` |
| --- | --- |
| `OpenLane-V2/val/...` | the parent of `OpenLane-V2` |
| `val/...` | the `OpenLane-V2` directory |

Check all images before inference:

```bash
python tools/check_data.py --subset A \
  --annotations data/data_dict_subset_A_val.pkl \
  --data-root /path/to/subset_a_image_root --check-images
```

Use `--subset B` and the SubsetB paths to check SubsetB.

## Evaluation

### SubsetA

First check both branches on three frames:

```bash
python tools/reproduce.py --subset A \
  --data-root /path/to/subset_a_image_root \
  --annotations data/data_dict_subset_A_val.pkl \
  --limit 3 --output-dir outputs/check_subset_a
```

Then evaluate all 4,806 frames:

```bash
python tools/reproduce.py --subset A \
  --data-root /path/to/subset_a_image_root \
  --annotations data/data_dict_subset_A_val.pkl \
  --output-dir outputs/subset_a
```

### SubsetB

Check three frames:

```bash
python tools/reproduce.py --subset B \
  --data-root /path/to/subset_b_image_root \
  --annotations data/data_dict_subset_B_val.pkl \
  --limit 3 --output-dir outputs/check_subset_b
```

Then evaluate all 6,019 frames:

```bash
python tools/reproduce.py --subset B \
  --data-root /path/to/subset_b_image_root \
  --annotations data/data_dict_subset_B_val.pkl \
  --output-dir outputs/subset_b
```

Each command downloads the matching checkpoint and runs GPU inference followed by official CPU scoring for `singleframe` and `temporal`. Use the full split for the results below.

Outputs are `singleframe.pkl`, `temporal.pkl`, `singleframe_metrics.json` and `temporal_metrics.json`. The JSON `percent_metrics` field contains the percentages shown below.

Add `--branches temporal` or `--branches singleframe` to select one branch, or `--device cuda:1` to select a GPU.

## Download weights separately

```bash
python tools/download_weights.py --subset A
python tools/download_weights.py --subset B
```

Files are saved in `weights/`. For an existing checkpoint, add `--weights /path/to/model.safetensors` to `tools/reproduce.py`.

To run inference or scoring separately, use `tools/infer.py` or `tools/evaluate.py` with `--subset A --help` (or `--subset B --help`).

## Results

Validation results (%).

| Subset | Branch | OLS | DET_l | DET_t | TOP_ll | TOP_lt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Single-frame | 48.65 | 31.22 | 49.77 | 31.88 | 32.64 |
| A | Temporal | 50.14 | 33.48 | 49.77 | 34.71 | 34.08 |
| B | Single-frame | 53.65 | 39.56 | 56.73 | 42.82 | 27.93 |
| B | Temporal | 54.97 | 41.48 | 56.73 | 45.88 | 29.10 |

Results and evaluation protocols: [SubsetA](backends/subset_a/results.json) · [SubsetB](backends/subset_b/results.json).

[License](LICENSE) · [Attribution](NOTICE)
