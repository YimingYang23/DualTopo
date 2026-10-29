# DualTopo — PyTorch

Inference and evaluation for OpenLane-V2 SubsetA lane segments.

- **Environment:** Python 3.11+, PyTorch 2.x, Transformers 4.57.3.
- **Data:** LaneSegNet `data_dict_subset_A_val_lanesegnet.pkl`; seven cameras and 10 XYZ points per center/left/right line.
- **Weights:** [DualTopo SubsetA](https://huggingface.co/y352184741/DualTopo/blob/main/dualtopo_remake_scale025_subset_a.pth).

## Installation

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch
git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-eval.txt
pip install -e . --no-deps
```

## Data

Prepare the metadata with [LaneSegNet's converter](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset). Set `--data-root` so that `data_root / image_path` resolves to each image. See [dataset instructions](docs/data.md).

## Testing

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

Runs 4,806 frames in the `init` and `temporal` pathways, followed by official lane-segment evaluation. Predictions and metrics are saved in the output directory. Use `--branches temporal` to test the temporal pathway alone.

Both pathways update temporal memory. See [inference details](docs/inference.md) for branch semantics and individual commands in `tools/infer.py --help` and `tools/evaluate.py --help`.

## Results (%)

SubsetA lane segments, 4,806 frames.

| Branch | mAP | AP_ls | AP_ped | TOP_lsls | mAP_type |
| --- | ---: | ---: | ---: | ---: | ---: |
| init | 36.94 | 35.18 | 38.69 | 28.45 | 50.49 |
| temporal | 40.47 | 39.53 | 41.42 | 32.11 | 53.85 |

[License](LICENSE) · [Attribution](NOTICE)
