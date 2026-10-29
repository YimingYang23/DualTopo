# DualTopo on RATopo

MMCV inference and evaluation for OpenLane-V2 SubsetA and SubsetB centerlines.

## Installation

Python 3.8, PyTorch 1.9, MMCV-full 1.5.2 and CUDA 11.1.

```bash
conda create -n dualtopo-ratopo python=3.8 -y
conda activate dualtopo-ratopo

python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/ratopo
python -m pip install -r requirements.txt
```

Use a CUDA toolkit/compiler compatible with PyTorch 1.9 and CUDA 11.1.

## Dataset

Follow [RATopo's dataset preparation](https://github.com/buaa-colalab/RATopo#prepare-dataset). Set `--data-root` to the image root referenced by the annotations.

| Subset | Annotation file | Geometry | Frames |
| --- | --- | --- | ---: |
| A | `data_dict_subset_A_val.pkl` | Seven cameras, 11-point XYZ centerlines | 4,806 |
| B | `data_dict_subset_B_val.pkl` | Six cameras, 11-point XY centerlines | 6,019 |

Configs: [SubsetA](configs/subset_a.py) · [SubsetB](configs/subset_b.py).

## Evaluation

### SubsetA

```bash
python tools/reproduce.py --subset A \
  --data-root /path/to/subset_a_image_root \
  --annotations /path/to/data_dict_subset_A_val.pkl \
  --output-dir outputs/subset_a
```

### SubsetB

```bash
python tools/reproduce.py --subset B \
  --data-root /path/to/subset_b_image_root \
  --annotations /path/to/data_dict_subset_B_val.pkl \
  --output-dir outputs/subset_b
```

Each command downloads the matching checkpoint ([SubsetA](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_a) / [SubsetB](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_b)), evaluates `singleframe` and `temporal` over the full validation split, and saves predictions and official metrics in the output directory.

Use `--branches temporal` to evaluate the temporal branch alone. `singleframe` resets history each frame; `temporal` carries history within each scene.

Individual commands are available in `tools/download_weights.py`, `tools/infer.py` and `tools/evaluate.py`. Each accepts `--subset A/B`; run it with `--subset A --help` for options.

## Results

Validation results (%).

| Subset | Branch | OLS | DET_l | DET_t | TOP_ll | TOP_lt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Single-frame | 48.65 | 31.22 | 49.77 | 31.88 | 32.64 |
| A | Temporal | 50.14 | 33.48 | 49.77 | 34.71 | 34.08 |
| B | Single-frame | 53.65 | 39.56 | 56.73 | 42.82 | 27.93 |
| B | Temporal | 54.97 | 41.48 | 56.73 | 45.88 | 29.10 |

SubsetA scoring converts five polygonal traffic-element annotations to native bounding boxes. Results and evaluation protocols: [SubsetA](backends/subset_a/results.json) · [SubsetB](backends/subset_b/results.json).

[License](LICENSE) · [Attribution](NOTICE)
