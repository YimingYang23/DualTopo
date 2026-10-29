# DualTopo on RATopo — OpenLane-V2 Subset B

Independent inference and official evaluation project for the selected complete DualTopo-enhanced RATopo Stage2 checkpoint. This folder is self-contained; use it in its own environment. The existing DualTopo project at the repository root is unchanged.

## Results

Full validation results from the accepted original evaluation, in percent:

| Branch | OLS | DET_l | DET_t | TOP_ll | TOP_lt |
| --- | ---: | ---: | ---: | ---: | ---: |
| singleframe | 53.6457 | 39.5619 | 56.7350 | 42.8199 | 27.9303 |
| temporal | 54.9715 | 41.4766 | 56.7350 | 45.8780 | 29.0965 |

The same checkpoint supports both branches. `singleframe` clears temporal memory; `temporal` uses chronological, scene-aware BEV/query memory. Geometry is 2D centerlines with 11 points, using the complete trained model including its point positional encoding and attention. The two subset checkpoints/configurations are not interchangeable.

## Installation

Use Linux and a CUDA GPU. The code was checked in the following legacy OpenMMLab environment; do not install these dependencies into the original root project's environment.

```bash
conda create -n dualtopo-ratopo-b python=3.8 -y
conda activate dualtopo-ratopo-b
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/ratopo_subset_b
pip install -r requirements.txt
python tools/verify_release.py
python tools/download_weights.py
python tools/verify_release.py --weights weights/model.safetensors
```

MMDetection3D may build CUDA extensions: use a compiler and CUDA toolkit compatible with PyTorch 1.9/CUDA 11.1. The validated runtime stack is recorded in `validation.json`; installation on a fresh machine was not separately tested. Dataset preparation follows [RATopo](https://github.com/buaa-colalab/RATopo#prepare-dataset) and [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2). Images/annotations are not distributed here.

## Data and inference

Point `DATA_ROOT` at the native OpenLane-V2 image root (containing `val/`), and `ANN` at `data_dict_subset_B_val.pkl`. Preserve annotation scene/frame order. The reported annotation SHA256 is `6358379714dd0885b65c22f7811da4d31d3df93eabe8b8d6e8844867875361a9`. A different pickle encoding may require `--allow-annotation-mismatch`; that option does not establish equivalence to the reported benchmark.

```bash
export DATA_ROOT=/path/to/OpenLane-V2
export ANN=/path/to/data_dict_subset_B_val.pkl
python tools/infer.py --weights weights/model.safetensors --data-root "$DATA_ROOT" --annotations "$ANN" --branch singleframe --output outputs/singleframe.pkl
python tools/infer.py --weights weights/model.safetensors --data-root "$DATA_ROOT" --annotations "$ANN" --branch temporal --output outputs/temporal.pkl
```

Inference uses batch size 1, FP32, deterministic chronological iteration and explicit scene reset. It includes native prediction decoding with `rescale=True`. The complete validation split is required; neither output is overwritten if it already exists. Process one project/branch per GPU at a time. The `.pkl` prediction files use Python pickle: load only trusted files.

## Official CPU scoring

```bash
python tools/evaluate.py --predictions outputs/singleframe.pkl --data-root "$DATA_ROOT" --annotations "$ANN" --output outputs/singleframe_metrics.json
python tools/evaluate.py --predictions outputs/temporal.pkl --data-root "$DATA_ROOT" --annotations "$ANN" --output outputs/temporal_metrics.json
```

Subset B has 6,019 frames and 22,916 traffic elements. All traffic-element GT boxes already have two points; no Subset A bbox normalization is applied. The original annotations and official evaluator code are unchanged.

The scorer verifies complete frame coverage, annotation checksum, expected GT shapes/counts and the vendored official evaluator SHA, and writes metrics plus a GT preprocessing receipt. The implementation and metric formulas remain the native OpenLane-V2 centerline protocol.

## Weights and validation scope

Weights: [Hugging Face / ratopo_subset_b](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_b). `model.safetensors` contains only model tensors, without optimizer, scheduler or runtime pickle objects. The SHA256 is `24bb40427a28ac8d3c7553206e1add1d5c4ffdabbc716214051a2f0151b84e97`; `weights.json` records the original checkpoint SHA, byte size and tensor count (2,677). Loading requires a strict key/shape match and checks the file hash and finite tensors.

The safe export was checked tensor-exactly against the selected original checkpoint. Separately launched public code was compared with the original implementation on 24 real frames from three scenes **per branch** on a V100S, including first frames, continuing history and scene reset. Inputs, BEV tensors and decoded outputs matched exactly. This bounded extraction check is distinct from the original full-split evaluation; it is not a new full-split GPU reproduction. See `validation.json` and `results.json`.

Historical provenance: the original Stage1 GPU supervisor lost its four exit receipts (`[null, null, null, null]`). Complete optimization records and finite model/optimizer/runtime checks were preserved. Later CPU finalization checks performed no optimization and do not recover those original GPU waits. Stage2 and full evaluation have their real successful exit receipts. Accordingly, `original_training_exit_receipts_complete=false` and `overall_contract_fully_satisfied=false`; evaluation acceptance must not be read as recovery of missing historical receipts.

This release provides inference/evaluation only. Native class definitions retain the methods required by OpenMMLab construction, but no training entrypoint, training datasets or optimizer/runtime checkpoint is distributed. Retained third-party code is credited in `NOTICE`; code is Apache-2.0 and dataset use remains subject to the dataset's terms. See the [RATopo source](https://github.com/buaa-colalab/RATopo), [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet), and [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2).
