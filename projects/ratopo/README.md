# DualTopo on RATopo — one MMCV project, two subsets

This shared **centerline** project loads either RATopo checkpoint using `--subset A` or `--subset B`. Both use the same MMCV environment and common command-line tools. The [rebuilt PyTorch DualTopo lane-segment project](../dualtopo/) has a different environment, model and annotation schema.

| Selection | Data | Geometry | Full validation frames | Weights / results |
| --- | --- | --- | ---: | --- |
| `--subset A` | `data_dict_subset_A_val.pkl`, native centerline format | Seven cameras, 11-point **XYZ** centerlines | 4,806 | [HF](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_a) / [results](backends/subset_a/results.json) |
| `--subset B` | `data_dict_subset_B_val.pkl`, native centerline format | Six cameras, 11-point **XY** centerlines | 6,019 | [HF](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_b) / [results](backends/subset_b/results.json) |

The shared entry selects the corresponding native geometry backend, config and checksum manifest. The weights are not interchangeable. `configs/subset_a.py` and `configs/subset_b.py` retain the accepted full trained model, including point positional encoding, attention and temporal modules.

## Installation — separate from PyTorch DualTopo

```bash
conda create -n dualtopo-ratopo python=3.8 -y
conda activate dualtopo-ratopo
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/ratopo
pip install -r requirements.txt
python tools/verify_release.py
```

MMDetection3D may compile extensions; use a compiler/toolkit compatible with PyTorch 1.9/CUDA 11.1. The legacy native scorer's ORTools/protobuf versions are pinned. Do not install this stack into the other project's PyTorch 2.x environment. The runtime stack was validated in an existing environment; a fresh-machine installation is not implied by a source-integrity check.

## Prepare data

Follow [RATopo's native dataset preparation](https://github.com/buaa-colalab/RATopo#prepare-dataset) and [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2). Use the matching native **centerline** pickle; the LaneSegNet `*_lanesegnet.pkl` used by the other project is incompatible. `--data-root` must resolve the `image_path` values stored inside each metadata frame, usually the directory containing `val/`.

```bash
python tools/check_data.py --subset A --data-root /path/to/subset_a_image_root \
  --annotations /path/to/data_dict_subset_A_val.pkl --check-images
python tools/check_data.py --subset B --data-root /path/to/subset_b_image_root \
  --annotations /path/to/data_dict_subset_B_val.pkl --check-images
```

The inference/scoring entry additionally verifies the published annotation checksum. `--allow-annotation-mismatch` is available in individual commands for another data encoding/version; it does not establish equivalence to the reported benchmark and is not used by the strict reproduction workflow.

## Complete reproduction

```bash
python tools/reproduce.py --subset A --data-root /path/to/subset_a_image_root \
  --annotations /path/to/data_dict_subset_A_val.pkl --output-dir outputs/subset_a

python tools/reproduce.py --subset B --data-root /path/to/subset_b_image_root \
  --annotations /path/to/data_dict_subset_B_val.pkl --output-dir outputs/subset_b
```

Run sequentially. For each subset the workflow verifies source/data/weights, downloads the selected safetensors if absent, runs **every validation frame** in `singleframe` and `temporal`, then executes native official CPU scoring and writes `reproduction.json`. Matching completed outputs are verified and reused. Saved prediction or metric files are never silently overwritten. One process/GPU job runs at a time.

Individual steps share one CLI:

```bash
python tools/download_weights.py --subset A
python tools/download_weights.py --subset B
python tools/verify_release.py --subset A --weights weights/ratopo_subset_a.safetensors
python tools/verify_release.py --subset B --weights weights/ratopo_subset_b.safetensors

python tools/infer.py --subset A --branch temporal --data-root /path/to/subset_a_image_root \
  --annotations /path/to/data_dict_subset_A_val.pkl --output outputs/a_temporal.pkl
python tools/evaluate.py --subset A --predictions outputs/a_temporal.pkl \
  --data-root /path/to/subset_a_image_root --annotations /path/to/data_dict_subset_A_val.pkl \
  --output outputs/a_temporal_metrics.json
```

For B, change **all** of subset, data root, annotation and output paths. The selected default weight path is `weights/ratopo_subset_a.safetensors` or `weights/ratopo_subset_b.safetensors`; `--weights` can override it. `singleframe` clears history every frame; `temporal` uses actual chronological history and resets at scene changes. Neither branch is equivalent to the rebuilt project's history-updating `init` pathway.

## Complete accepted centerline results

Percent metrics from the original complete accepted evaluations:

| Subset / branch | OLS | DET_l | DET_t | TOP_ll | TOP_lt |
| --- | ---: | ---: | ---: | ---: | ---: |
| A singleframe | 48.6484 | 31.2227 | 49.7748 | 31.8840 | 32.6385 |
| A temporal | 50.1364 | 33.4838 | 49.7748 | 34.7070 | 34.0756 |
| B singleframe | 53.6457 | 39.5619 | 56.7350 | 42.8199 | 27.9303 |
| B temporal | 54.9715 | 41.4766 | 56.7350 | 45.8780 | 29.0965 |

**SubsetA protocol:** five three-point `tl_99` traffic-element polygons are normalized to their native min/max enclosing bbox only in a GT scoring copy; the other 17,642 boxes, topology, predictions and original annotations are unchanged. All 4,806 frames are retained. **SubsetB:** all 22,916 TE boxes already have two points, so no such normalization is applied; all 6,019 frames are retained.

Both safe exports preserve all 2,677 model tensors exactly and omit optimizer/runtime pickle objects. The released inference sources passed exact input/BEV/decoded-output parity on 24 real frames per branch across three scenes, with actual history/reset checks; the relocation/shared entry is regression-checked against those same saved fixtures. These bounded checks are not a new complete-split GPU benchmark.

SubsetB's original Stage1 supervisor lost its four optimization exit receipts. Later CPU finalization performed no optimization and did not recover those GPU waits. Stage2 and full evaluation retain their actual accepted receipts. The historical limitation remains explicit in `backends/subset_b/results.json`; evaluation acceptance does not imply recovered training evidence.

Code is inference/evaluation only. Native definitions required by OpenMMLab construction are retained, with no training launcher. See `LICENSE`, `NOTICE`, per-subset manifests/validation records and the source checksum manifest. Load only trusted annotation/prediction pickle files.
