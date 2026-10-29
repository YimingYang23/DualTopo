**Start here — two independent projects, three checkpoints:** [PyTorch DualTopo · SubsetA lane segments](projects/dualtopo/) | [MMCV RATopo · SubsetA / SubsetB centerlines](projects/ratopo/) | [All three weights](https://huggingface.co/y352184741/DualTopo)

<div align="center">

<h1>DualTopo</h1>

<h3>Spatio-Temporal Lane Segment Graph Generation with Dual Latent World Models and Parallel Query Supervision</h3>

<p>
  <a href="https://yimingyang23.github.io/">Yiming Yang</a><sup>1</sup>&emsp;
  <a href="https://scholar.google.com/citations?user=LqX1k5QAAAAJ&amp;hl=zh-CN&amp;oi=ao">Hongbin Lin</a><sup>1</sup>&emsp;
  <a href="https://scholar.google.com/citations?user=B588EyYAAAAJ&amp;hl=zh-CN&amp;oi=ao">Yueru Luo</a><sup>1</sup>&emsp;
  <a href="https://scholar.google.com/citations?user=jwPyH6wAAAAJ&amp;hl=zh-CN">Suzhong Fu</a><sup>1</sup>&emsp;
  <a href="https://scholar.google.com/citations?user=6A1yEFMAAAAJ&amp;hl=zh-CN&amp;oi=ao">Chao Zheng</a><sup>3</sup><br>
  Xinrui Yan<sup>3</sup>&emsp;
  Kun Tang<sup>3</sup>&emsp;
  <a href="https://lvchen.wixsite.com/automan">Chen Lv</a><sup>2</sup>&emsp;
  <a href="https://sse.cuhk.edu.cn/en/faculty/cuishuguang">Shuguang Cui</a><sup>1</sup>&emsp;
  <a href="https://sse.cuhk.edu.cn/faculty/lizhen">Zhen Li</a><sup>1</sup>
</p>

<p>
  <sup>1</sup>The Chinese University of Hong Kong, Shenzhen<br>
  <sup>2</sup>Nanyang Technological University&emsp;
  <sup>3</sup>Tencent Maps (T Lab)
</p>

<p>
  <a href="https://www.cuhk.edu.cn/"><img src="assets/logos/cuhk_shenzhen.png" width="38%" alt="The Chinese University of Hong Kong, Shenzhen"></a>&ensp;
  <a href="https://www.ntu.edu.sg/"><img src="assets/logos/ntu.png" width="28%" alt="Nanyang Technological University"></a>&ensp;
  <a href="https://map.qq.com/"><img src="assets/logos/tencent_maps.png" width="24%" alt="Tencent Maps"></a>
</p>

<h4>
  <a href="https://github.com/YimingYang23/DualTopo"><img src="https://img.shields.io/badge/GitHub-Inference%20Code-181717?logo=github" alt="Inference code"></a>
  <a href="https://huggingface.co/y352184741/DualTopo"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoint-FFD21E?logo=huggingface" alt="Checkpoint"></a>
  <a href="#three-checkpoints-two-independent-projects"><img src="https://img.shields.io/badge/Docs-Installation-2ea44f" alt="Installation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</h4>

<img src="assets/framework.png" width="100%" alt="Overall framework of DualTopo">

</div>

## Three checkpoints, two independent projects

This repository contains **two implementations with separate environments and data protocols**. Choose the project and checkpoint together.

| Implementation | Task / validation data | Environment | Checkpoint |
| --- | --- | --- | --- |
| [DualTopo — rebuilt PyTorch implementation](projects/dualtopo/) | **Lane segment**, OpenLane-V2 SubsetA; `data_dict_subset_A_val_lanesegnet.pkl` | Python 3.11+, PyTorch 2.x, Transformers; **no MMCV** | [DualTopo SubsetA](https://huggingface.co/y352184741/DualTopo/blob/main/dualtopo_remake_scale025_subset_a.pth) |
| [DualTopo on RATopo — shared MMCV project](projects/ratopo/) | **3D centerline**, SubsetA; `data_dict_subset_A_val.pkl` | Python 3.8, PyTorch 1.9/CUDA 11.1, MMCV-full 1.5.2 | [RATopo SubsetA](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_a) |
| [DualTopo on RATopo — same project, another config](projects/ratopo/) | **2D centerline**, SubsetB; `data_dict_subset_B_val.pkl` | The same RATopo MMCV environment | [RATopo SubsetB](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_b) |

The two RATopo checkpoints share one installation and one CLI: select `--subset A` or `--subset B`. The PyTorch DualTopo project has its own model, environment, LaneSegNet-format data loader and lane-segment evaluator. A centerline pickle cannot replace a lane-segment pickle, even when both are called “SubsetA”.

```text
projects/
├── dualtopo/              # rebuilt PyTorch DualTopo, lane-segment task
│   ├── dualtopo/          # model implementation
│   ├── configs/
│   ├── docs/
│   └── tools/             # download / check data / infer / evaluate / reproduce
└── ratopo/                # one MMCV project for both RATopo checkpoints
    ├── configs/
    │   ├── subset_a.py    # seven cameras, 11-point XYZ centerlines
    │   └── subset_b.py    # six cameras, 11-point XY centerlines
    ├── backends/          # validated native geometry implementations
    └── tools/             # common CLI; always select --subset A or B
```

## Reproduce a complete validation split

Install **one project in its own environment**, following its README. Each project supplies a full workflow covering data checks, checksum-verified weight download, sequential inference, native CPU scoring and a saved receipt. Completed matching predictions/metrics are verified and reused rather than overwritten. The default workflow has no small-sample limit.

### Rebuilt PyTorch DualTopo: SubsetA lane segments

Follow [its installation and dataset instructions](projects/dualtopo/README.md), then:

```bash
cd projects/dualtopo
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

This runs all 4,806 frames for `init` and `temporal`. The **init pathway still updates temporal memory**; it is distinct from RATopo's cold single-frame branch. This lane-only checkpoint has no traffic-element detector.

### MMCV RATopo: SubsetA or SubsetB centerlines

Follow the [shared RATopo installation](projects/ratopo/README.md), then choose the matching checkpoint/config/data:

```bash
cd projects/ratopo
python tools/reproduce.py --subset A \
  --data-root /path/to/subset_a_image_root \
  --annotations /path/to/data_dict_subset_A_val.pkl \
  --output-dir outputs/subset_a

python tools/reproduce.py --subset B \
  --data-root /path/to/subset_b_image_root \
  --annotations /path/to/data_dict_subset_B_val.pkl \
  --output-dir outputs/subset_b
```

Each runs `singleframe` and `temporal`, with 4,806 frames per branch for A or 6,019 for B. `singleframe` clears temporal history every frame. Use one GPU job at a time and keep frames in chronological scene order.

## Results and verification scope

Full RATopo centerline metrics and protocol details are in [SubsetA results](projects/ratopo/backends/subset_a/results.json) and [SubsetB results](projects/ratopo/backends/subset_b/results.json). SubsetA's five native GT bbox normalizations are specific to its centerline scoring protocol. SubsetB's historical Stage1 exit-receipt limitation remains disclosed.

The rebuilt PyTorch project's [validation record](projects/dualtopo/validation.json) covers tensor-exact loading and bounded first/history/reset fixtures. The RATopo projects' [A](projects/ratopo/backends/subset_a/validation.json) / [B](projects/ratopo/backends/subset_b/validation.json) records cover tensor-exact safe exports and real-frame numerical parity. These records distinguish original complete evaluation results from bounded public-code checks. Complete reproduction commands are provided; a new complete public-code benchmark is not claimed merely from a smoke test.

## Existing release compatibility

All three weight files retain their existing HF URLs and bytes. The previous layout is preserved at tag [`legacy-layout-20261029`](https://github.com/YimingYang23/DualTopo/tree/legacy-layout-20261029). Root-level launchers/imports remain compatibility entry points for the PyTorch DualTopo project; new users should work inside the selected project directory. The earlier separate RATopo A/B directory layout remains available in that tag.

## License and attribution

Code is distributed under [Apache-2.0](LICENSE), retaining third-party notices in each project. Images and annotations are not distributed; obtain them through [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) and follow the appropriate [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet) or [RATopo](https://github.com/buaa-colalab/RATopo) preparation protocol.
