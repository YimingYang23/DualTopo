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
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/Docs-Installation-2ea44f" alt="Installation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</h4>

<img src="assets/framework.png" width="100%" alt="Overall framework of DualTopo">

</div>

## TODO

- [ ] Train code
- [x] Checkpoint
- [x] Inference code

## Visualizations

Surround-camera views, temporal lane tracks in the current ego frame, and fused topology in the global BEV.

![DualTopo example 1](assets/visualizations/scene_01.gif)

![DualTopo example 2](assets/visualizations/scene_02.gif)

![DualTopo example 3](assets/visualizations/scene_03.gif)

## Usage

### Installation

Follow the [installation instructions](docs/installation.md). Dependencies are listed in [requirements.txt](requirements.txt).

### Prepare Dataset

Follow [SubsetA data preparation](docs/data.md), using the LaneSegNet preprocessing format for OpenLane-V2.

### Inference

The checkpoint is available on [Hugging Face](https://huggingface.co/y352184741/DualTopo). Download it and run:

```bash
python tools/download_weights.py --directory weights

python tools/infer.py \
  --checkpoint weights/dualtopo_remake_scale025_subset_a.pth \
  --data-root data \
  --annotations data/OpenLane-V2/data_dict_subset_A_val_lanesegnet.pkl \
  --output outputs/subset_a_temporal.pkl \
  --branch temporal
```

Use `--branch init` for the init pathway. Both pathways use the same checkpoint. Keep frames sequential to preserve temporal state. See [inference details](docs/inference.md) and [checkpoint details](docs/weights.md).

## Main Results

Results reported in the paper on OpenLane-V2.

<table>
  <tr>
    <th>SubsetA</th>
    <th>SubsetB</th>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top"><img src="assets/results/subset_a.png" width="100%" alt="OpenLane-V2 SubsetA main results"></td>
    <td width="50%" align="center" valign="top"><img src="assets/results/subset_b.png" width="100%" alt="OpenLane-V2 SubsetB main results"></td>
  </tr>
</table>

## License

This project is licensed under [Apache-2.0](LICENSE).

## Acknowledgement

We thank the authors of [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet), [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2), and [BEVFormer](https://github.com/fundamentalvision/BEVFormer). Third-party attribution is retained in [NOTICE](NOTICE).
