**Test code:** [DualTopo](projects/dualtopo/) · [RATopo](projects/ratopo/) · [Weights](https://huggingface.co/y352184741/DualTopo)

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
  <a href="#projects"><img src="https://img.shields.io/badge/Docs-Installation-2ea44f" alt="Installation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</h4>

<img src="assets/framework.png" width="100%" alt="Overall framework of DualTopo">

</div>

## Projects

Test code for two implementations and three checkpoints.

| Project | Environment | Dataset / task | Weights |
| --- | --- | --- | --- |
| [DualTopo](projects/dualtopo/) | Python 3.11+, PyTorch 2.x | SubsetA lane segments, LaneSegNet format | [SubsetA](https://huggingface.co/y352184741/DualTopo/blob/main/dualtopo_remake_scale025_subset_a.pth) |
| [RATopo](projects/ratopo/) | Python 3.8, PyTorch 1.9, MMCV 1.5.2 | SubsetA 3D centerlines | [SubsetA](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_a) |
| [RATopo](projects/ratopo/) | Same RATopo environment | SubsetB 2D centerlines | [SubsetB](https://huggingface.co/y352184741/DualTopo/tree/main/ratopo_subset_b) |

RATopo uses one project with `--subset A` or `--subset B`.

## Installation and testing

- **[DualTopo instructions](projects/dualtopo/README.md):** environment, LaneSegNet data preparation and testing.
- **[RATopo instructions](projects/ratopo/README.md):** shared MMCV environment, SubsetA/B data and testing.
- **[All weights on Hugging Face](https://huggingface.co/y352184741/DualTopo)**

Each project provides `tools/reproduce.py` for weight download, inference and official evaluation.

## License

[Apache-2.0](LICENSE) · [Attribution](NOTICE) · [Previous release](https://github.com/YimingYang23/DualTopo/tree/legacy-layout-20261029)
