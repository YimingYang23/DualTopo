# Installation

Use a separate Python environment with **PyTorch 2.6.0 + CUDA 12.4**, as shown below.

```bash
conda create -n dualtopo python=3.11 -y
conda activate dualtopo
git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

No MMCV, MMDetection, MMDetection3D, MMEngine or custom CUDA-extension compilation is required. Multi-scale deformable attention uses the PyTorch implementation provided by `transformers==4.57.3`. Install the CUDA wheel matching your NVIDIA driver; a separate locally compiled CUDA toolkit is not needed for these wheel-based operations.

Check the environment:

```bash
python -c "import torch, torchvision, transformers, dualtopo; print(torch.__version__, torchvision.__version__, transformers.__version__); print(torch.cuda.is_available())"
python tools/infer.py --help
```

For CPU-only compatibility work, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Full sequential inference is intended for an NVIDIA GPU. CPU full-dataset inference is supported by the tensor operations but will be slow. Inference uses float32 and batch size 1. CUDA memory use depends on the padded camera-image sizes and device; this release does not assert a measured minimum VRAM for the polished entry point. If needed, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching. Do not use mixed precision or `torch.compile` when comparing with the accepted FP32 run unless separately validated.

The OpenLane-V2 devkit is only needed for annotation conversion or external scoring, not for this repository's forward pass. Follow the upstream devkit installation from [LaneSegNet's dataset instructions](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset). Keep its metric version fixed when comparing results: upstream notes that OpenLane-V2 v2.1.0 changed the TOP metric.
