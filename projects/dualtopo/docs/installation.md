# Installation

Use Linux and an NVIDIA driver compatible with the PyTorch CUDA 12.4 wheels.

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install -e . --no-deps
```

## Check the environment

```bash
python -m pip check
python -c "import torch, torchvision, transformers, dualtopo; print(torch.__version__, torchvision.__version__, transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` should be `True`. Inference uses FP32 and one frame per batch.

Next: [prepare data](data.md), then [run evaluation](inference.md).
