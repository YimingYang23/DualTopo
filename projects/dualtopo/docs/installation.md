# Installation

Use Python 3.11 and a CUDA-enabled PyTorch 2.6.0 environment for GPU inference.

```bash
conda create -n dualtopo-pytorch python=3.11 -y
conda activate dualtopo-pytorch

git clone https://github.com/YimingYang23/DualTopo.git
cd DualTopo/projects/dualtopo

python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -r requirements-eval.txt
python -m pip install -e . --no-deps
```

The CUDA 12.4 wheels require a compatible NVIDIA driver. Multi-scale deformable attention uses Transformers 4.57.3; this project runs directly in PyTorch.

## Check the environment

```bash
python -c "import torch, torchvision, transformers, dualtopo; print(torch.__version__, torchvision.__version__, transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
python tools/reproduce.py --help
```

`CUDA available` should be `True` for GPU inference. The standard evaluation uses FP32 and one frame per batch.

Next: [prepare the dataset](data.md), then run [evaluation](../README.md#evaluation). Weights are downloaded automatically; see [manual download](weights.md) to use a local checkpoint.
