> Legacy PyTorch DualTopo documentation. Current project: [projects/dualtopo](../projects/dualtopo/). The separate MMCV RATopo project is [projects/ratopo](../projects/ratopo/).

# SubsetA checkpoint distribution

The official target is [y352184741/DualTopo](https://huggingface.co/y352184741/DualTopo). The prepared file is `dualtopo_remake_scale025_subset_a.pth`; its bytes and SHA256 are listed in `configs/weights.json`. It contains only `state_dict` tensors and portable `metadata`, and is loadable with `torch.load(..., weights_only=True)`.

The original training checkpoint was exported without changing tensor values. Optimizer states, training scheduler/RNG objects and private machine paths are not distributed. No Stage1 checkpoint or SubsetB weight is needed for inference with this file.

## Maintainer upload

Use a Hugging Face account with write access to `y352184741/DualTopo`. Authenticate locally; never put a token in source code or a command committed to Git.

```bash
python -m pip install "huggingface_hub>=0.34,<1"
hf auth login
hf upload y352184741/DualTopo /path/to/huggingface_subset_a . \
  --repo-type model --commit-message "Release DualTopo SubsetA inference checkpoint"
```

The prepared folder should contain only the `.pth` file, `README.md` model card, `config.json`, `weights.json`, `LICENSE` and `NOTICE`. The existing remote `.gitattributes` can remain. Uploading through `huggingface_hub` handles large-file storage; do not commit the model binary to the code repository.

After upload, download and verify with:

```bash
python tools/download_weights.py --directory weights
```

For reproducible use, pass a fixed Hugging Face commit hash through `--revision`. The current release manifest checks the checkpoint SHA256 even when using `main`. See the [official upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload) for authentication and upload options.
