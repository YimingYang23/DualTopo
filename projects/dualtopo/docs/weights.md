# Download weights

Run these commands from `DualTopo/projects/dualtopo` after [installation](installation.md).

## Automatic download

The evaluation command in the [project README](../README.md#evaluation) downloads the SubsetA checkpoint automatically.

## Manual download

```bash
python tools/download_weights.py --directory weights
```

This saves `weights/dualtopo_remake_scale025_subset_a.pth` and checks the downloaded file against the checkpoint manifest.

You can also [download the checkpoint directly](https://huggingface.co/y352184741/DualTopo/resolve/main/dualtopo_remake_scale025_subset_a.pth). Place it in the same `weights/` directory, or pass its location with `--checkpoint`:

```bash
python tools/reproduce.py \
  --checkpoint /path/to/dualtopo_remake_scale025_subset_a.pth \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```
