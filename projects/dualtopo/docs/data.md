# Prepare SubsetA data

## Download

Download [OpenLane-V2 SubsetA](https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md#download). Use the **Map Element Bucket info** and **Image (val)** downloads for SubsetA, plus the split index. Extract them into:

```text
OpenLane-V2/
├── data_dict_subset_A.json
└── val/
    └── <segment_id>/
        ├── info/<timestamp>-ls.json
        └── ... camera images ...
```

## Convert annotations

From `DualTopo/projects/dualtopo`:

```bash
python tools/prepare_data.py \
  --data-root /path/to/OpenLane-V2 \
  --output data/data_dict_subset_A_val_lanesegnet.pkl
```

The converter prepares the 4,806 validation frames in LaneSegNet format. To use an index stored elsewhere, add `--index /path/to/data_dict_subset_A.json`.

## Set the image root

The converter's `--data-root` is the dataset directory above. For inference, choose `--data-root` according to `sensor[...]['image_path']` in the generated pickle:

| Image path in the pickle | Inference `--data-root` |
| --- | --- |
| `OpenLane-V2/val/...` | `/path/to` |
| `val/...` | `/path/to/OpenLane-V2` |

Check the complete image set:

```bash
python tools/check_data.py \
  --annotations data/data_dict_subset_A_val_lanesegnet.pkl \
  --data-root /path/to/image_root --check-images
```

Next: [run evaluation](inference.md).
