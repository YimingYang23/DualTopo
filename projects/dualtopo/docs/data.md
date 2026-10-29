# Prepare SubsetA data

## Download and convert

Download the dataset from [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2), then follow [LaneSegNet's data preparation instructions](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset) to generate `data_dict_subset_A_val_lanesegnet.pkl`.

In the LaneSegNet checkout, point `data/OpenLane-V2` at the downloaded dataset and run:

```bash
python tools/data_process.py
```

The validation files should be arranged as follows:

```text
data/OpenLane-V2/
├── data_dict_subset_A_val_lanesegnet.pkl
└── val/
    └── ...
```

## Set the input paths

Pass the generated pickle file to `--annotations`. Choose `--data-root` according to the image paths stored in that file:

| Image path in the annotations | `--data-root` |
| --- | --- |
| `OpenLane-V2/val/...` | `/path/to/data` |
| `val/...` | `/path/to/data/OpenLane-V2` |

## Check the files

From `DualTopo/projects/dualtopo`, run:

```bash
python tools/check_data.py \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --data-root /path/to/image_root \
  --check-images
```

After the check succeeds, proceed to [inference and evaluation](inference.md).
