# OpenLane-V2 SubsetA data

Obtain the images and lane-segment annotations through [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2). Use [LaneSegNet's preparation instructions](https://github.com/OpenDriveLab/LaneSegNet#prepare-dataset) and its official [tools/data_process.py](https://github.com/OpenDriveLab/LaneSegNet/blob/main/tools/data_process.py) to generate `data_dict_subset_A_val_lanesegnet.pkl`.

The official converter reads SubsetA's `*-ls.json` frames, converts camera calibration and ego poses to float64 arrays, and resamples each lane centerline/left/right boundary to **10 points**. It preserves the original area polygon points. Use the `*_lanesegnet.pkl` convention, not the separate Map Element Bucket `*_ls.pkl` convention.

Example arrangement:

```text
data/
└── OpenLane-V2/
    ├── data_dict_subset_A.json
    ├── data_dict_subset_A_val_lanesegnet.pkl
    ├── train/
    ├── val/
    └── test/
```

In the LaneSegNet checkout, point `data/OpenLane-V2` at the downloaded dataset and run:

```bash
python tools/data_process.py
```

The converter writes the pickle inside `data/OpenLane-V2`. You can use that file directly with `--annotations`; it need not be moved. The image root passed as `--data-root` must satisfy `data_root / sensor[camera]['image_path']` for every camera. For metadata containing `OpenLane-V2/val/...`, pass `--data-root data`; if the metadata instead contains `val/...`, pass `--data-root data/OpenLane-V2`. Check this prefix in your own converted metadata rather than adding it twice.

## Required inference metadata

Each frame provides `segment_id`, `timestamp`, `pose.rotation`, `pose.translation`, and the seven-camera `sensor` dictionary containing `image_path`, intrinsic `K`, and extrinsic rotation/translation. Keep the official camera insertion order. Ego rotation is a 3×3 matrix and translation is an XYZ vector. Annotation targets and future ego poses are not consumed by the public inference loader.

Scene frames must be contiguous and strictly ordered by timestamp. The loader rejects interleaved scenes or reversed timestamps. It assigns sequential sample indices and resets temporal memory at scene changes.
