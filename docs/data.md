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

## Image pipeline

1. Load seven images as float32 BGR.
2. Crop the front-center image with the same vertical crop as the accepted configuration; update its projection matrix.
3. Resize all camera images with fixed image scale **0.5**; update calibration.
4. Convert BGR to RGB and normalize using mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`.
5. Pad the bottom/right of every camera image to the common maximum shape, rounded up to a multiple of 32.

The name **scale025** denotes the **directed graph residual coefficient 0.25**, not the image resize ratio. Images use 0.5. Projection visibility uses each camera's valid resized `img_shape`; sampling normalization uses its actual padded `pad_shape`. Do not replace one with the other.

Prediction formatting keeps the centerline and both boundaries in metric XYZ. It preserves confidence ordering and aligns topology rows/columns with the same lane ordering after pedestrian-crossing entries are removed. This release uses only SubsetA's 3D lane segments, with 90 regression values per candidate.
