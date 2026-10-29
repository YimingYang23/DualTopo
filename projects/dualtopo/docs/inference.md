# Inference and evaluation

Run these commands from `DualTopo/projects/dualtopo` after [installation](installation.md), [data preparation](data.md) and [weight download](weights.md).

## Complete evaluation

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

This evaluates all 4,806 validation frames through both pathways. The output directory contains `init.pkl`, `temporal.pkl` and their corresponding `*_metrics.json` files. Use `--branches temporal` to evaluate the temporal pathway alone.

## Inference only

```bash
python tools/infer.py \
  --checkpoint weights/dualtopo_remake_scale025_subset_a.pth \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --branch temporal \
  --output outputs/subset_a/temporal.pkl
```

`--branch temporal` selects predictions from the history-conditioned pathway. `--branch init` selects initial-query predictions. Both pathways update temporal memory; scene changes reset history. Frames are processed in scene and timestamp order.

## Evaluate saved predictions

```bash
python tools/evaluate.py \
  --predictions outputs/subset_a/temporal.pkl \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output outputs/subset_a/temporal_metrics.json
```

The evaluator expects the complete 4,806-frame split and writes `mAP`, `AP_ls`, `AP_ped`, `TOP_lsls` and `mAP_type` to the specified JSON file.

Prediction pickles use the LaneSegNet `results[(split, segment_id, timestamp)]['predictions']` format. Entries contain `lane_segment`, `area`, `traffic_element`, `topology_lsls` and `topology_lste`. Traffic-element predictions are empty for this lane-segment checkpoint.

The metric file paths are listed in `reproduction.json`.
