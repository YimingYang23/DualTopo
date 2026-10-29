# Run inference and evaluation

First complete [installation](installation.md) and [data preparation](data.md). Run the following commands from `DualTopo/projects/dualtopo`.

## Check the setup

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --limit 3 --output-dir outputs/check_subset_a
```

This runs inference and scoring on three frames for both branches.

## Evaluate the dataset

```bash
python tools/reproduce.py \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output-dir outputs/subset_a
```

This downloads the weights and evaluates the 4,806 validation frames for `init` and `temporal`. Add `--branches temporal` to run only `temporal`, or `--branches init` to run only `init`.

Predictions are saved as `init.pkl` and `temporal.pkl`. Metrics are saved as JSON files in the output directory; their paths are listed in `reproduction.json`.

## Generate predictions only

Download the weights with the [weight download command](weights.md), then run:

```bash
python tools/infer.py \
  --checkpoint weights/dualtopo_remake_scale025_subset_a.pth \
  --data-root /path/to/image_root \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --branch temporal \
  --output outputs/subset_a/temporal.pkl
```

`--branch` accepts `init` or `temporal`. Use `--device cuda:1` to select another GPU.

## Score a prediction file

```bash
python tools/evaluate.py \
  --predictions outputs/subset_a/temporal.pkl \
  --annotations /path/to/data_dict_subset_A_val_lanesegnet.pkl \
  --output outputs/subset_a/temporal_metrics.json
```

The prediction file must cover all 4,806 validation frames. Metric values are percentages. The output JSON contains `mAP`, `AP_ls`, `AP_ped`, `TOP_lsls` and `mAP_type`.
