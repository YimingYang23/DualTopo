"""Sequential SubsetA inference with detached, scene-aware DualTopo memories."""

import argparse
import copy
import json
import pickle
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dualtopo import DualTopo
from dualtopo.checkpoint import load_checkpoint, sha256
from dualtopo.dataset import OpenLaneV2SubsetALaneSegNetDataset, collate_fn

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/dualtopo_remake_scale025_subset_a.json"
)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("init", "temporal"), default="temporal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Debug prefix only; omit for full validation")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = json.loads(args.config.read_text(encoding="utf8"))
    data_config = copy.deepcopy(config["data"])
    data_config.update(data_root=str(args.data_root), ann_file=str(args.annotations))
    dataset = OpenLaneV2SubsetALaneSegNetDataset(**data_config)
    if args.limit is not None:
        dataset.data_infos = dataset.data_infos[: args.limit]
        dataset._set_sequence_info()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )
    model = DualTopo(config)
    metadata = load_checkpoint(model, args.checkpoint, args.checkpoint_sha256)
    model.lane_head.eval_output = "singleframe" if args.branch == "init" else "temporal"
    model.to(args.device).eval()
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=args.branch):
            # Do not bypass forward_test: it applies native can_bus delta semantics.
            outputs.extend(model.forward_test(**move_to_device(batch, args.device)))
    predictions = dataset.format_results(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = dict(
        frames=len(outputs),
        branch=args.branch,
        subset="A",
        checkpoint_metadata=metadata,
        checkpoint_sha256=sha256(args.checkpoint),
        config_sha256=sha256(args.config),
        annotation_sha256=sha256(args.annotations),
        predictions_sha256=sha256(args.output),
        torch_version=torch.__version__,
    )
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(f"Saved {len(outputs)} frames to {args.output}")


if __name__ == "__main__":
    main()
