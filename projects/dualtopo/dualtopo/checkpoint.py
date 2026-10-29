"""Load inference-only checkpoints and map legacy Remake parameter names."""

import hashlib
from pathlib import Path
import torch


def public_parameter_name(name):
    """Rename keys only; this never changes, drops or approximates tensors."""
    name = name.replace("._query_wm_decoder.", ".query_world_model.")
    name = name.replace(".wm_bev_constructor.", ".bev_world_model.")
    name = name.replace(".topology_aggregation.", ".directed_graph_propagation.")
    if ".directed_graph_propagation." in name:
        name = name.replace(".pre_proj.", ".predecessor_projection.")
        name = name.replace(".succ_proj.", ".successor_projection.")
        name = name.replace(".fusion.", ".graph_fusion.")
    return name


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_checkpoint(model, path, expected_sha256=None):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    # Official release contains tensors and primitive metadata only.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    renamed = {public_parameter_name(key): value for key, value in state.items()}
    if len(renamed) != len(state):
        raise ValueError("Duplicate parameter names after checkpoint conversion")
    model.load_state_dict(renamed, strict=True)
    return checkpoint.get("metadata", {})
