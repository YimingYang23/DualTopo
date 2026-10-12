# Inference semantics

## QWM, BWM and shared pathways

`DualTopo.forward_test` constructs current BEV features, predicts history-conditioned query and BEV latents, fuses BEV features and runs the shared Dual-Pathway Decoder. Relative ego rotation and translation form the 12-dimensional action latent used by QWM/BWM.

The init pathway starts from current-frame queries. After the first decoder layer, the temporal pathway uses retained query history and current-frame candidates. Both pathways share learned layers and heads. `--branch init` selects init outputs, while the temporal memories continue to update from the temporal pathway. It is not a cold-start-per-frame ablation.

Call `forward_test`, not `simple_test` directly: the entry handles `can_bus[:3]` and yaw increments according to the native BEV interface. With no previous BEV, these increments are zeroed. Full `lidar2global_rotation` and translation are retained for QWM/BWM; do not zero those poses. The released configuration preserves the accepted `video_test_mode` behavior rather than enabling a different upstream BEV history path.

## Directed topology memory

The previous frame's adjacency is aligned with exactly the same ordered Top-K queries. `A[i,j]` represents a directed edge from i to j. Only lane-class nodes are valid; invalid edges and self-edges are excluded from propagation. Successor messages use row-normalized A and predecessor messages use row-normalized Aᵀ, with denominator floor `1e-6`.

The graph fusion residual is confidence-gated and scaled by **0.25**, matching the paper's directed graph propagation definition. Cached query/BEV/adjacency tensors are detached. Scene changes clear the corresponding history. The raw learned adjacency is cached before any output-only postprocessing. No geometric distance score is fed into the memories.

## Checkpoint names

Public module and parameter names follow the manuscript: `query_world_model`, `bev_world_model`, `directed_graph_propagation`, `predecessor_projection`, `successor_projection`, `graph_fusion`. `dualtopo/checkpoint.py` explicitly maps the corresponding legacy Remake names when loading. The exported checkpoint already uses the public names; no tensor values change.

A few unused learned embeddings from the accepted checkpoint are retained in the model layout to preserve strict loading of all 1,023 tensors. They do not introduce a training entry point, denoising procedure or extra inference supervision. Public `forward` performs inference only.

## Prediction contract

The output has the LaneSegNet `results[(split, segment_id, timestamp)]['predictions']` structure. It contains `lane_segment`, `area`, `traffic_element`, `topology_lsls` and `topology_lste`. Traffic output is empty for this lane-only checkpoint. Use the official devkit for scoring outside this inference-only repository and record its version.
