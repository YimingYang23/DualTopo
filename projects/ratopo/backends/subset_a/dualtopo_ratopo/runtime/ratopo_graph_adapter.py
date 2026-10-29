"""DualTopo RATopo inference components."""
import torch


def refresh_ratopo_graphs(lane_features, lane_world_coordinates, te_features,
                         img_metas, lclc_branches, lcte_branches,
                         producing_layer_index, previous_lclc, previous_lcte):
    if lane_features.ndim != 3:
        raise ValueError('Expected native lane features [B,N,C]')
    b, n, _ = lane_features.shape
    if lane_world_coordinates.shape != (b, n, 33):
        raise ValueError('Keep all native 11 three-dimensional centerline points')
    if te_features.ndim != 3 or te_features.shape[0] != b:
        raise ValueError('Expected native TE features [B,T,C]')
    if previous_lclc.ndim != 5 or previous_lcte.ndim != 5:
        raise ValueError('Expected native inquiry/batch/graph-channel axes')
    heads = previous_lclc.shape[0]
    if previous_lclc.shape != (heads, b, 2, n, n):
        raise ValueError('Only audited native main plus auxiliary graph supported')
    if previous_lcte.shape != (heads, b, 2, n, te_features.shape[1]):
        raise ValueError('Lane/TE graph axes do not match the native node order')
    if not (0 <= producing_layer_index < len(lclc_branches)
            and producing_layer_index < len(lcte_branches)):
        raise ValueError('Original producing head index is required')
    ll_logits, ll_graph = lclc_branches[producing_layer_index](
        lane_features, lane_features, lane_world_coordinates,
        lane_world_coordinates)
    lt_logits = lcte_branches[producing_layer_index](
        lane_features, te_features, img_metas)
    if ll_logits.shape != (b, n, n, 1) or ll_graph.shape != ll_logits.shape:
        raise ValueError('Native lane-lane head returned incompatible axes')
    if lt_logits.shape != (b, n, te_features.shape[1], 1):
        raise ValueError('Native lane-TE head returned incompatible axes')
    # Keep the native LL head's own geometric/feature detach convention;
    # the decoder retains its graph tensor and detaches LT adjacency only.
    main_ll = ll_graph.clone().squeeze(-1).unsqueeze(0).unsqueeze(2)
    main_lt = lt_logits.clone().detach().squeeze(-1).sigmoid().unsqueeze(0).unsqueeze(2)
    return dict(
        lclc_adj=torch.cat((main_ll.expand(heads, -1, -1, -1, -1),
                            previous_lclc[:, :, 1:2]), dim=2),
        lcte_adj=torch.cat((main_lt.expand(heads, -1, -1, -1, -1),
                            previous_lcte[:, :, 1:2]), dim=2),
        lclc_logits=ll_logits, lcte_logits=lt_logits,
        producing_layer_index=producing_layer_index)

