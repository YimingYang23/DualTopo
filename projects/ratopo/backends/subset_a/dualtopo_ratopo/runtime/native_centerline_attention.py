"""DualTopo RATopo inference components."""
from torch import nn


from centerline_dynamic_pe import CenterlineDynamicPointPE, build_point_attention_class


class NativeCenterlineAttention(nn.Module):
    def __init__(self, native_attention, donor_source, cuda_function=None):
        super().__init__()
        if native_attention.embed_dims != 256 or native_attention.num_levels != 1:
            raise ValueError('Only audited 256-channel, one-level native BEV supported')
        if native_attention.batch_first:
            raise ValueError('Native query layout must be Q,B,C')
        self.native_attention = native_attention
        self.dynamic_pe = CenterlineDynamicPointPE(num_points=11, num_layers=6)
        cls = build_point_attention_class(donor_source, cuda_function=cuda_function)
        self.point_attention = cls(embed_dims=256, num_heads=native_attention.num_heads,
            num_levels=1, num_points=32, im2col_step=native_attention.im2col_step,
            dropout=native_attention.dropout.p, batch_first=False,
            num_centerline_points=11)
        self.cuda_ready = cuda_function is not None

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                key_pos=None, attn_mask=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None, level_start_index=None,
                dualtopo_centerline_points=None, dualtopo_layer_index=None, **kwargs):
        common = dict(key_padding_mask=key_padding_mask, reference_points=reference_points,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        if dualtopo_centerline_points is None:
            if dualtopo_layer_index is not None:
                raise ValueError('Layer index without explicit geometry is ambiguous')
            return self.native_attention(query, key, value, identity,
                query_pos=query_pos, key_pos=key_pos, attn_mask=attn_mask, **common, **kwargs)
        if dualtopo_layer_index is None:
            raise ValueError('Original producing decoder layer index required')
        points = dualtopo_centerline_points
        if points.shape != (query.shape[1], query.shape[0], 11, 3) or query.shape[0] != 300:
            raise ValueError('Exactly 300 main native lane queries; exclude DN/point/auxiliary regions')
        if query.is_cuda and not self.cuda_ready:
            raise ValueError('Explicit audited CUDA MSDA function required')
        pos, refs = self.dynamic_pe(dualtopo_layer_index, points)
        common['reference_points'] = refs
        # Donor attention consumes per-point PE. Native static query_pos is
        # still used by native self attention; no boundary points or mean.
        return self.point_attention(query, key, value, identity,
            pt_query_pos=pos, **common, **kwargs)

