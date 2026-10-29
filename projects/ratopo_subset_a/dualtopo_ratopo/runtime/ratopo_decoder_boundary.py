"""DualTopo RATopo inference components."""
import ast


import copy


import torch


from query_adapter import merge_lane_region


from ratopo_graph_adapter import refresh_ratopo_graphs


def make_boundary_forward(native_forward_source, namespace):
    tree = ast.parse(native_forward_source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == 'RATopoDecoder')
    fn = copy.deepcopy(next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                            and x.name == 'forward'))
    loop = next(x for x in fn.body if isinstance(x, ast.For))
    # Consume the opt-in keyword before any native attention layer receives it.
    setup = ast.parse("boundary = kwargs.pop('dualtopo_lane_boundary', None)").body
    fn.body[0:0] = setup
    hook = ast.parse('''
if lid == 1 and boundary is not None:
    output, reference_points, prev_lclc_adj, prev_lcte_adj, kwargs = boundary(
        output, reference_points, prev_lclc_adj, prev_lcte_adj,
        outputs_class[:, :self.num_lanes_one2one],
        outputs_coord[:, :self.num_lanes_one2one],
        te_feats[lid - 1], kwargs, lclc_branches, lcte_branches, lid - 1)
''').body
    loop.body[0:0] = hook
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = dict(namespace)
    exec(compile(module, '<private-RATopo-PQS-boundary>', 'exec'), ns)
    return ns['forward']


class RATopoLaneBoundary:
    def __init__(self, query, query_pos, anchors, centerline_world, has_history):
        self.query = query
        self.query_pos = query_pos
        self.anchors = anchors
        self.centerline_world = centerline_world
        self.has_history = tuple(has_history)
        self.calls = []

    def __call__(self, output, anchors, ll_adj, lt_adj, scores, world,
                 te, kwargs, ll_heads, lt_heads, producing_layer):
        self.calls.append(producing_layer)
        if producing_layer != 0:
            raise ValueError('Inject immediately after the original shared layer 0')
        b, n, c = output.permute(1, 0, 2).shape
        if n != 300 or self.query.shape != (b, 100, c):
            raise ValueError('Only the audited 300 lane queries and donor K=100')
        if self.centerline_world.shape != (b, 100, 33) or world.shape != (b, n, 33):
            raise ValueError('All 11 native 3D centerline points must be supplied')
        if not any(self.has_history):
            return output, anchors, ll_adj, lt_adj, kwargs
        pos = kwargs.get('query_pos')
        if pos is None or pos.shape != output.shape:
            raise ValueError('Explicit native query position [N,B,C] required')
        mixed = merge_lane_region(output.permute(1, 0, 2), pos.permute(1, 0, 2),
            anchors, scores, self.query, self.query_pos, self.anchors,
            self.has_history, 0, n)
        coordinates = []
        for slot, has in enumerate(self.has_history):
            if not has:
                coordinates.append(world[slot])
            else:
                coordinates.append(torch.cat((self.centerline_world[slot],
                    world[slot, mixed['retained_current_indices'][slot]]), dim=0))
        graphs = refresh_ratopo_graphs(mixed['query'], torch.stack(coordinates),
            te, kwargs['img_metas'], ll_heads, lt_heads, producing_layer, ll_adj, lt_adj)
        # No-history slots keep the exact old graph and gradient path.
        mask = torch.tensor(self.has_history, device=output.device, dtype=torch.bool)
        mask = mask.view(1, b, 1, 1, 1)
        new_ll = torch.where(mask, graphs['lclc_adj'], ll_adj)
        new_lt = torch.where(mask, graphs['lcte_adj'], lt_adj)
        new_kwargs = dict(kwargs, query_pos=mixed['query_pos'].permute(1, 0, 2))
        return mixed['query'].permute(1, 0, 2), mixed['references'], new_ll, new_lt, new_kwargs

