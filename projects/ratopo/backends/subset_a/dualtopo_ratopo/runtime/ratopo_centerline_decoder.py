"""DualTopo RATopo inference components."""
import ast, copy, inspect, types


import torch


from ratopo_decoder_boundary import RATopoLaneBoundary


from ratopo_shared_decoder import RATopoSharedFork, make_shared_forward


from native_centerline_attention import NativeCenterlineAttention


def _export_boundary_call():
    source = inspect.getsource(RATopoLaneBoundary)
    cls = ast.parse(source).body[0]
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__'))
    # Preserve the native graph computation and exact PQS retention indices.
    for node in ast.walk(fn):
        if isinstance(node,ast.Return) and isinstance(node.value,ast.Tuple):
            last=node.value.elts[-1]
            if isinstance(last,ast.Name) and last.id=='kwargs':
                node.value.elts[-1]=ast.parse("dict(kwargs,dualtopo_centerline_world=world)",mode='eval').body
            elif isinstance(last,ast.Name) and last.id=='new_kwargs':
                node.value.elts[-1]=ast.parse("dict(new_kwargs,dualtopo_centerline_world=torch.stack(coordinates))",mode='eval').body
    ns=dict(RATopoLaneBoundary.__call__.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 '<private-native-PQS-geometry-export>','exec'),ns)
    return ns['__call__']


class CenterlineBoundary(RATopoLaneBoundary):
    __call__ = _export_boundary_call()


class CenterlineGeometryRouter:
    def __init__(self, initial_normalized_points, pc_range):
        if initial_normalized_points.ndim!=4 or initial_normalized_points.shape[1:]!=(300,11,3):
            raise ValueError('Explicit initial normalized B,300,11,3 geometry required')
        self.current=initial_normalized_points.detach()
        self.pc_range=tuple(pc_range)
        if any(pc_range[i+3]<=pc_range[i] for i in range(3)):
            raise ValueError('Positive native point cloud extent required')
        self.inputs=[];self.outputs=[]

    def normalize(self, world):
        origin=world.new_tensor(self.pc_range[:3])
        extent=world.new_tensor(self.pc_range[3:])-origin
        return ((world.detach().reshape(world.shape[0],300,11,3)-origin)/extent).detach()

    def before_layer(self, layer, output, kwargs):
        world=kwargs.pop('dualtopo_centerline_world',None)
        if world is not None:
            if layer!=1:raise ValueError('PQS geometry may change order only immediately after shared layer0')
            self.current=self.normalize(world)
        if self.current.shape[:2]!=(output.shape[1],output.shape[0]):
            raise ValueError('Query, branch batch and all eleven geometry points must align')
        if layer!=len(self.inputs):raise ValueError('Original six-layer order required')
        self.inputs.append(self.current.clone())
        return dict(kwargs,dualtopo_centerline_points=self.current,dualtopo_layer_index=layer)

    def after_layer(self, layer, main_world):
        if layer!=len(self.outputs):raise ValueError('Record each original layer once')
        if main_world.shape[1:]!=(300,33):raise ValueError('Do not use auxiliary coordinates as main geometry')
        self.current=self.normalize(main_world)
        self.outputs.append(self.current.clone())


class CenterlineFork(RATopoSharedFork):
    def __init__(self, temporal_value, router, lane_boundary=None):
        super().__init__(temporal_value,lane_boundary)
        if lane_boundary is not None and not isinstance(lane_boundary,CenterlineBoundary):
            raise ValueError('PQS must export its actual eleven-point ordering')
        self.centerline_router=router

    def fork(self, output, references, ll, lt, scores, coordinates, te,
             te_scores, padding, kwargs, ll_heads, lt_heads):
        # One call to the established fork and one call to the actual boundary.
        boundary=self.lane_boundary
        captured={}
        if boundary is not None:
            def recording(*args,**kw):
                result=boundary(*args,**kw)
                captured['world']=result[-1]['dualtopo_centerline_world']
                return result
            self.lane_boundary=recording
        try:
            result=super().fork(output,references,ll,lt,scores,coordinates,
                te,te_scores,padding,kwargs,ll_heads,lt_heads)
        finally:
            self.lane_boundary=boundary
        kw=dict(result[-1],dualtopo_centerline_world=torch.cat((coordinates,
            captured.get('world',coordinates)),dim=0))
        return result[:-1]+(kw,)


def make_centerline_forward(source,namespace):
    # Derive from the already audited private shared-fork builder. Inject
    # geometry calls into its native AST, preserving its entire fork/return.
    builder=ast.parse(inspect.getsource(make_shared_forward)).body[0]
    builder.name='make_private_centerline_forward'
    injection=ast.parse('''
fn.body[0:0] = ast.parse("router = getattr(kwargs.get('dualtopo_shared_fork'), 'centerline_router', None)").body
loop.body.insert(1, ast.parse("if router is not None:\\n    kwargs = router.before_layer(lid, output, kwargs)").body[0])
main_assignments=[(i,n) for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='outputs_coord' for t in n.targets)]
if len(main_assignments)!=1:raise ValueError('Native main coordinate assignment changed; re-audit')
loop.body.insert(main_assignments[0][0]+1,ast.parse("if router is not None:\\n    router.after_layer(lid, outputs_coord[:, :self.num_lanes_one2one])").body[0])
''').body
    index=next(i for i,n in enumerate(builder.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='module' for t in n.targets))
    builder.body[index:index]=injection
    ns=dict(make_shared_forward.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[builder],type_ignores=[])),
                 '<private-centerline-shared-builder>','exec'),ns)
    return ns['make_private_centerline_forward'](source,namespace)


def attach_centerline_decoder(decoder, donor_source, native_source, namespace, cuda_function=None):
    if getattr(decoder,'_private_centerline_attached',False):raise ValueError('Use a fresh private decoder')
    for inquiry in decoder.mi_layers:
        for layer in inquiry:
            index=[op for op in layer.operation_order if op.endswith('attn')].index('cross_attn')
            layer.attentions[index]=NativeCenterlineAttention(layer.attentions[index],donor_source,cuda_function)
    decoder.forward_shared=types.MethodType(make_centerline_forward(native_source,namespace),decoder)
    decoder._private_centerline_attached=True

