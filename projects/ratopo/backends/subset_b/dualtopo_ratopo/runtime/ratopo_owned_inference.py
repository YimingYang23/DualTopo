"""DualTopo RATopo inference components."""
import ast,copy,sys,types


from ratopo_query_state import prepare_query_boundary


from ratopo_centerline_query_training import CenterlineQueryFork,export_query_boundary


from ratopo_centerline_decoder import CenterlineGeometryRouter


def attach_owned_centerline_inference(model,source_path,*,geometry_router_class=None,require_isolated_mode=False):
    if type(model).__name__!='RATopo' or hasattr(model,'simple_test_owned_centerline_query'):
        raise ValueError('Fresh private enhanced RATopo instance required')
    bridge=model._modules.get('dualtopo_bev_bridge');qwm=model._modules.get('dualtopo_query_world_model')
    if bridge is None or qwm is None or not getattr(model.pts_bbox_head.transformer.decoder,'_private_centerline_attached',False):
        raise ValueError('Audited registered WMs and centerline decoder required')
    owned_ids=tuple(id(p) for m in (bridge,qwm) for p in m.parameters());keys=tuple(model.state_dict())
    def guard(self,prev_bev,evaluation_branch):
        if self is not model or self.training or bridge.training or qwm.training:
            raise ValueError('Use the owned model in evaluation mode')
        if self.video_test_mode or prev_bev is not None:
            raise ValueError('Native video history must not be mixed with registered WM history')
        if self._modules.get('dualtopo_bev_bridge') is not bridge or self._modules.get('dualtopo_query_world_model') is not qwm or tuple(id(p) for m in (bridge,qwm) for p in m.parameters())!=owned_ids:
            raise ValueError('Registered WM identity changed; rebuild model and ownership binding')
        if require_isolated_mode:
            from migration_temporal_evaluation import guard_native_evaluation
            guard_native_evaluation(self,evaluation_branch,prev_bev)
        elif evaluation_branch!='temporal':raise ValueError('SF evaluation requires isolated mode')
    cls=next(n for n in ast.parse(source_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RATopo')
    def method(name,newname):
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
        fn.name=newname;fn.decorator_list=[]
        fn.args.kwonlyargs.append(ast.arg(arg='initial_centerline_points'));fn.args.kw_defaults.append(ast.Constant(value=None))
        fn.args.kwonlyargs.append(ast.arg(arg='evaluation_branch'));fn.args.kw_defaults.append(ast.Constant(value='temporal'))
        fn.body[0:0]=ast.parse('guard(self,prev_bev,evaluation_branch)\nif initial_centerline_points is None:\n    raise ValueError("Explicit audited eleven-point geometry required")').body
        return fn
    fn=method('simple_test_pts','simple_test_pts_owned_centerline_query')
    sites=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bev_feats' for t in n.targets)]
    tries=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Try)]
    if len(sites)!=1 or len(tries)!=1:raise ValueError('Native inference boundaries changed; re-audit')
    replacement=ast.parse('''
bev_packet = self.dualtopo_bev_bridge.prepare(bev_feats,img_metas)
boundary = prepare_query_boundary(bev_packet,img_metas,self.dualtopo_query_world_model,self.pts_bbox_head.pc_range)
router = CenterlineGeometryRouter(initial_centerline_points,self.pts_bbox_head.pc_range)
fork = CenterlineQueryFork(bev_packet['temporal'].permute(1,0,2),router,export_query_boundary(boundary))
groups = self.pts_bbox_head.forward_shared(x,bev_packet['singleframe'],img_metas,te_feats,te_cls_scores,dualtopo_shared_fork=fork)
outs = groups[evaluation_branch]
self._owned_eval_last_trace=dict(branch=evaluation_branch,has_history=bev_packet['has_history'],observed_WM_packet=True)
''').body
    fn.body[tries[0]:tries[0]+1]=replacement
    returns=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Return)]
    if len(returns)!=1:raise ValueError('Native inference return changed')
    ret=fn.body[returns[0]]
    if not isinstance(ret.value,ast.Tuple) or not isinstance(ret.value.elts[0],ast.Name) or ret.value.elts[0].id!='bev_feats':
        raise ValueError('Native BEV/result layout changed')
    fn.body[returns[0]:returns[0]]=ast.parse("if evaluation_branch=='temporal':\n self.dualtopo_bev_bridge.commit(bev_packet,img_metas,fork.cache_payloads(groups))").body
    ret.value.elts[0]=ast.parse("bev_packet[evaluation_branch]",mode='eval').body
    outer=method('simple_test','simple_test_owned_centerline_query')
    calls=[n for n in ast.walk(outer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='simple_test_pts']
    if len(calls)!=1:raise ValueError('Native whole-image inference call changed')
    calls[0].func.attr='simple_test_pts_owned_centerline_query'
    calls[0].keywords.append(ast.keyword(arg='initial_centerline_points',value=ast.Name(id='initial_centerline_points',ctx=ast.Load())))
    calls[0].keywords.append(ast.keyword(arg='evaluation_branch',value=ast.Name(id='evaluation_branch',ctx=ast.Load())))
    namespace=dict(vars(sys.modules[type(model).__module__]),guard=guard,prepare_query_boundary=prepare_query_boundary,
        CenterlineGeometryRouter=CenterlineGeometryRouter,CenterlineQueryFork=CenterlineQueryFork,export_query_boundary=export_query_boundary)
    if geometry_router_class is not None:namespace['CenterlineGeometryRouter']=geometry_router_class
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn,outer],type_ignores=[])),str(source_path)+'<private owned inference>','exec'),namespace)
    model.simple_test_pts_owned_centerline_query=types.MethodType(namespace[fn.name],model)
    model.simple_test_owned_centerline_query=types.MethodType(namespace[outer.name],model)
    assert tuple(model.state_dict())==keys
    return dict(original_methods_unchanged=True,state_keys_unchanged=True,registered_WM_parameter_tensors=len(owned_ids),
        temporal_branch_decoded_by_native_get_lanes=True,production_runner=False,full_stage2_ready=False,dispatch_allowed=False)

