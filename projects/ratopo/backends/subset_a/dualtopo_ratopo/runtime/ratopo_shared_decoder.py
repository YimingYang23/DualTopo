"""DualTopo RATopo inference components."""
import ast


import copy


import torch


class RATopoSharedFork:
    def __init__(self, temporal_value, lane_boundary=None):
        self.temporal_value=temporal_value
        self.lane_boundary=lane_boundary
        self.calls=[]
        self.batch_size=None

    def fork(self, output, references, ll, lt, scores, coordinates, te,
             te_scores, padding, kwargs, ll_heads, lt_heads):
        self.calls.append(0)
        if len(self.calls)!=1:
            raise ValueError('Use a fresh fork object for each native forward')
        b=output.shape[1];self.batch_size=b
        value=kwargs.get('value')
        if value is None or value.ndim!=3 or value.shape[1]!=b:
            raise ValueError('Explicit native current BEV [HW,B,C] required')
        if self.temporal_value.shape!=value.shape:
            raise ValueError('Temporal BEV must keep the native layout and batch')
        temporal_output,temporal_references,temporal_ll,temporal_lt,temporal_kwargs=(output,references,ll,lt,kwargs)
        if self.lane_boundary is not None:
            temporal_output,temporal_references,temporal_ll,temporal_lt,temporal_kwargs=self.lane_boundary(
                output,references,ll,lt,scores,coordinates,te[0],kwargs,ll_heads,lt_heads,0)
        metas=kwargs['img_metas']
        if len(metas)!=b:
            raise ValueError('Native metadata must align with the original batch')
        combined=dict(kwargs,value=torch.cat((value,self.temporal_value),dim=1),
            query_pos=torch.cat((kwargs['query_pos'],temporal_kwargs['query_pos']),dim=1),
            img_metas=list(metas)+list(metas))
        if padding is not None:
            if padding.shape[0]!=b:raise ValueError('Native key padding batch axis changed')
            padding=torch.cat((padding,padding),dim=0)
        return (torch.cat((output,temporal_output),dim=1),
            torch.cat((references,temporal_references),dim=0),
            torch.cat((ll,temporal_ll),dim=1),torch.cat((lt,temporal_lt),dim=1),
            torch.cat((te,te),dim=1),torch.cat((te_scores,te_scores),dim=1),padding,combined)

    def pack(self, intermediates):
        if self.calls!=[0] or self.batch_size is None:
            raise ValueError('Native shared-layer fork did not execute exactly once')
        if len(intermediates)!=6 or any(len(x)!=6 for x in intermediates):
            raise ValueError('Only audited native six-layer intermediate output supported')
        shared=tuple(x[0].unsqueeze(0) for x in intermediates)
        branches=tuple(torch.stack(x[1:]) for x in intermediates)
        for index,(a,c) in enumerate(zip(shared,branches)):
            batch_axis=2 if index==0 else 1
            if a.shape[batch_axis]!=self.batch_size or c.shape[batch_axis]!=2*self.batch_size:
                raise ValueError('Shared versus forked native batch axes changed')
        return dict(shared=shared,branches=branches,shared_layer_indices=(0,),
            branch_layer_indices=(1,2,3,4,5),batch_size=self.batch_size,
            batch_order=('singleframe','temporal'))


def make_shared_forward(source, namespace):
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RATopoDecoder')
    fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward'))
    loop=next(x for x in fn.body if isinstance(x,ast.For))
    setup=ast.parse("fork = kwargs.pop('dualtopo_shared_fork', None)\nif fork is not None and not self.return_intermediate:\n    raise ValueError('Native shared fork requires intermediate outputs')").body
    fn.body[0:0]=setup
    loop.body[0:0]=ast.parse('''
if lid == 1 and fork is not None:
    output, reference_points, prev_lclc_adj, prev_lcte_adj, te_feats, te_cls_scores, key_padding_mask, kwargs = fork.fork(
        output, reference_points, prev_lclc_adj, prev_lcte_adj,
        outputs_class[:, :self.num_lanes_one2one], outputs_coord[:, :self.num_lanes_one2one],
        te_feats, te_cls_scores, key_padding_mask, kwargs, lclc_branches, lcte_branches)
''').body
    # Replace only the stacked intermediate return, retaining the native
    # no-fork return and every actual layer/head computation unchanged.
    returns=[x for x in ast.walk(fn) if isinstance(x,ast.Return) and isinstance(x.value,ast.Tuple)
        and isinstance(x.value.elts[0],ast.Call)]
    if len(returns)!=1:raise ValueError('Native intermediate return changed; re-audit')
    target=returns[0]
    replacement=ast.If(test=ast.Compare(left=ast.Name(id='fork',ctx=ast.Load()),ops=[ast.IsNot()],comparators=[ast.Constant(value=None)]),
        body=ast.parse('return fork.pack((intermediate, intermediate_reference_points, intermediate_cls, intermediate_reg, intermediate_lclc_rel, intermediate_lcte_rel))').body,
        orelse=[target])
    class Replace(ast.NodeTransformer):
        def visit_Return(self,node):return replacement if node is target else node
    fn=Replace().visit(fn);fn.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ns=dict(namespace);exec(compile(module,'<private-RATopo-shared-fork>','exec'),ns)
    return ns['forward']


def prediction_groups(result):
    """Expose native head dictionaries with shared layer 0 supervised once.

    The caller must align TE assignments to original indices. It must not
    prepend the shared layer to each branch and double its supervision.
    """
    b=result['batch_size']
    def pack(values):
        hs,refs,cls,coords,ll,lt=values
        return dict(history_states=hs.permute(0,2,1,3),all_cls_scores=cls,
            all_lanes_preds=coords,all_lclc_preds=ll,all_lcte_preds=lt)
    def branch(offset):
        return tuple(x[:,:,offset:offset+b] if i==0 else x[:,offset:offset+b]
                     for i,x in enumerate(result['branches']))
    return dict(shared=pack(result['shared']),singleframe=pack(branch(0)),
        temporal=pack(branch(b)),shared_layer_indices=result['shared_layer_indices'],
        branch_layer_indices=result['branch_layer_indices'])

