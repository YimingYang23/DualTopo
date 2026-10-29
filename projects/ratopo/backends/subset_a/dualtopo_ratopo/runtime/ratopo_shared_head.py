"""DualTopo RATopo inference components."""
import ast,copy,hashlib,sys,types


from ratopo_shared_decoder import make_shared_forward,prediction_groups


def _forward(source,class_name):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))


def _method(fn,path,namespace,owner):
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ns=dict(namespace,prediction_groups=prediction_groups)
    exec(compile(module,str(path),'exec'),ns)
    return types.MethodType(ns['forward_shared'],owner)


def attach_shared_head(head,head_path,transformer_path,decoder_path):
    if type(head).__name__!='RATopoHead' or getattr(head,'_dualtopo_shared_attached',False):
        raise ValueError('Use a fresh audited native RATopoHead instance')
    transformer=head.transformer;decoder=transformer.decoder
    keys=tuple(head.state_dict())
    fn=_forward(head_path.read_text(),'RATopoHead');fn.name='forward_shared'
    fn.args.kwonlyargs.append(ast.arg(arg='dualtopo_shared_fork'));fn.args.kw_defaults.append(None)
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='transformer']
    if len(calls)!=1:raise ValueError('Native transformer call changed')
    calls[0].func=ast.Attribute(value=calls[0].func,attr='forward_shared',ctx=ast.Load())
    calls[0].keywords.append(ast.keyword(arg='dualtopo_shared_fork',value=ast.Name(id='dualtopo_shared_fork',ctx=ast.Load())))
    assignments=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='outputs' for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Native head prediction return changed')
    fn.body=fn.body[:assignments[0]+1]+ast.parse('return prediction_groups(outputs)').body
    head.forward_shared=_method(fn,head_path,vars(sys.modules[type(head).__module__]),head)

    fn=_forward(transformer_path.read_text(),type(transformer).__name__);fn.name='forward_shared'
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='decoder']
    if len(calls)!=1:raise ValueError('Native decoder call changed')
    calls[0].func=ast.Attribute(value=calls[0].func,attr='forward_shared',ctx=ast.Load())
    assignments=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and n.value is calls[0]]
    if len(assignments)!=1:raise ValueError('Native transformer decoder return changed')
    i=assignments[0];fn.body[i].targets=[ast.Name(id='branch_result',ctx=ast.Store())]
    fn.body=fn.body[:i+1]+ast.parse('return branch_result').body
    transformer.forward_shared=_method(fn,transformer_path,vars(sys.modules[type(transformer).__module__]),transformer)
    decoder.forward_shared=types.MethodType(make_shared_forward(decoder_path.read_text(),vars(sys.modules[type(decoder).__module__])),decoder)
    head._dualtopo_shared_attached=True
    assert tuple(head.state_dict())==keys
    return dict(head_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest(),
        transformer_sha256=hashlib.sha256(transformer_path.read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256(decoder_path.read_bytes()).hexdigest(),original_forward_unchanged=True)

