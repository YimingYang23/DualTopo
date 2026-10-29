"""DualTopo RATopo inference components."""
import copy


from contextlib import contextmanager


import torch


from torch.utils.data import Sampler


def scene_rank_plan(data_infos,world_size=4):
    if world_size!=4 or not data_infos:raise ValueError('Actual four-rank validation metadata required')
    sequences=[];closed=set();scene=None;previous=None
    for index,info in enumerate(data_infos):
        current=str(info['segment_id']);timestamp=int(info['timestamp'])
        if current!=scene:
            if current in closed:raise ValueError('Interleaved scene metadata requires audit')
            if scene is not None:closed.add(scene)
            sequences.append([]);scene=current;previous=None
        if previous is not None and timestamp<=previous:raise ValueError('Scene timestamps must be strictly chronological')
        sequences[-1].append(index);previous=timestamp
    ranks=[[] for _ in range(world_size)];loads=[0]*world_size
    for sequence in sorted(sequences,key=lambda x:(-len(x),x[0])):
        rank=min(range(world_size),key=lambda r:(loads[r],r))
        ranks[rank].extend(sequence);loads[rank]+=len(sequence)
    ranks=[sorted(indices) for indices in ranks]
    if not all(ranks) or sorted(index for rows in ranks for index in rows)!=list(range(len(data_infos))):
        raise ValueError('Each native frame must appear once without padding/dropping')
    return ranks


class NativeSceneEvaluationSampler(Sampler):
    def __init__(self,dataset,rank,world_size=4):
        if type(rank)!=int or not 0<=rank<world_size:raise ValueError('Actual rank required')
        self.indices=scene_rank_plan(dataset.data_infos,world_size)[rank]
    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)


def collect_native_indexed_results(parts,native_size):
    if len(parts)!=4:raise ValueError('Four actual rank result parts required')
    ordered=[None]*native_size;seen=set()
    for part in parts:
        for index,native_prediction in part:
            if type(index)!=int or not 0<=index<native_size or index in seen:raise ValueError('Duplicate/missing/foreign native index')
            seen.add(index);ordered[index]=native_prediction
    if len(seen)!=native_size:raise ValueError('Full native validation size required')
    return ordered


def _states(model):
    result=[model.dualtopo_bev_bridge.state]
    gt=model._modules.get('dualtopo_native_GT_state')
    if gt is not None:result.append(gt.runtime)
    return result


@contextmanager
def isolated_native_evaluation_mode(model,branch):
    if branch not in ('singleframe','temporal'):raise ValueError('Explicit SF/temporal branch required')
    if hasattr(model,'_owned_evaluation_branch'):raise ValueError('Nested evaluation is unsupported')
    modes={module:module.training for module in model.modules()};states=_states(model)
    train_buffers=[state._buffers[True] for state in states];state_modes=[state.training for state in states]
    model.eval()
    for state in states:state.clear()
    model._owned_evaluation_branch=branch
    try:
        yield branch
    finally:
        changed=any(state._buffers[True] is not training for state,training in zip(states,train_buffers))
        for state,training in zip(states,train_buffers):
            state._buffers[True]=training
            state.clear()
        delattr(model,'_owned_evaluation_branch')
        model.train(modes[model])
        # Retain native mixed frozen/norm children modes exactly.
        for module,mode in modes.items():module.training=mode
        for state,mode in zip(states,state_modes):state.train(mode)
        if changed:raise ValueError('Evaluation replaced native training memory')


def guard_native_evaluation(model,branch,prev_bev):
    if model.training or model.video_test_mode or prev_bev is not None:
        raise ValueError('Owned eval mode only; native video BEV must not be mixed with WMs')
    if getattr(model,'_owned_evaluation_branch',None)!=branch or branch not in ('singleframe','temporal'):
        raise ValueError('Use the isolated explicit branch evaluation scope')
    for state in _states(model):
        if state.training:raise ValueError('Evaluation runtime state must be isolated')
        if branch=='singleframe':state.clear()

