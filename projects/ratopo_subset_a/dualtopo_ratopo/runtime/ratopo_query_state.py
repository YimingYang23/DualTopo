"""DualTopo RATopo inference components."""
import torch


from temporal_state import previous_to_current


from query_adapter import transport_normalized_points


from query_world_model import select_history_queries


from ratopo_decoder_boundary import RATopoLaneBoundary


from ratopo_shared_decoder import RATopoSharedFork


def prepare_query_boundary(bev_packet, img_metas, query_world_model, pc_range):
    reference=bev_packet['temporal']
    if len(img_metas)!=len(bev_packet['records']):
        raise ValueError('BEV/query metadata and batch slots must agree')
    queries,positions,anchors,worlds=[],[],[],[]
    origin=reference.new_tensor(pc_range[:3]);extent=reference.new_tensor(pc_range[3:])-origin
    for slot, record in enumerate(bev_packet['records']):
        if record is None:
            queries.append(reference.new_zeros((100,256)))
            positions.append(reference.new_zeros((100,256)))
            anchors.append(reference.new_zeros((100,3)))
            worlds.append(reference.new_zeros((100,33)))
            continue
        payload=record['payload'].get('native_lane')
        if payload is None:
            raise ValueError('BEV history exists without matched native query history; complete the cache before Stage2')
        if any(payload[k].shape!=shape for k,shape in [('query',(100,256)),('query_pos',(100,256)),('anchor',(100,3)),('world',(100,33))]):
            raise ValueError('Audited native 100-query/3D-anchor/11-point cache required')
        action=previous_to_current(record['meta'],img_metas[slot],reference).unsqueeze(0)
        queries.append(query_world_model(payload['query'].unsqueeze(0),action)[0])
        # RATopo's query position is a learned embedding. Keep its original
        # association rather than inventing a coordinate-to-position encoder.
        positions.append(payload['query_pos'])
        anchors.append(transport_normalized_points(payload['anchor'].unsqueeze(0),action,pc_range)[0])
        normalized=(payload['world'].reshape(1,100,11,3)-origin)/extent
        moved=transport_normalized_points(normalized,action,pc_range)[0]
        worlds.append((moved*extent+origin).reshape(100,33))
    return RATopoLaneBoundary(torch.stack(queries),torch.stack(positions),
        torch.stack(anchors),torch.stack(worlds),bev_packet['has_history'])


class RATopoQueryFork(RATopoSharedFork):
    """Capture native positional associations after temporal-only replacement."""
    def fork(self,*args,**kwargs):
        result=super().fork(*args,**kwargs)
        b=self.batch_size
        self.temporal_anchor=result[1][b:]
        self.temporal_position=result[-1]['query_pos'][:,b:].permute(1,0,2)
        return result

    def pack(self, intermediates):
        # The original native decoder keeps its learned anchor fixed across
        # layers. Reject silent iterative/averaged-anchor substitutions.
        b=self.batch_size
        for anchor in intermediates[1][1:]:
            torch.testing.assert_close(anchor[b:],self.temporal_anchor,rtol=0,atol=0,check_stride=False)
        return super().pack(intermediates)

    def cache_payloads(self,groups):
        if self.calls!=[0] or groups['branch_layer_indices']!=(1,2,3,4,5):
            raise ValueError('Complete the audited native shared forward first')
        temporal=groups['temporal']
        query=temporal['history_states'][-1,:,:300]
        score=temporal['all_cls_scores'][-1,:,:300].sigmoid()
        world=temporal['all_lanes_preds'][-1,:,:300]
        selected_q,selected_world,indices=select_history_queries(score,query,world,100)
        def gather(value):return value.gather(1,indices.unsqueeze(-1).expand(-1,-1,value.shape[-1])).detach().clone()
        selected_pos=gather(self.temporal_position);selected_anchor=gather(self.temporal_anchor)
        return [dict(native_lane=dict(query=q,query_pos=p,anchor=a,world=w)) for q,p,a,w in
            zip(selected_q,selected_pos,selected_anchor,selected_world)]

