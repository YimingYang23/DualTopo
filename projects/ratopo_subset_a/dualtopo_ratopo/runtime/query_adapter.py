"""DualTopo RATopo inference components."""
import torch


def transport_normalized_points(points, previous_to_current, pc_range):
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError('Expected [B,K,...,3] references')
    if previous_to_current.shape != (points.shape[0], 4, 4):
        raise ValueError('Expected [B,4,4] ego actions')
    origin = points.new_tensor(pc_range[:3])
    extent = points.new_tensor(pc_range[3:]) - origin
    if (extent <= 0).any():
        raise ValueError('Invalid native point-cloud range')
    world = points * extent + origin
    homogeneous = torch.cat((world, torch.ones_like(world[..., :1])), -1)
    shape = homogeneous.shape
    moved = torch.bmm(homogeneous.reshape(shape[0], -1, 4).double(),
                      previous_to_current.double().transpose(1, 2))
    moved = moved.reshape(shape)[..., :3].to(points)
    return ((moved - origin) / extent).clamp(0, 1)


def merge_lane_region(query, query_pos, references, scores,
                      propagated_query, propagated_pos, propagated_references,
                      has_history, start, stop):
    """Keep the native total query budget and every external region's ordering.

Queries/positions are [B,N,C], scores are [B,N,classes]. References can be
native anchors [B,N,3] or explicit point-token references [B,N,P,3].
This does not decide how a native model groups its point tokens.
"""
    b, n, c = query.shape
    k = propagated_query.shape[1]
    if query_pos.shape != query.shape or scores.shape[:2] != (b, n):
        raise ValueError('Native query axes do not agree')
    if references.shape[:2] != (b, n) or propagated_query.shape != (b, k, c):
        raise ValueError('History/native axes do not agree')
    if propagated_pos.shape != (b, k, c) or propagated_references.shape != (b, k) + references.shape[2:]:
        raise ValueError('Native position/reference contracts do not agree')
    if not (0 <= start < stop <= n and 0 < k < stop - start):
        raise ValueError('Top-k must fit the explicitly eligible lane region')
    if len(has_history) != b:
        raise ValueError('Per-slot scene/history flags required')
    result_q, result_pos, result_ref, current_indices = [], [], [], []
    for i in range(b):
        if not has_history[i]:
            result_q.append(query[i]); result_pos.append(query_pos[i]); result_ref.append(references[i])
            current_indices.append(torch.arange(start, stop, device=query.device))
            continue
        with torch.no_grad():
            keep = scores[i, start:stop].max(-1)[0].topk(stop - start - k)[1] + start
        for result, native, history in [(result_q, query, propagated_query),
                                         (result_pos, query_pos, propagated_pos),
                                         (result_ref, references, propagated_references)]:
            result.append(torch.cat((native[i, :start], history[i], native[i, keep], native[i, stop:]), 0))
        current_indices.append(keep)
    return {'query': torch.stack(result_q), 'query_pos': torch.stack(result_pos),
            'references': torch.stack(result_ref), 'retained_current_indices': current_indices}

