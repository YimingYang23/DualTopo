"""DualTopo RATopo inference components."""
import torch


from torch import nn


from temporal_state import TemporalState, tokens_to_bev, bev_to_tokens


class BEVWorldModelState(nn.Module):
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model
        self.state = TemporalState()

    def train(self, mode=True):
        super().train(mode)
        self.state.train(mode)
        return self

    def prepare(self, current_tokens, img_metas):
        current = tokens_to_bev(current_tokens, 100, 200)
        if current.shape[1] != 256 or current.shape[0] != len(img_metas):
            raise ValueError('Native [B,20000,256] BEV and aligned metadata required')
        records = self.state.get(img_metas)
        temporal, singleframe, predictions, valid = [], [], [], []
        neck = self.world_model.stream_fusion_neck
        for i, record in enumerate(records):
            if record is None:
                fused = neck(current[i].clone().detach(), current[i])
                prediction = fused
            else:
                prediction = self.world_model.predict_one(
                    record['payload']['bev'], record['meta'], img_metas[i])
                fused = neck(prediction, current[i])
            temporal.append(fused)
            predictions.append(prediction)
            # Preserve the donor's self-fusion SF branch and its computation
            # order even for a first frame. Do not bypass GRU as an identity.
            singleframe.append(neck(current[i].clone().detach(), current[i]))
            valid.append(record is not None)
        return dict(singleframe=bev_to_tokens(torch.stack(singleframe)),
            temporal=bev_to_tokens(torch.stack(temporal)),
            prediction=bev_to_tokens(torch.stack(predictions)),
            has_history=tuple(valid), records=records, training_mode=self.training,
            owner=id(self))

    def reconstruction_loss(self, prepared):
        self._check_packet(prepared)
        if not any(prepared['has_history']):
            return prepared['prediction'].new_zeros((1,))
        # Actual donor loss_plan_rec is unreduced MSE. Its target is the fused
        # temporal BEV, detached, rather than the unfused current backbone BEV.
        error = (prepared['prediction'] - prepared['temporal'].detach()).square()
        per_sample = error.mean(dim=(1, 2))
        weights = per_sample.new_tensor(prepared['has_history'])
        return (per_sample * weights).sum() / weights.sum()

    def commit(self, prepared, img_metas, extra_payloads=None):
        self._check_packet(prepared)
        if len(img_metas) != len(prepared['has_history']):
            raise ValueError('Native metadata batch changed before state commit')
        if extra_payloads is None:
            extra_payloads = [{} for _ in img_metas]
        if len(extra_payloads) != len(img_metas) or any('bev' in x for x in extra_payloads):
            raise ValueError('Extra query payloads must not replace the fused BEV')
        planar = tokens_to_bev(prepared['temporal'], 100, 200)
        self.state.update([dict(extra, bev=bev) for extra, bev in zip(extra_payloads, planar)], img_metas)

    def _check_packet(self, prepared):
        if prepared['owner'] != id(self) or prepared['training_mode'] != self.training:
            raise ValueError('Do not mix module or train/test BEV packets')

