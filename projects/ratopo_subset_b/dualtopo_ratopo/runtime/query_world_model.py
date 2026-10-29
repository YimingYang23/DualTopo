"""DualTopo RATopo inference components."""
import torch


from torch import nn


class QueryWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_aware_query_encoder = nn.Sequential(
            nn.Linear(256 + 12, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True)
        self._query_wm_decoder = nn.TransformerDecoder(decoder_layer, 2)

    def forward(self, historical_queries, previous_to_current):
        if historical_queries.ndim != 3 or historical_queries.shape[-1] != 256:
            raise ValueError('Expected [B,K,256] history queries')
        if previous_to_current.shape != (historical_queries.shape[0], 4, 4):
            raise ValueError('Expected [B,4,4] previous-to-current ego transform')
        action = previous_to_current[:, :3].to(historical_queries).reshape(-1, 1, 12)
        action = action.expand(-1, historical_queries.shape[1], -1)
        latent = self.action_aware_query_encoder(torch.cat((historical_queries, action), -1))
        return self._query_wm_decoder(latent, latent)


def select_history_queries(probabilities, queries, references, topk=66):
    """Donor max-class confidence top-k, over native eligible lane queries only.

    Callers must exclude DN, path and one-to-many query regions before calling.
    Probabilities are already sigmoid outputs, as in the donor. This helper
    does not decide native query grouping or how selected queries are injected.
    """
    if probabilities.ndim != 3 or queries.ndim != 3:
        raise ValueError('Expected batch/query/class and batch/query/channel')
    if probabilities.shape[:2] != queries.shape[:2] or references.shape[:2] != queries.shape[:2]:
        raise ValueError('Query, score and reference axes must agree')
    if topk > queries.shape[1] or topk <= 0:
        raise ValueError('topk must fit the eligible lane query region')
    indices = probabilities.max(-1)[0].topk(topk, dim=-1)[1]
    def gather(value):
        shape = indices.shape + (1,) * (value.ndim - 2)
        return value.gather(1, indices.reshape(shape).expand(indices.shape + value.shape[2:]))
    return gather(queries).detach().clone(), gather(references).detach().clone(), indices

