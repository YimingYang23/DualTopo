# SPDX-License-Identifier: Apache-2.0
# Portions copyright OpenDriveLab; see NOTICE.
# Modified for the DualTopo inference-only release: paper names and portable interfaces.
"""Inference decoding without assigners or losses."""

from torch import nn


class LaneSegmentPseudoCoder(nn.Module):
    """Decode metric XYZ lane segments and sigmoid class/type predictions."""

    def __init__(self, denormalize=False):
        super().__init__()
        self.denormalize = denormalize

    def decode_single(self, cls_scores, lane_preds, lane_left_type_scores, lane_right_type_scores):
        """Decode Q lane segments with 90 coordinates: center, left and right, each 10 XYZ points."""
        final_lane_preds = lane_preds
        cls_scores = cls_scores.sigmoid()
        scores, labels = cls_scores.max(-1)
        left_type_scores, left_type_labels = lane_left_type_scores.sigmoid().max(-1)
        right_type_scores, right_type_labels = lane_right_type_scores.sigmoid().max(-1)
        predictions_dict = {
            "lane3d": final_lane_preds.detach().cpu().numpy(),
            "scores": scores.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
            "left_type_scores": left_type_scores.detach().cpu().numpy(),
            "left_type_labels": left_type_labels.detach().cpu().numpy(),
            "right_type_scores": right_type_scores.detach().cpu().numpy(),
            "right_type_labels": right_type_labels.detach().cpu().numpy(),
        }
        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_lanes_preds = preds_dicts["all_lanes_preds"][-1]
        all_lanes_left_types = preds_dicts["all_lanes_left_type"][-1]
        all_lanes_right_types = preds_dicts["all_lanes_right_type"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i],
                    all_lanes_preds[i],
                    all_lanes_left_types[i],
                    all_lanes_right_types[i],
                )
            )
        return predictions_list
