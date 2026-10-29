#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
from tqdm import tqdm
from openlanev2.lanesegment.io import io
from openlanev2.lanesegment.evaluation.distance import (pairwise, area_distance,
                                                        lane_segment_distance, lane_segment_distance_c)
from openlanev2.lanesegment.evaluation.evaluate import (THRESHOLDS_LANESEG, THRESHOLDS_AREA,
                                                        _mAP_over_threshold, _mAP_topology_lsls )


def _mAP_type_lsls(gts, preds, distance_thresholds):
    r"""
    Calculate mAP on topology among lane segments.

    Parameters
    ----------
    gts : dict
        Dict storing ground truth for all samples.
    preds : dict
        Dict storing predictions for all samples.
    distance_thresholds : list
        Distance thresholds.

    Returns
    -------
    float
        mAP over all samples abd distance thresholds.

    """
    acc_type = []
    for distance_threshold in distance_thresholds:
        for token in gts.keys():


            idx_match_gt = preds[token][f'lane_segment_{distance_threshold}_idx_match_gt']
            gt_pred = {m: i for i, m in enumerate(idx_match_gt) if not np.isnan(m)} ##哪些匹配上了



            gt_indices = np.array(list(gt_pred.keys())).astype(int)
            pred_indices = np.array(list(gt_pred.values())).astype(int)

            gt_left_type = [lanesegment['left_laneline_type'] for lanesegment in gts[token]['lane_segment']]
            gt_right_type = [lanesegment['right_laneline_type'] for lanesegment in gts[token]['lane_segment']]

            pred_left_type = [lanesegment['left_laneline_type'] for lanesegment in preds[token]['lane_segment']]
            pred_right_type = [lanesegment['right_laneline_type'] for lanesegment in preds[token]['lane_segment']]

            left_type_matching = np.zeros(len(gt_left_type))
            right_type_matching = np.zeros(len(gt_left_type))

            for idx, gt_indice in enumerate(gt_indices):
                if gt_left_type[gt_indice] == pred_left_type[pred_indices[idx]]:
                    left_type_matching[gt_indice] = 1.0
                if gt_right_type[gt_indice] == pred_right_type[pred_indices[idx]]:
                    right_type_matching[gt_indice] = 1.0
            

            acc_type.append((np.mean(left_type_matching)+np.mean(right_type_matching))/2)

    if len(acc_type) == 0:
        return np.float32(0)

    return np.hstack(acc_type).mean()


def lanesegnet_evaluate(ground_truth, predictions, verbose=True):

    if isinstance(ground_truth, str):
        ground_truth = io.pickle_load(ground_truth)

    if predictions is None:
        preds = {}
        print('\nDummy evaluation on ground truth.\n')
    else:
        if isinstance(predictions, str):
            predictions = io.pickle_load(predictions)
        predictions = predictions['results']

    gts = {}
    preds = {}
    for token in ground_truth.keys():
        gts[token] = ground_truth[token]['annotation']
        if predictions is None:
            preds[token] = gts[token]
            for i, _ in enumerate(preds[token]['lane_segment']):
                preds[token]['lane_segment'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['area']):
                preds[token]['area'][i]['confidence'] = np.float32(1)
            for i, _ in enumerate(preds[token]['traffic_element']):
                preds[token]['traffic_element'][i]['confidence'] = np.float32(1)
        else:
            preds[token] = predictions[token]['predictions']

    assert set(gts.keys()) == set(preds.keys()), '#frame differs'

    """
        calculate distances between gts and preds    
    """

    distance_matrices = {
        'laneseg': {},
        'area': {},
    }

    for token in tqdm(gts.keys(), desc='calculating distances:', ncols=80, disable=not verbose):

        mask = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance_c,
            relax=True,
        ) < THRESHOLDS_LANESEG[-1]

        distance_matrices['laneseg'][token] = pairwise(
            [gt for gt in gts[token]['lane_segment']],
            [pred for pred in preds[token]['lane_segment']],
            lane_segment_distance,
            mask=mask,
            relax=True,
        )

        distance_matrices['area'][token] = pairwise(
            [gt for gt in gts[token]['area']],
            [pred for pred in preds[token]['area']],
            area_distance,
        )

    """
        evaluate
    """

    metrics = {
        'mAP': 0
    }

    metrics['AP_ls'] = _mAP_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['laneseg'], 
        distance_thresholds=THRESHOLDS_LANESEG,
        object_type='lane_segment',
        filter=lambda _: True,
        inject=True, # save tp for eval on graph
    ).mean()

    metrics['AP_ped'] = _mAP_over_threshold(
        gts=gts, 
        preds=preds, 
        distance_matrices=distance_matrices['area'], 
        distance_thresholds=THRESHOLDS_AREA, 
        object_type='area',
        filter=lambda x: x['category'] == 1,
        inject=False,
    ).mean()

    metrics['TOP_lsls'] = _mAP_topology_lsls(gts, preds, THRESHOLDS_LANESEG)
    metrics['mAP_type'] =   _mAP_type_lsls(gts, preds, THRESHOLDS_LANESEG)
    metrics['mAP'] = (metrics['AP_ls'] + metrics['AP_ped']) / 2

    return metrics
