"""CPU OpenLane-V2 centerline scoring of complete saved native predictions."""
import argparse
import copy
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dualtopo_ratopo.data import build_dataset
from dualtopo_ratopo.model import ROOT, sha256
from openlanev2.centerline.evaluation import evaluate


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-annotation-mismatch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new metrics output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / 'results.json').read_text())
    dataset = build_dataset(args.data_root, args.annotations, args.allow_annotation_mismatch)
    with Path(args.predictions).open('rb') as stream:
        # Only open prediction files produced by tools/infer.py or another trusted source.
        predictions = pickle.load(stream)
    ground_truth = copy.deepcopy(dataset.format_openlanev2_gt())
    expected = {(dataset.split, item['segment_id'], str(item['timestamp'])) for item in dataset.data_infos}
    if set(ground_truth) != expected or set(predictions['results']) != expected:
        raise ValueError('Scoring requires every native frame exactly once')
    changes, traffic_elements = [], 0
    for token, frame in ground_truth.items():
        for traffic_element in frame['annotation']['traffic_element']:
            points = np.asarray(traffic_element['points'])
            traffic_elements += 1
            if not np.isfinite(points).all():
                raise ValueError('Non-finite traffic-element GT')
            if points.shape != (2, 2):
                if protocol['subset'] != 'A' or points.shape != (3, 2):
                    raise ValueError('Unexpected GT shape; do not silently change the benchmark')
                bbox = np.stack([points.min(axis=0), points.max(axis=0)])
                changes.append(dict(token=token, id=traffic_element.get('id', traffic_element.get('ID')), original_points=points, bbox=bbox))
                traffic_element['points'] = bbox
    if len(changes) != protocol['GT_protocol']['changed_traffic_elements'] or traffic_elements != protocol['GT_protocol']['traffic_elements']:
        raise ValueError('GT preprocessing does not match the published evaluation protocol')
    preprocessing = dict(subset=protocol['subset'], frames=len(ground_truth), traffic_elements=traffic_elements, changes=changes,
                         original_annotation_unchanged=True, official_evaluator_unchanged=True)
    preprocessing_path = output.with_suffix('.GT_preprocessing.json')
    preprocessing_path.write_text(json.dumps(preprocessing, indent=2, default=json_value))
    evaluator = Path(inspect.getsourcefile(evaluate)).resolve()
    evaluator_expected = json.loads((ROOT / 'evaluator.json').read_text())
    if sha256(evaluator) != evaluator_expected['sha256']:
        raise ValueError('Official evaluator source differs from this release')
    metrics = evaluate(ground_truth, predictions)
    score = metrics['OpenLane-V2 Score']
    record = dict(subset=protocol['subset'], samples=len(ground_truth), raw_metrics=metrics,
                  percent_metrics={name: 100 * score[key] for name, key in [('OLS', 'score'), ('DET_l', 'DET_l'), ('DET_t', 'DET_t'), ('TOP_ll', 'TOP_ll'), ('TOP_lt', 'TOP_lt')]},
                  prediction_sha256=sha256(args.predictions), annotation_sha256=sha256(args.annotations), evaluator_sha256=sha256(evaluator),
                  GT_preprocessing_sha256=sha256(preprocessing_path), CUDA_initialized=torch.cuda.is_initialized())
    if record['CUDA_initialized'] or not all(np.isfinite(value) for value in record['percent_metrics'].values()):
        raise ValueError('Scoring must stay on CPU and return finite metrics')
    output.write_text(json.dumps(record, indent=2, default=json_value))
    print(json.dumps(record['percent_metrics'], indent=2))


if __name__ == '__main__':
    main()
