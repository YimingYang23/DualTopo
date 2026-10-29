"""Run every validation frame in native chronological order, preserving scenes."""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dualtopo_ratopo.data import build_dataset, prepare
from dualtopo_ratopo.model import build_model, inference_session, predict, sha256


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if value.dtype.kind in 'fc' else True
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--branch', choices=['singleframe', 'temporal'], default='temporal')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-annotation-mismatch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output path; complete predictions are never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(args.weights, args.device)
    dataset = build_dataset(args.data_root, args.annotations, args.allow_annotation_mismatch)
    results, previous_scene, previous_index = [], None, None
    start = time.time()
    with inference_session(model, args.branch):
        for index in range(len(dataset)):
            data = prepare(dataset, index, args.device)
            bev, prediction, trace = predict(model, data, args.branch)
            scene = data['img_metas'][0]['scene_token']
            expected_history = args.branch == 'temporal' and scene == previous_scene and index == previous_index + 1
            if trace['has_history'] != [expected_history] or not finite(prediction) or not finite(bev):
                raise ValueError('Non-finite inference or invalid chronological history')
            results.append(prediction[0])
            previous_scene, previous_index = scene, index
            if index < 3 or (index + 1) % 100 == 0:
                print(json.dumps(dict(index=index, samples=index + 1, branch=args.branch, has_history=trace['has_history'])), flush=True)
    formatted = dataset.format_results(results)
    if len(formatted['results']) != len(dataset) or not finite(formatted):
        raise ValueError('Incomplete or non-finite native prediction formatting')
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('wb') as stream:
        pickle.dump(formatted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    receipt = dict(samples=len(dataset), indices=list(range(len(dataset))), branch=args.branch,
                   predictions_sha256=sha256(output), checkpoint_sha256=sha256(args.weights), annotation_sha256=sha256(args.annotations),
                   finite=True, elapsed_seconds=time.time() - start)
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
