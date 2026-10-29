# SPDX-License-Identifier: Apache-2.0
# SubsetB conversion adapted from OpenDriveLab/OpenLane-V2.
"""Download SubsetA validation metadata or prepare SubsetB from raw annotations."""
import argparse
import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def download_subset_a(output):
    manifest = json.loads((ROOT / 'configs/data.json').read_text())['A']
    temporary = output.with_suffix(output.suffix + '.partial')
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(manifest['url']) as source, temporary.open('wb') as target:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            target.write(block)
            digest.update(block)
            size += len(block)
    if digest.hexdigest() != manifest['sha256'] or size != manifest['bytes']:
        raise ValueError('Annotation download is incomplete; run the command again')
    temporary.replace(output)


def prepare_subset_b(data_root, index):
    segments = json.loads(index.read_text())['val']
    frames = [('val', segment, str(timestamp).split('.')[0])
              for segment, timestamps in segments.items() for timestamp in timestamps]
    if len(frames) != 6019:
        raise ValueError('Expected the 6,019-frame SubsetB validation index')
    metadata = {}
    for split, segment, timestamp in tqdm(frames, desc='Preparing SubsetB'):
        frame = json.loads((data_root / split / segment / 'info' / (timestamp + '.json')).read_text())
        for key, value in frame['pose'].items():
            frame['pose'][key] = np.array(value, dtype=np.float64)
        for sensor in frame['sensor'].values():
            for kind in ['intrinsic', 'extrinsic']:
                for key, value in sensor[kind].items():
                    sensor[kind][key] = np.array(value, dtype=np.float64)
        annotation = frame['annotation']
        for lane in annotation['lane_centerline']:
            lane['points'] = np.array(lane['points'][::20], dtype=np.float32)
        for traffic in annotation['traffic_element']:
            traffic['points'] = np.array(traffic['points'], dtype=np.float32)
        for key in ['topology_lclc', 'topology_lcte']:
            annotation[key] = np.array(annotation[key], dtype=np.int8)
        metadata[(split, segment, timestamp)] = frame
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', required=True, choices=['A', 'B', 'a', 'b'])
    parser.add_argument('--data-root', type=Path, help='Raw SubsetB dataset directory')
    parser.add_argument('--index', type=Path, help='Optional SubsetB split-index path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use that file or choose another output path')
    if args.subset.upper() == 'B' and args.data_root is None:
        parser.error('SubsetB requires --data-root')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.subset.upper() == 'A':
        download_subset_a(args.output)
    else:
        index = args.index or args.data_root / 'data_dict_subset_B.json'
        metadata = prepare_subset_b(args.data_root, index)
        with args.output.open('xb') as target:
            pickle.dump(metadata, target, protocol=4)
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
