# SPDX-License-Identifier: Apache-2.0
# Data conversion adapted from OpenDriveLab/LaneSegNet and OpenLane-V2.
"""Prepare validation annotations for this inference project."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/openlane'))
from openlanev2.lanesegment.io import io

def _fix_pts_interpolate(curve, n_points):
    ls = LineString(curve)
    distances = np.linspace(0, ls.length, n_points)
    curve = np.array([ls.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)
    return curve

def collect(root_path : str, data_dict : dict, collection : str, n_points : dict) -> None:

    data_list = [(split, segment_id, timestamp.split('.')[0]) \
        for split, segment_ids in data_dict.items() \
            for segment_id, timestamps in segment_ids.items() \
                for timestamp in timestamps
    ]
    meta = {}
    for split, segment_id, timestamp in tqdm(data_list, desc=f'collecting {collection}', ncols=100):
        identifier = (split, segment_id, timestamp)
        frame = io.json_load(f'{root_path}/{split}/{segment_id}/info/{timestamp}-ls.json')

        for k, v in frame['pose'].items():
            frame['pose'][k] = np.array(v, dtype=np.float64)
        for camera in frame['sensor'].keys():
            for para in ['intrinsic', 'extrinsic']:
                for k, v in frame['sensor'][camera][para].items():
                    frame['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

        if 'annotation' not in frame:
            meta[identifier] = frame
            continue

        # NOTE: We don't interpolate the points for ped crossing and road bouadary.
        for i, area in enumerate(frame['annotation']['area']):
            frame['annotation']['area'][i]['points'] = np.array(area['points'], dtype=np.float32)
        for i, lane_segment in enumerate(frame['annotation']['lane_segment']):
            frame['annotation']['lane_segment'][i]['centerline'] = _fix_pts_interpolate(np.array(lane_segment['centerline']), n_points['centerline'])
            frame['annotation']['lane_segment'][i]['left_laneline'] = _fix_pts_interpolate(np.array(lane_segment['left_laneline']), n_points['left_laneline'])
            frame['annotation']['lane_segment'][i]['right_laneline'] = _fix_pts_interpolate(np.array(lane_segment['right_laneline']), n_points['right_laneline'])
        for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
            frame['annotation']['traffic_element'][i]['points'] = np.array(traffic_element['points'], dtype=np.float32)
        frame['annotation']['topology_lsls'] = np.array(frame['annotation']['topology_lsls'], dtype=np.int8)
        frame['annotation']['topology_lste'] = np.array(frame['annotation']['topology_lste'], dtype=np.int8)
        meta[identifier] = frame

    return meta

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; use that file or choose another output path')
    index=args.index or args.data_root/'data_dict_subset_A.json'
    segments=json.loads(index.read_text())['val']
    if sum(map(len,segments.values()))!=4806:parser.error('Expected the 4,806-frame SubsetA validation index')
    metadata=collect(str(args.data_root),{'val':segments},'validation',{'centerline':10,'left_laneline':10,'right_laneline':10})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:pickle.dump(metadata,stream,protocol=4)
    print(str(args.output.resolve()))
if __name__=='__main__':main()
