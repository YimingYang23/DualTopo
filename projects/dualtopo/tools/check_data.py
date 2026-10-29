"""Check task-specific OpenLane-V2 metadata before full sequential inference."""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def inspect_data(annotations,data_root,subset,task,check_images=False):
    with Path(annotations).open('rb') as f:raw=pickle.load(f)
    infos=list(raw.values()) if isinstance(raw,dict) else list(raw)
    expected=4806 if subset=='A' else 6019
    if len(infos)!=expected:raise ValueError('Expected the full validation split: '+str(expected))
    previous=None;stamp=None;seen=set();scenes=0;images=0
    cameras=7 if subset=='A' else 6
    for i,info in enumerate(infos):
        scene=info['segment_id'];timestamp=int(info['timestamp'])
        if scene!=previous:
            if scene in seen:raise ValueError('Interleaved scenes')
            seen.add(scene);scenes+=1;stamp=None
        if stamp is not None and timestamp<=stamp:raise ValueError('Non-chronological frames')
        if len(info['sensor'])!=cameras:raise ValueError('Wrong subset camera count')
        annotation=info['annotation']
        if task=='lane_segment':
            if 'lane_segment' not in annotation:raise ValueError('DualTopo needs LaneSegNet lane-segment metadata, not centerline metadata')
            for lane in annotation['lane_segment']:
                for key in ['centerline','left_laneline','right_laneline']:
                    points=np.asarray(lane[key])
                    if points.shape!=(10,3) or not np.isfinite(points).all():raise ValueError('LaneSegNet converter must provide 10 XYZ points for each line')
        elif 'lane_centerline' not in annotation:raise ValueError('RATopo needs native centerline metadata, not the LaneSegNet lane-segment pickle')
        if check_images or i==0:
            for camera in info['sensor'].values():
                image=Path(data_root)/camera['image_path']
                if not image.is_file():raise FileNotFoundError(str(image))
                images+=1
        previous,stamp=scene,timestamp
    return dict(task=task,subset=subset,frames=len(infos),scenes=scenes,cameras=cameras,checked_images=images,all_images_checked=check_images,annotation_sha256=hashlib.sha256(Path(annotations).read_bytes()).hexdigest())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotations',required=True);p.add_argument('--data-root',required=True);p.add_argument('--subset',choices=['A','B','a','b']);p.add_argument('--check-images',action='store_true')
    a=p.parse_args();project=json.loads((ROOT/'project.json').read_text())
    subset=(a.subset or 'A').upper()
    if project['implementation']=='mmcv' and not a.subset:p.error('RATopo requires --subset A or B')
    if subset not in project['subsets']:p.error('Unsupported subset for this project')
    print(json.dumps(inspect_data(a.annotations,a.data_root,subset,project['task'],a.check_images),indent=2))
if __name__=='__main__':main()
