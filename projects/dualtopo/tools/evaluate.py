"""Full SubsetA lane-segment scoring with the vendored native evaluator."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,inspect,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openlane')]
import numpy as np
from dualtopo.dataset import fix_pts_interpolate
from dualtopo.evaluation import lanesegnet_evaluate as evaluate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def format_gt(infos,split='val'):
    """Exact native LaneSegNet GT formatting: pedestrian-area boundaries."""
    output={}
    for item in infos:
        info=copy.deepcopy(item);areas=[]
        for area in info['annotation']['area']:
            if area['category']==1:
                points=np.asarray(area['points'])
                left=fix_pts_interpolate(points[[0,1]],10)
                right=fix_pts_interpolate(points[[2,3]],10)
                area['points']=np.concatenate([left,right],axis=0);areas.append(area)
        info['annotation']['area']=areas
        output[(split,info['segment_id'],str(info['timestamp']))]=info
    return output
def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='val');p.add_argument('--limit',type=int,help='Score the first N validation frames')
    a=p.parse_args()
    if a.limit is not None and not 1<=a.limit<=4806:p.error('--limit must be between 1 and 4806')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new metrics path')
    with Path(a.annotations).open('rb') as f:raw=pickle.load(f)
    infos=list(raw.values()) if isinstance(raw,dict) else list(raw)
    if len(infos)!=4806 or any('lane_segment' not in x['annotation'] for x in infos):raise ValueError('Requires all 4,806 native LaneSegNet-format SubsetA frames')
    if a.limit is not None:infos=infos[:a.limit]
    with Path(a.predictions).open('rb') as f:pred=pickle.load(f)
    gt=format_gt(infos,a.split)
    if set(gt)!=set(pred['results']):raise ValueError('Predictions must cover the entire GT split exactly once')
    spec=json.loads((ROOT/'evaluator.json').read_text())
    for name,expected in spec['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Native evaluator/helper source checksum changed: '+name)
    metrics=evaluate(gt,pred)
    result=dict(task='lane_segment',subset='A',frames=len(gt),full_validation_split=len(gt)==4806,metrics=metrics,metric_protocol=spec['protocol'],evaluator_manifest_sha256=sha(ROOT/'evaluator.json'),prediction_sha256=sha(a.predictions),annotation_sha256=sha(a.annotations),evaluator_sha256=sha(inspect.getsourcefile(evaluate)),GT_protocol='Native LaneSegNet pedestrian-area formatting; centerline-task GT normalization is not applied; no frame removal',traffic_detector_in_checkpoint=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,default=encode,allow_nan=False));print(json.dumps(metrics,indent=2,default=encode,allow_nan=False))
if __name__=='__main__':main()
