"""Run one subset's full centerline workflow in the separate MMCV environment."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subset',required=True,choices=['A','B','a','b']);p.add_argument('--data-root',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--weights');p.add_argument('--device',default='cuda:0');p.add_argument('--check-images',action='store_true');p.add_argument('--branches',nargs='+',choices=['singleframe','temporal'],default=['singleframe','temporal'])
    a=p.parse_args();subset=a.subset.upper();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);frames=4806 if subset=='A' else 6019;stages=[]
    def run(name,args):
        r=subprocess.run([sys.executable,str(ROOT/'tools'/name)]+args)
        stages.append(dict(tool=name,executed=True,wait=r.returncode));r.check_returncode()
    run('verify_release.py',[])
    run('check_data.py',['--subset',subset,'--annotations',a.annotations,'--data-root',a.data_root]+(['--check-images'] if a.check_images else []))
    ck=Path(a.weights) if a.weights else ROOT/'weights'/('ratopo_subset_'+subset.lower()+'.safetensors')
    if not ck.exists():
        if a.weights:raise FileNotFoundError(str(ck))
        run('download_weights.py',['--subset',subset,'--output',str(ck)])
    run('verify_release.py',['--subset',subset,'--weights',str(ck)])
    ann_sha=sha(a.annotations)
    for branch in a.branches:
        pred=out/(branch+'.pkl');metric=out/(branch+'_metrics.json')
        if pred.exists():
            receipt=json.loads(pred.with_suffix('.pkl.json').read_text())
            assert receipt['samples']==frames and receipt['branch']==branch and receipt['checkpoint_sha256']==sha(ck) and receipt['annotation_sha256']==ann_sha and receipt['predictions_sha256']==sha(pred)
            stages.append(dict(tool='infer.py',branch=branch,executed=False,verified_saved_predictions=True))
        else:run('infer.py',['--subset',subset,'--weights',str(ck),'--data-root',a.data_root,'--annotations',a.annotations,'--output',str(pred),'--branch',branch,'--device',a.device])
        if metric.exists():
            saved=json.loads(metric.read_text());assert saved['prediction_sha256']==sha(pred) and saved['annotation_sha256']==ann_sha and saved['samples']==frames
            stages.append(dict(tool='evaluate.py',branch=branch,executed=False,verified_saved_metrics=True))
        else:run('evaluate.py',['--subset',subset,'--predictions',str(pred),'--annotations',a.annotations,'--data-root',a.data_root,'--output',str(metric)])
    (out/'reproduction.json').write_text(json.dumps(dict(completed=True,task='centerline',subset=subset,frames_per_branch=frames,branches=a.branches,checkpoint_sha256=sha(ck),annotation_sha256=ann_sha,stages=stages),indent=2))
if __name__=='__main__':main()
