"""Run the full lane-segment inference/scoring workflow in its PyTorch environment."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint');p.add_argument('--device',default='cuda:0');p.add_argument('--check-images',action='store_true');p.add_argument('--branches',nargs='+',choices=['init','temporal'],default=['init','temporal'])
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);stages=[]
    def run(name,args):
        r=subprocess.run([sys.executable,str(ROOT/'tools'/name)]+args)
        stages.append(dict(tool=name,executed=True,wait=r.returncode));r.check_returncode()
    run('verify_release.py',[])
    run('check_data.py',['--annotations',a.annotations,'--data-root',a.data_root]+(['--check-images'] if a.check_images else []))
    manifest=json.loads((ROOT/'configs/weights.json').read_text());ck=Path(a.checkpoint) if a.checkpoint else ROOT/'weights'/manifest['filename']
    if not ck.exists():
        if a.checkpoint:raise FileNotFoundError(str(ck))
        run('download_weights.py',['--directory',str(ck.parent)])
    run('verify_release.py',['--weights',str(ck)])
    ann_sha=sha(a.annotations);spec=json.loads((ROOT/'evaluator.json').read_text());metric_outputs={}
    def check_saved_metric(path,pred):
        saved=json.loads(path.read_text())
        assert saved['prediction_sha256']==sha(pred) and saved['annotation_sha256']==ann_sha and saved['frames']==4806
        return saved.get('metric_protocol')==spec['protocol'] and saved.get('evaluator_manifest_sha256')==sha(ROOT/'evaluator.json') and set(saved.get('metrics',{}))==set(spec['metrics'])
    for branch in a.branches:
        pred=out/(branch+'.pkl');metric=out/(branch+'_metrics.json')
        if pred.exists():
            receipt=json.loads(pred.with_suffix('.json').read_text())
            assert receipt['frames']==4806 and receipt['branch']==branch and receipt['checkpoint_sha256']==sha(ck) and receipt['annotation_sha256']==ann_sha and receipt['predictions_sha256']==sha(pred)
            stages.append(dict(tool='infer.py',branch=branch,executed=False,verified_saved_predictions=True))
        else:run('infer.py',['--checkpoint',str(ck),'--data-root',a.data_root,'--annotations',a.annotations,'--output',str(pred),'--branch',branch,'--device',a.device])
        if metric.exists() and not check_saved_metric(metric,pred):
            metric=out/(branch+'_lanesegnet_metrics.json')
        if metric.exists():
            if not check_saved_metric(metric,pred):raise ValueError('Saved metrics use a different evaluator; score saved predictions with tools/evaluate.py and a new --output path')
            stages.append(dict(tool='evaluate.py',branch=branch,executed=False,verified_saved_metrics=True))
        else:
            run('evaluate.py',['--predictions',str(pred),'--annotations',a.annotations,'--output',str(metric)])
            if not check_saved_metric(metric,pred):raise ValueError('Scoring did not produce the configured LaneSegNet metrics')
        metric_outputs[branch]=str(metric)
    (out/'reproduction.json').write_text(json.dumps(dict(completed=True,task='lane_segment',subset='A',frames_per_branch=4806,branches=a.branches,checkpoint_sha256=sha(ck),annotation_sha256=ann_sha,metric_protocol=spec['protocol'],metrics_files=metric_outputs,stages=stages),indent=2))
if __name__=='__main__':main()
