"""Verify this isolated project's source files and optional checkpoint."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights');p.add_argument('--subset',choices=['A','B','a','b'])
    a=p.parse_args();project=json.loads((ROOT/'project.json').read_text())
    files=json.loads((ROOT/'source_manifest.json').read_text())
    for name,expected in files.items():
        if sha(ROOT/name)!=expected:raise ValueError('Source checksum mismatch: '+name)
    if a.weights:
        if project['implementation']=='mmcv':
            if not a.subset:p.error('--subset A or B is required to check RATopo weights')
            manifest=json.loads((ROOT/'backends'/('subset_'+a.subset.lower())/'weights.json').read_text())
        else:manifest=json.loads((ROOT/'configs/weights.json').read_text())
        if sha(a.weights)!=manifest['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
        if project['implementation']=='mmcv':
            import numpy as np
            from safetensors.numpy import load_file
            state=load_file(a.weights)
            assert len(state)==manifest['tensor_count'] and all(np.isfinite(x).all() for x in state.values())
        else:
            import torch
            obj=torch.load(a.weights,map_location='cpu',weights_only=True)
            state=obj.get('state_dict',obj)
            assert len(state)==manifest['tensor_count']
            assert all(torch.is_tensor(x) and (not x.is_floating_point() or bool(torch.isfinite(x).all())) for x in state.values())
    print('Verified',len(files),'source files'+(' and checkpoint integrity' if a.weights else ''))
if __name__=='__main__':main()
