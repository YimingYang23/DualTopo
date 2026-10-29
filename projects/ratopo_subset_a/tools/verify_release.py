"""Verify release source files and optionally the data-only model download."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights');a=p.parse_args()
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,expected in manifest.items():
        if sha(ROOT/name)!=expected:raise ValueError('Source checksum mismatch: '+name)
    if a.weights:
        weights=json.loads((ROOT/'weights.json').read_text())
        if sha(Path(a.weights))!=weights['sha256']:raise ValueError('Weight checksum mismatch')
        from safetensors.numpy import load_file
        import numpy as np
        tensors=load_file(a.weights)
        if len(tensors)!=weights['tensor_count']:raise ValueError('Weight key count mismatch')
        if any(not np.isfinite(v).all() for v in tensors.values()):raise ValueError('Non-finite weights')
    print('Verified',len(manifest),'source files'+(' and finite model tensors' if a.weights else ''))
if __name__=='__main__':main()
