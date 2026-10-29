"""Download this project's safetensors checkpoint and verify its SHA256."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'weights.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(root / 'weights' / 'model.safetensors'))
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/{repo}/resolve/{revision}/{filename}'.format(repo=manifest['repo_id'], revision=manifest.get('revision', 'main'), filename=manifest['filename'])
    if not target.exists():
        temporary = target.with_suffix('.partial')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('Weight checksum mismatch; do not load this file')
    print(str(target.resolve()))


if __name__ == '__main__':
    main()
