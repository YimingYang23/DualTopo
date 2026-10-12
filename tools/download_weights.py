"""Download the SubsetA inference checkpoint from the official model repository."""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from dualtopo.checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("weights"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    manifest = __import__("json").loads(
        (Path(__file__).resolve().parents[1] / "configs/weights.json").read_text()
    )
    path = hf_hub_download(
        repo_id="y352184741/DualTopo",
        filename=manifest["filename"],
        local_dir=args.directory,
        revision=args.revision,
    )
    if sha256(path) != manifest["sha256"]:
        raise RuntimeError("Downloaded checkpoint does not match the release manifest")
    print(path)


if __name__ == "__main__":
    main()
