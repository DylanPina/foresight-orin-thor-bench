#!/usr/bin/env python3
"""Pre-download a CLIP model to the workspace checkpoints directory.

Run this once (inside the container) so the lelan_planner_node does not need
internet access at startup:

    python scripts/download_clip.py
    python scripts/download_clip.py --model ViT-B/32 --dest checkpoints
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download CLIP weights.")
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="CLIP model name, e.g. ViT-B/32 (default) or ViT-L/14.",
    )
    parser.add_argument(
        "--dest",
        default="checkpoints",
        help="Directory to download into (default: ./checkpoints).",
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    try:
        import clip  # type: ignore
    except ImportError:
        print(
            "ERROR: openai-clip is not installed. "
            "Run: pip install git+https://github.com/openai/CLIP.git",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading CLIP '{args.model}' → {dest} ...")
    clip.load(args.model, download_root=dest)
    print("Done. The weight file is now in:", dest)
    for f in os.listdir(dest):
        fp = os.path.join(dest, f)
        if os.path.isfile(fp) and f.endswith(".pt"):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  {f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
