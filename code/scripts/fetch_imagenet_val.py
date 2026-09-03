"""Fetch the ImageNet-1k validation split (50,000 images) to a flat directory.

Three routes exist; this script automates the one with no approval wait.

  1. image-net.org -- register, accept the terms, download
     ILSVRC2012_img_val.tar (~6.3 GB), untar.  Access is granted by a human,
     so allow for a delay.  Nothing else is needed: the measurement pipeline
     reads a flat directory and never touches the labels, so the devkit and
     the usual valprep reorganisation step are both unnecessary.

  2. Hugging Face `ILSVRC/imagenet-1k` -- gated, but the gate is an instant
     click-through on any account rather than a review.  The validation split
     is a handful of parquet shards that can be fetched on their own without
     the 150 GB training split.  THIS IS WHAT THIS SCRIPT DOES.

  3. Academic Torrents hosts the same validation tar.  Same data, same terms
     of use, no account.

Whichever route: the data is under the ImageNet terms of access, which permit
non-commercial research use only.

Usage:
    huggingface-cli login          # once, after accepting the gate in a browser
    python scripts/fetch_imagenet_val.py --out /data/imagenet_val

Then:
    python -m layerspec.run --data /data/imagenet_val --out results/
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="directory to write JPEGs into")
    p.add_argument("--repo", default="ILSVRC/imagenet-1k")
    p.add_argument("--pattern", default="data/validation-*",
                   help="shard glob; the default is the validation split only. "
                        "Verified 2026-09-03 against the repo: the shards are "
                        "named data/validation-000NN-of-00014.parquet")
    p.add_argument("--cache", default=None, help="parquet download cache dir")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many images (for a trial run)")
    p.add_argument("--keep-parquet", action="store_true",
                   help="do not delete the downloaded shards afterwards")
    args = p.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as e:
        print(f"missing dependency: {e}\n"
              f"  pip install huggingface_hub pyarrow pillow", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)

    print(f"downloading {args.pattern} from {args.repo}")
    print("(if this 401s, accept the dataset terms in a browser first, "
          "then run: huggingface-cli login)")
    t0 = time.time()
    local = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=args.pattern,
        cache_dir=args.cache,
    )
    print(f"  shards in {local}  ({time.time() - t0:.0f}s)")

    shards = []
    for root, _, files in os.walk(local):
        shards += [os.path.join(root, f) for f in files if f.endswith(".parquet")]
    shards.sort()
    if not shards:
        # Naming in these repos does change.  Show what is actually there
        # rather than making the user guess at the glob.
        print(f"no parquet shards matched {args.pattern!r}", file=sys.stderr)
        try:
            from huggingface_hub import list_repo_files
            available = list_repo_files(args.repo, repo_type="dataset")
            parquet = [f for f in available if f.endswith(".parquet")]
            print("\navailable parquet files in the repo (first 10):",
                  file=sys.stderr)
            for f in parquet[:10]:
                print(f"  {f}", file=sys.stderr)
            print(f"  ... {len(parquet)} total\n"
                  f"pass a matching glob with --pattern", file=sys.stderr)
        except Exception as e:
            print(f"(could not list repo files: {e})", file=sys.stderr)
        return 1
    print(f"  {len(shards)} shards")

    written = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["image"])
        col = table.column("image").to_pylist()
        for rec in col:
            # The image column is a struct of {bytes, path}; fall back to raw
            # bytes if a future revision changes the layout.
            raw = rec.get("bytes") if isinstance(rec, dict) else rec
            if raw is None:
                continue
            name = None
            if isinstance(rec, dict):
                name = rec.get("path")
            name = os.path.basename(name) if name else f"val_{written:08d}.JPEG"

            dest = os.path.join(args.out, name)
            if not os.path.exists(dest):
                # Re-encode rather than writing raw bytes: this normalises the
                # CMYK and greyscale JPEGs that ImageNet's validation set
                # contains, so the measurement run does not trip over them.
                Image.open(io.BytesIO(raw)).convert("RGB").save(
                    dest, format="JPEG", quality=95
                )
            written += 1
            if written % 5000 == 0:
                print(f"  {written} images", flush=True)
            if args.limit and written >= args.limit:
                break
        if args.limit and written >= args.limit:
            break

    print(f"\nwrote {written} images to {args.out}")
    if written not in (50000,) and not args.limit:
        print(f"!! expected 50000 validation images, got {written} -- "
              f"check the shard pattern before using this for the paper")

    if not args.keep_parquet:
        print("parquet shards left in the HF cache; "
              "`huggingface-cli delete-cache` to reclaim the space")

    print(f"\nnext:\n  python -m layerspec.run --data {args.out} --out results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
