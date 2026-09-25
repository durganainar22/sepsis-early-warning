"""Fetch the PhysioNet/CinC Challenge 2019 training data (sepsis early prediction).

Source: https://physionet.org/content/challenge-2019/1.0.0/  (CC BY 4.0, open access)
Reyna MA et al. "Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in
Cardiology Challenge 2019." Critical Care Medicine 48(2):210-217, 2020.

40,336 patients, one pipe-delimited .psv file each, in two folders that are two different
hospital systems - which is what makes a train-on-A / test-on-B design possible at all.

Why the S3 mirror and not physionet.org: PhysioNet offers no single archive for this
project, and ~40k small files over one HTTP connection at a time is slow. The public
physionet-open bucket serves the same files, lists them with sizes and MD5 ETags, and takes
parallel requests - so every file is also VERIFIED, not just downloaded.

Raw data is gitignored; this script is what makes the repo reproducible from a clean clone.

Usage:
    python src/download_data.py [--workers 16] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training/"
SETS = {"training_setA": 20_336, "training_setB": 20_000}  # counts from the dataset page

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix: str) -> list[dict]:
    """Every .psv key under prefix, with size and MD5, following S3 pagination."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            key = c.find("s3:Key", NS).text
            if key.endswith(".psv"):
                out.append({"key": key, "size": int(c.find("s3:Size", NS).text),
                            # For single-part uploads the ETag IS the MD5 of the file.
                            "md5": c.find("s3:ETag", NS).text.strip('"')})
        if root.find("s3:IsTruncated", NS).text != "true":
            return out
        token = root.find("s3:NextContinuationToken", NS).text


def fetch(obj: dict, force: bool) -> str:
    dest = RAW_DIR / obj["key"][len(PREFIX):]
    if dest.exists() and not force and dest.stat().st_size == obj["size"]:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{obj['key']}", timeout=60) as r:
                data = r.read()
            if hashlib.md5(data).hexdigest() != obj["md5"]:
                raise ValueError("MD5 mismatch")
            dest.write_bytes(data)
            return "downloaded"
        except Exception as e:  # network blips are expected over ~40k requests
            if attempt == 3:
                raise RuntimeError(f"{obj['key']}: {e}") from e
            time.sleep(2 ** attempt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="re-download files already present")
    args = ap.parse_args()

    objs = []
    for s, expected in SETS.items():
        found = list_objects(PREFIX + s + "/")
        print(f"{s}: {len(found):,} files listed (dataset page says {expected:,})")
        if len(found) != expected:
            print(f"  ERROR: count mismatch for {s}", file=sys.stderr)
            return 1
        objs += found

    t0, counts = time.perf_counter(), {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, o, args.force) for o in objs]
        for i, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if i % 5000 == 0 or i == len(objs):
                print(f"  {i:>6,}/{len(objs):,}  {time.perf_counter() - t0:5.0f}s", flush=True)

    total_mb = sum(o["size"] for o in objs) / 1e6
    print(f"Done: {counts['downloaded']:,} downloaded, {counts['skipped']:,} already present, "
          f"{total_mb:.1f} MB, every file MD5-verified -> {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
