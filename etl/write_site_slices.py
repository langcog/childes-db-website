#!/usr/bin/env python3
"""Write per-collection data slices for the childes-db site.

Reads the staged parquet export of a childes-db release and writes:

  data/collections.json                small: selector options + headline stats
  data/corpora.json                    small: corpus -> collection mapping
  slices/speaker_stats/<slug>.json     one per collection (by name), from
                                       transcript_by_speaker; column-oriented
                                       JSON, lazily fetched by the viz pages

Ages are stored in the database in DAYS; slices carry months
(days / (365.2425 / 12), matching childesr), rounded to 0.1.
Rows with no target-child age are dropped (they cannot be plotted).

Usage: python etl/write_site_slices.py [staging_dir]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

STAGING = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/Users/mcfrank/Projects/childes-db/pipeline/parquet_compact"
)
SITE = Path(__file__).resolve().parent.parent

# release version shipped in data/stats.json (the staging dir is no longer
# named after the release, so it is recorded here)
VERSION = "2026.1"

DAYS_PER_MONTH = 365.2425 / 12

# columns shipped in each speaker_stats slice, in order
SLICE_COLS = [
    "transcript_id", "corpus_name", "target_child_id", "target_child_name",
    "target_child_sex", "speaker_role", "age", "num_utterances", "num_tokens",
    "num_types", "num_morphemes", "mlu_w", "mlu_m", "mtld", "hdd",
]


def slug(name):
    return re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", name.lower()))


def rnd(x, digits):
    return None if x is None else round(x, digits)


def main():
    # each table is a directory of parquet part files
    tbs = pq.read_table(STAGING / "transcript_by_speaker")
    corpus = pq.read_table(STAGING / "corpus")
    transcript = pq.read_table(STAGING / "transcript")

    corpus_name = dict(zip(corpus.column("id").to_pylist(),
                           corpus.column("name").to_pylist()))

    # ---- speaker_stats slices: one per collection name --------------------
    rows = tbs.to_pylist()
    by_collection = defaultdict(list)
    for r in rows:
        if r["target_child_age"] is None:
            continue
        by_collection[r["collection_name"]].append(r)

    out_dir = SITE / "slices" / "speaker_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, coll_rows in sorted(by_collection.items()):
        cols = {c: [] for c in SLICE_COLS}
        for r in coll_rows:
            cols["transcript_id"].append(r["transcript_id"])
            cols["corpus_name"].append(corpus_name.get(r["corpus_id"]))
            cols["target_child_id"].append(r["target_child_id"])
            cols["target_child_name"].append(r["target_child_name"])
            cols["target_child_sex"].append(r["target_child_sex"])
            cols["speaker_role"].append(r["speaker_role"])
            cols["age"].append(rnd(r["target_child_age"] / DAYS_PER_MONTH, 1))
            cols["num_utterances"].append(r["num_utterances"])
            cols["num_tokens"].append(r["num_tokens"])
            cols["num_types"].append(r["num_types"])
            cols["num_morphemes"].append(r["num_morphemes"])
            cols["mlu_w"].append(rnd(r["mlu_w"], 3))
            cols["mlu_m"].append(rnd(r["mlu_m"], 3))
            cols["mtld"].append(rnd(r["mtld"], 2))
            cols["hdd"].append(rnd(r["hdd"], 4))
        path = out_dir / f"{slug(name)}.json"
        path.write_text(json.dumps(cols, separators=(",", ":")))
        print(f"wrote {path.relative_to(SITE)} ({len(coll_rows)} rows, "
              f"{path.stat().st_size / 1e6:.1f} MB)")

    # ---- small embedded tables -------------------------------------------
    data_dir = SITE / "data"
    data_dir.mkdir(exist_ok=True)

    tr = transcript.to_pylist()
    tr_by_coll = defaultdict(list)
    for t in tr:
        tr_by_coll[t["collection_name"]].append(t)

    collections = []
    for name, coll_rows in sorted(by_collection.items()):
        trs = tr_by_coll.get(name, [])
        collections.append({
            "name": name,
            "slug": slug(name),
            "n_corpora": len({r["corpus_id"] for r in coll_rows}),
            "n_transcripts": len({t["id"] for t in trs}),
            "n_children": len({t["target_child_id"] for t in trs
                               if t["target_child_id"] is not None}),
        })
    (data_dir / "collections.json").write_text(
        json.dumps(collections, indent=1))
    print(f"wrote data/collections.json ({len(collections)} collections)")

    corpora = sorted(
        ({"collection_name": c["collection_name"], "name": c["name"],
          "data_source": c["data_source"]}
         for c in corpus.to_pylist()),
        key=lambda c: (c["collection_name"], c["name"]))
    (data_dir / "corpora.json").write_text(json.dumps(corpora, indent=1))
    print(f"wrote data/corpora.json ({len(corpora)} corpora)")

    stats = {
        "version": VERSION,
        "n_collections": len(collections),
        "n_corpora": len(corpora),
        "n_transcripts": len({t["id"] for t in tr}),
        "n_children": len({t["target_child_id"] for t in tr
                           if t["target_child_id"] is not None}),
    }
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote data/stats.json {stats}")


if __name__ == "__main__":
    main()
