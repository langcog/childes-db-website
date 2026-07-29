#!/usr/bin/env python3
"""Write gloss-hash-sharded parquet for the Frequency Counts page.

Reads the token_frequency parquet export of a childes-db release and writes
N_SHARDS zstd parquet shards to slices/freq/shard-<n>.parquet. Each row of
token_frequency is one (gloss, speaker, transcript) count; a shard holds every
row whose FNV-1a 32-bit hash of the NFC-normalized, lowercased gloss is
congruent to <n> mod N_SHARDS. The page computes the same hash in JS to fetch
only the shard(s) containing the queried words.

Also writes slices/freq/hash_check.json: test vectors (gloss -> shard) that
the page asserts against at load time, so any hash drift between this script
and the JS implementation fails loudly instead of silently returning no rows.

Columns kept per shard row:
  gloss_lower  NFC-lowercased gloss (match key; shards are sorted by it)
  gloss        original gloss as it appears in the corpus
  count        token count for this (gloss, speaker, transcript)
  speaker_role, transcript_id, corpus_name, collection_name,
  target_child_id, target_child_name
  age          target-child age in months (days / (365.2425/12), rounded 0.1)

Rows without a target-child age are dropped (they cannot be plotted, and the
old shiny app dropped them too). ppm denominators are NOT written here: the
per-collection speaker_stats slices already carry num_tokens per
(transcript, speaker), which is exactly childesr::get_speaker_statistics.

Usage: python etl/write_freq_shards.py [staging_dir]
"""

import glob
import json
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STAGING = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/Users/mcfrank/Projects/childes-db/pipeline/parquet_compact"
)
SITE = Path(__file__).resolve().parent.parent
OUT = SITE / "slices" / "freq"

N_SHARDS = 64
DAYS_PER_MONTH = 365.2425 / 12

# glosses whose (gloss -> shard) mapping the page asserts at load time
HASH_CHECK_GLOSSES = ["ball", "the", "dog", "I've", "mamá", "犬", "Mommy"]


def norm_gloss(g):
    """NFC-normalize + lowercase; must match normGloss() in frequency.qmd."""
    return unicodedata.normalize("NFC", g).lower()


def fnv1a(s):
    """FNV-1a 32-bit over UTF-8 bytes; must match fnv1a() in frequency.qmd."""
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def shard_of(gloss):
    return fnv1a(norm_gloss(gloss)) % N_SHARDS


def main():
    corpus = pq.read_table(STAGING / "corpus")
    corpus_name = dict(zip(corpus.column("id").to_pylist(),
                           corpus.column("name").to_pylist()))

    files = sorted(glob.glob(str(STAGING / "token_frequency" / "*.parquet")))
    if not files:
        sys.exit(f"no token_frequency parquet under {STAGING}")

    OUT.mkdir(parents=True, exist_ok=True)
    parts = [[] for _ in range(N_SHARDS)]
    total_in = total_kept = 0

    # cache per-gloss derived values; vocabulary is far smaller than the rows
    gloss_cache = {}

    for f in files:
        t = pq.read_table(f, columns=[
            "gloss", "count", "speaker_role", "target_child_age",
            "target_child_name", "target_child_id", "transcript_id",
            "corpus_id", "collection_name",
        ])
        total_in += t.num_rows
        t = t.filter(pc.is_valid(t.column("target_child_age")))
        total_kept += t.num_rows

        glosses = t.column("gloss").to_pylist()
        lowers = [None] * len(glosses)
        shards = [0] * len(glosses)
        for i, g in enumerate(glosses):
            hit = gloss_cache.get(g)
            if hit is None:
                lo = norm_gloss(g)
                hit = (lo, fnv1a(lo) % N_SHARDS)
                gloss_cache[g] = hit
            lowers[i] = hit[0]
            shards[i] = hit[1]

        age = pc.round(pc.divide(t.column("target_child_age"),
                                 DAYS_PER_MONTH), 1)
        cids = t.column("corpus_id").to_pylist()
        out = pa.table({
            "gloss_lower": pa.array(lowers, pa.string()),
            "gloss": t.column("gloss"),
            "count": t.column("count"),
            "speaker_role": t.column("speaker_role"),
            "transcript_id": t.column("transcript_id"),
            "corpus_name": pa.array([corpus_name.get(c) for c in cids],
                                    pa.string()),
            "collection_name": t.column("collection_name"),
            "target_child_id": t.column("target_child_id"),
            "target_child_name": t.column("target_child_name"),
            "age": age,
        })
        shard_arr = pa.array(shards, pa.int32())
        for s in range(N_SHARDS):
            sub = out.filter(pc.equal(shard_arr, s))
            if sub.num_rows:
                parts[s].append(sub)
        print(f"processed {Path(f).name} ({t.num_rows} rows)", flush=True)

    sizes = []
    for s in range(N_SHARDS):
        shard = pa.concat_tables(parts[s]).sort_by("gloss_lower")
        path = OUT / f"shard-{s}.parquet"
        pq.write_table(shard, path, compression="zstd")
        sizes.append((shard.num_rows, path.stat().st_size))

    rows = [r for r, _ in sizes]
    mbs = [b / 1e6 for _, b in sizes]
    print(f"input rows {total_in}, kept {total_kept} "
          f"(dropped {total_in - total_kept} with no age)")
    print(f"{N_SHARDS} shards: rows/shard min {min(rows)} max {max(rows)}, "
          f"MB/shard min {min(mbs):.2f} max {max(mbs):.2f} "
          f"total {sum(mbs):.1f} MB")

    check = {
        "n_shards": N_SHARDS,
        "vectors": [
            {"gloss": g, "gloss_lower": norm_gloss(g), "shard": shard_of(g)}
            for g in HASH_CHECK_GLOSSES
        ],
    }
    (OUT / "hash_check.json").write_text(json.dumps(check, indent=1,
                                                    ensure_ascii=False))
    print("wrote hash_check.json:", check["vectors"])


if __name__ == "__main__":
    main()
