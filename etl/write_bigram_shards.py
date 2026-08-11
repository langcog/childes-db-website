#!/usr/bin/env python3
"""Write anchor-hash-sharded bigram parquet for the Bigram Browser page.

Reads the token parquet export of a childes-db release and computes
within-utterance adjacent word pairs: tokens are ordered by token_order within
each utterance_id, glosses are NFC-normalized and lowercased, and any pair
touching an unintelligible/absent gloss (xxx, yyy, www, empty) is dropped.
Pairs are aggregated to (collection_name, corpus_name, speaker_role, w1, w2, n)
and rows with n = 1 are pruned (singleton pairs; see the printed stats).

Each surviving row is emitted twice with an `anchor` column so that one shard
fetch answers both directions for a queried word:
  anchor = w1, pos = 'after'   (word = w2: words appearing after the anchor)
  anchor = w2, pos = 'before'  (word = w1: words appearing before the anchor)

Shards use the SAME hash as the frequency page: FNV-1a 32-bit of the
NFC-lowercased anchor, mod N_SHARDS (imported from write_freq_shards.py).
Output: slices/bigrams/shard-<n>.parquet (zstd, sorted by anchor) plus
slices/bigrams/hash_check.json test vectors that bigrams.qmd asserts at load
time, so hash drift between Python and JS fails loudly.

Usage: python etl/write_bigram_shards.py [staging_dir]
"""

import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_freq_shards import (  # noqa: E402
    HASH_CHECK_GLOSSES, N_SHARDS, fnv1a, norm_gloss, shard_of,
)

STAGING = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/Users/mcfrank/Projects/childes-db/pipeline/parquet_compact"
)
SITE = Path(__file__).resolve().parent.parent
OUT = SITE / "slices" / "bigrams"

DROP = ("xxx", "yyy", "www")  # unintelligible / phonology-only / URL glosses


def main():
    glob = str(STAGING / "token" / "*.parquet")
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{OUT.parent / 'tmp_duckdb'}'")
    con.execute("SET preserve_insertion_order = false")

    drop_list = ", ".join(f"'{w}'" for w in DROP)
    print("aggregating within-utterance bigrams (window over token_order)...",
          flush=True)
    con.execute(f"""
        CREATE TEMP TABLE pair_counts AS
        WITH tok AS (
            SELECT collection_name, corpus_name, speaker_role,
                   utterance_id, token_order,
                   lower(nfc_normalize(gloss)) AS w
            FROM read_parquet('{glob}')
        ),
        pairs AS (
            SELECT collection_name, corpus_name, speaker_role, w AS w1,
                   lead(w) OVER (PARTITION BY utterance_id
                                 ORDER BY token_order) AS w2
            FROM tok
        )
        SELECT collection_name, corpus_name, speaker_role, w1, w2,
               count(*)::INT AS n
        FROM pairs
        WHERE w2 IS NOT NULL AND length(w1) > 0 AND length(w2) > 0
          AND w1 NOT IN ({drop_list}) AND w2 NOT IN ({drop_list})
        GROUP BY ALL
    """)

    # duckdb's lower(nfc_normalize()) should equal norm_gloss(); if any gloss
    # is not a fixed point of the Python normalization, remap and re-aggregate
    # so pruning happens after variants merge.  Must run before pruning.
    words = [r[0] for r in con.execute(
        "SELECT DISTINCT w1 FROM pair_counts "
        "UNION SELECT DISTINCT w2 FROM pair_counts").fetchall()]
    fix = [(w, norm_gloss(w)) for w in words if norm_gloss(w) != w]
    if fix:
        print(f"re-normalizing {len(fix)} glosses where duckdb and Python "
              f"normalization differ, e.g. {fix[:5]}", flush=True)
        con.execute("CREATE TEMP TABLE fix (raw VARCHAR, norm VARCHAR)")
        con.executemany("INSERT INTO fix VALUES (?, ?)", fix)
        con.execute("""
            CREATE TEMP TABLE pair_counts2 AS
            SELECT collection_name, corpus_name, speaker_role,
                   coalesce(f1.norm, w1) AS w1, coalesce(f2.norm, w2) AS w2,
                   SUM(n)::INT AS n
            FROM pair_counts
            LEFT JOIN fix f1 ON w1 = f1.raw
            LEFT JOIN fix f2 ON w2 = f2.raw
            GROUP BY ALL
        """)
        con.execute("DROP TABLE pair_counts")
        con.execute("ALTER TABLE pair_counts2 RENAME TO pair_counts")
    else:
        print("duckdb and Python gloss normalization agree on all "
              f"{len(words)} distinct glosses", flush=True)

    rows_all, tokens_all, rows_single = con.execute("""
        SELECT count(*), SUM(n),
               count(*) FILTER (WHERE n = 1)
        FROM pair_counts
    """).fetchone()
    con.execute("DELETE FROM pair_counts WHERE n = 1")
    rows_kept, tokens_kept = con.execute(
        "SELECT count(*), SUM(n) FROM pair_counts").fetchone()
    print(f"pair rows (collection, corpus, role, w1, w2): {rows_all:,}; "
          f"pruned {rows_single:,} singletons ({100 * rows_single / rows_all:.1f}% "
          f"of rows, {100 * (tokens_all - tokens_kept) / tokens_all:.1f}% of "
          f"pair tokens); kept {rows_kept:,} rows / {tokens_kept:,} tokens",
          flush=True)

    # each row twice: anchored on w1 (its 'after' neighbors) and w2 ('before')
    con.execute("""
        CREATE TEMP TABLE doubled AS
        SELECT w1 AS anchor, 'after'  AS pos, w2 AS word,
               collection_name, corpus_name, speaker_role, n FROM pair_counts
        UNION ALL
        SELECT w2 AS anchor, 'before' AS pos, w1 AS word,
               collection_name, corpus_name, speaker_role, n FROM pair_counts
    """)

    anchors = [r[0] for r in
               con.execute("SELECT DISTINCT anchor FROM doubled").fetchall()]
    con.execute("CREATE TEMP TABLE shard_map (anchor VARCHAR, shard INT)")
    con.executemany("INSERT INTO shard_map VALUES (?, ?)",
                    [(a, fnv1a(a) % N_SHARDS) for a in anchors])

    sizes = []
    for s in range(N_SHARDS):
        path = OUT / f"shard-{s}.parquet"
        con.execute(f"""
            COPY (
                SELECT d.anchor, d.pos, d.word, d.collection_name,
                       d.corpus_name, d.speaker_role, d.n
                FROM doubled d JOIN shard_map m ON d.anchor = m.anchor
                WHERE m.shard = {s}
                ORDER BY d.anchor, d.pos, d.n DESC
            ) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n_rows = con.execute(f"SELECT count(*) FROM shard_map m "
                             f"JOIN doubled d ON d.anchor = m.anchor "
                             f"WHERE m.shard = {s}").fetchone()[0]
        sizes.append((n_rows, path.stat().st_size))
        print(f"shard-{s}: {n_rows:,} rows, "
              f"{path.stat().st_size / 1e6:.2f} MB", flush=True)

    rows = [r for r, _ in sizes]
    mbs = [b / 1e6 for _, b in sizes]
    print(f"{N_SHARDS} shards: rows/shard min {min(rows):,} max {max(rows):,}, "
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
