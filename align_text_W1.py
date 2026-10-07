"""Pad the 500 bag-of-words phrase counts to all 25,692 cohort admissions (2026-10-06).

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3 align_text_W1.py            # build out/text_open_W1_v2_aligned.npz, or check it if it exists
    python3 align_text_W1.py --force    # overwrite an existing file

WHY. `build_text_W1.py all --v2` (notes without discharge summaries) writes the phrase counts only
for admissions that have a note inside the 48-hour window: 25,320 rows, listed in
out/text_open_rows_W1_v2.parquet. 372 admissions have no such note. Every model script concatenates
the text block with out/model_tabular_W1_v2.parquet, which has one row per cohort admission
(25,692, sorted by hadm_id). This script places each text row on its admission's row and gives the
372 note-less admissions a row of zeros.

This step was first done as a one-off command on 2026-09-17 and never saved as a script. Its output,
out/text_open_W1_v2_aligned.npz, is reproduced exactly by this script (checked 2026-10-06).

IN   out/text_open_W1_v2.npz, out/text_open_rows_W1_v2.parquet, out/model_tabular_W1_v2.parquet
OUT  out/text_open_W1_v2_aligned.npz   (25,692 x 500, float32, CSR)
"""
import os, sys
import numpy as np, pandas as pd
from scipy import sparse

D = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
OUT = f"{U}/text_open_W1_v2_aligned.npz"

X = sparse.load_npz(f"{U}/text_open_W1_v2.npz").tocsr()
rows = pd.read_parquet(f"{U}/text_open_rows_W1_v2.parquet").hadm_id.to_numpy()
order = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet").hadm_id.to_numpy()
assert X.shape[0] == len(rows) and len(set(rows)) == len(rows), "text rows and their ids disagree"
assert (np.sort(order) == order).all() and set(rows) <= set(order), "tabular order or ids unexpected"

pos = pd.Index(rows).get_indexer(order)                  # text row for each cohort admission, -1 if none
have = np.flatnonzero(pos >= 0)
P = sparse.csr_matrix((np.ones(len(have), X.dtype), (have, pos[have])), shape=(len(order), X.shape[0]))
A = (P @ X).tocsr(); A.sort_indices()                    # selection only: values are copied, not summed
print(f"{len(order):,} admissions, {len(have):,} with text, {len(order) - len(have):,} padded with zeros; "
      f"{A.shape[1]} phrases")

if os.path.exists(OUT) and "--force" not in sys.argv:
    old = sparse.load_npz(OUT).tocsr()
    same = old.shape == A.shape and old.dtype == A.dtype and abs(old - A).max() == 0
    print(f"{OUT} exists: {'identical to the rebuilt matrix' if same else 'DIFFERS from the rebuilt matrix'}"
          " (not overwritten; use --force)")
    sys.exit(0 if same else 1)
sparse.save_npz(OUT, A); print(f"written {OUT}")
