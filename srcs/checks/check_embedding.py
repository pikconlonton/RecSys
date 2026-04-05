"""
Kiểm tra chất lượng embedding files (.pkl).

In thống kê cho mỗi file embedding:
- Số lượng entries, kích thước file
- Vector shape, dtype
- NaN/Inf detection
- L2 norm distribution (min, max, mean, std)

Dùng để verify embeddings trước khi ghép vào graph.
"""
import pickle
import numpy as np
from pathlib import Path

SEP = "=" * 60

def check_embedding_file(path: Path, label: str):
    """Kiểm tra 1 file embedding pkl.
    In: type, size, entries count, sample key/value,
    vector shapes/dtypes, NaN/Inf count, L2 norm stats.
    """
    print(SEP)
    print(f"  {label}")
    print(f"  File : {path}")
    print(SEP)

    if not path.exists():
        print(f"  [ERROR] File not found: {path}\n")
        return

    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  Size : {size_mb:.2f} MB")

    with path.open("rb") as f:
        data = pickle.load(f)

    # ── Type ──────────────────────────────────────────────────────
    print(f"  Type : {type(data).__name__}")

    if not isinstance(data, dict):
        print("  [WARN] Expected a dict, got something else. Skipping further checks.\n")
        return

    # ── Basic stats ───────────────────────────────────────────────
    num_entries = len(data)
    print(f"  Entries: {num_entries:,}")

    if num_entries == 0:
        print("  [WARN] Empty dict.\n")
        return

    # ── Sample key ────────────────────────────────────────────────
    sample_key = next(iter(data))
    sample_val = data[sample_key]
    print(f"\n  Sample key  : {sample_key!r}  (type={type(sample_key).__name__})")
    print(f"  Sample value: type={type(sample_val).__name__}", end="")

    if isinstance(sample_val, np.ndarray):
        print(f"  shape={sample_val.shape}  dtype={sample_val.dtype}")
    else:
        print()

    # ── Vector stats across all entries ──────────────────────────
    dims   = set()
    dtypes = set()
    has_nan = 0
    has_inf = 0

    for vec in data.values():
        if isinstance(vec, np.ndarray):
            dims.add(vec.shape)
            dtypes.add(str(vec.dtype))
            if np.isnan(vec).any():
                has_nan += 1
            if np.isinf(vec).any():
                has_inf += 1

    print(f"\n  Vector shapes : {dims}")
    print(f"  Vector dtypes : {dtypes}")
    print(f"  Has NaN       : {has_nan:,} entries")
    print(f"  Has Inf       : {has_inf:,} entries")

    # ── Norm stats (sample up to 10_000) ─────────────────────────
    sample_keys = list(data.keys())[:10_000]
    vecs = np.stack([data[k] for k in sample_keys if isinstance(data[k], np.ndarray)])
    norms = np.linalg.norm(vecs, axis=1)
    print(f"\n  L2 norm stats (first {len(sample_keys):,} entries):")
    print(f"    min={norms.min():.4f}  max={norms.max():.4f}"
          f"  mean={norms.mean():.4f}  std={norms.std():.4f}")

    print()


if __name__ == "__main__":
    BASE = Path("outputs")

    check_embedding_file(BASE / "business_embeddings.pkl", "BUSINESS EMBEDDINGS")
    check_embedding_file(BASE / "user_embeddings.pkl",     "USER EMBEDDINGS")

    print("✓ Check complete.")
