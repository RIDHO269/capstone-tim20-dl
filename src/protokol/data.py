from __future__ import annotations

import numpy as np
from src.common import split

TUGAS = "klasifikasi"


def muat():
    """Pemuat data simulasi sementara untuk verifikasi pipeline run_all.py."""
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((200, 16), dtype=np.float32)
    y = rng.integers(0, 3, size=200, dtype=np.int64)

    tr, va = split.indeks(len(X), porsi_val=0.2)
    Xte = rng.standard_normal((50, 16), dtype=np.float32)
    yte = rng.integers(0, 3, size=50, dtype=np.int64)
    return X[tr], y[tr], X[va], y[va], Xte, yte