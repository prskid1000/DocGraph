"""Bounded-memory top-k cosine neighbours for SIMILAR_TO.

Never builds an n x n matrix. Two regimes:

* n <= EXACT_MAX: exact, row blocks of `block` vectors against the whole
  (unit-normalised) matrix -- block x n float32 per step.
* larger: an IVF pass. k-means (on a sample) splits the vectors into
  ~sqrt(n) lists; every list is compared only against itself and its
  `nprobe - 1` nearest lists, in row blocks. Lists far larger than the
  target size are split again (a second k-means inside the list), so one
  dense region cannot degrade to n^2. Recall is that of IVF with nprobe
  lists -- plenty for "similar code" edges gated at cosine >= 0.5.

The incremental path does not come here at all: dirty entities query the
Kuzu HNSW index (O(changed)).

Pure numpy, deterministic (fixed seed).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EXACT_MAX = 8000
BLOCK = 1024
NPROBE = 3
KMEANS_ITERS = 8
SEED = 1234


def unit_rows(mat: np.ndarray) -> np.ndarray:
    """float32 copy with unit-norm rows (zero rows stay zero)."""
    m = np.asarray(mat, dtype=np.float32)
    if not m.flags.writeable or m.base is not None:
        m = np.array(m, dtype=np.float32, copy=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m /= norms
    return m


def _topk_rows(sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices, values) of the k largest entries of each row, unsorted."""
    kk = min(k, sims.shape[1])
    if kk <= 0:
        return np.empty((sims.shape[0], 0), np.int64), np.empty((sims.shape[0], 0), np.float32)
    part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(sims, part, axis=1)
    return part, vals


def _kmeans(x: np.ndarray, n_lists: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """Spherical k-means centroids (unit rows) on `x` (unit rows)."""
    n = len(x)
    n_lists = max(1, min(n_lists, n))
    cent = x[rng.choice(n, size=n_lists, replace=False)].copy()
    for _ in range(iters):
        assign = np.empty(n, dtype=np.int64)
        for s in range(0, n, 8192):
            assign[s:s + 8192] = np.argmax(x[s:s + 8192] @ cent.T, axis=1)
        sums = np.zeros_like(cent)
        np.add.at(sums, assign, x)
        counts = np.bincount(assign, minlength=n_lists)
        empty = counts == 0
        if empty.any():
            # re-seed empty lists with random points
            sums[empty] = x[rng.choice(n, size=int(empty.sum()), replace=True)]
        cent = unit_rows(sums)
    return cent


def _assign(x: np.ndarray, cent: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), 8192):
        out[s:s + 8192] = np.argmax(x[s:s + 8192] @ cent.T, axis=1)
    return out


def top_similar(mat: np.ndarray, k: int, threshold: float = 0.5,
                on_progress: Callable[[int], None] | None = None,
                exact_max: int = EXACT_MAX, block: int = BLOCK,
                nprobe: int = NPROBE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k neighbours (excluding self) with cosine >= threshold for every
    row of `mat` (float32, need not be normalised). Returns (src_row,
    dst_row, score) arrays. Peak extra memory ~ block x max(n, list span)."""
    x = unit_rows(mat)
    n = len(x)
    if n < 2 or k <= 0:
        e = np.empty(0, np.int64)
        return e, e, np.empty(0, np.float32)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    val: list[np.ndarray] = []

    where = np.full(n, -1, dtype=np.int64)   # row -> column in the candidate block

    def emit(q_rows: np.ndarray, c_rows: np.ndarray, sims: np.ndarray) -> None:
        # mask self-similarity where a query row is also a candidate
        j = where[q_rows]
        hit = j >= 0
        sims[np.nonzero(hit)[0], j[hit]] = -2.0
        idx, vals = _topk_rows(sims, k)
        keep = vals >= threshold
        qi = np.repeat(q_rows[:, None], idx.shape[1], axis=1)
        src.append(qi[keep])
        dst.append(c_rows[idx[keep]])
        val.append(vals[keep].astype(np.float32))

    if n <= exact_max:
        all_rows = np.arange(n, dtype=np.int64)
        where[:] = all_rows
        for s in range(0, n, block):
            q = all_rows[s:s + block]
            sims = x[q] @ x.T
            emit(q, all_rows, sims)
            if on_progress is not None:
                on_progress(len(q))
    else:
        rng = np.random.default_rng(SEED)
        n_lists = int(np.clip(np.sqrt(n), 16, 4096))
        sample = x if n <= n_lists * 64 else x[rng.choice(n, size=n_lists * 64, replace=False)]
        cent = _kmeans(sample, n_lists, KMEANS_ITERS, rng)
        assign = _assign(x, cent)
        # Split oversized lists so a dense region stays sub-quadratic.
        target = max(64, n // len(cent))
        members: list[np.ndarray] = []
        cents: list[np.ndarray] = []
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(len(cent) + 1))
        for li in range(len(cent)):
            m = order[bounds[li]:bounds[li + 1]]
            if len(m) == 0:
                continue
            if len(m) > 8 * target:
                sub_n = int(np.ceil(len(m) / target))
                sub_c = _kmeans(x[m], sub_n, 4, rng)
                sub_a = _assign(x[m], sub_c)
                for si in range(len(sub_c)):
                    mm = m[sub_a == si]
                    if len(mm):
                        members.append(mm)
                        cents.append(sub_c[si])
            else:
                members.append(m)
                cents.append(cent[li])
        cmat = np.stack(cents).astype(np.float32)
        near = cmat @ cmat.T
        probe = min(nprobe, len(members))
        nbr = np.argsort(-near, axis=1)[:, :probe]
        for li, q_all in enumerate(members):
            cand = np.concatenate([members[j] for j in nbr[li]])
            where[cand] = np.arange(len(cand))
            xc = x[cand]
            for s in range(0, len(q_all), block):
                q = q_all[s:s + block]
                emit(q, cand, x[q] @ xc.T)
            where[cand] = -1
            if on_progress is not None:
                on_progress(len(q_all))
    if not src:
        e = np.empty(0, np.int64)
        return e, e, np.empty(0, np.float32)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(val)
