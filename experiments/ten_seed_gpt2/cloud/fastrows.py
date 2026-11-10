"""Threaded-pread gather (with next-superblock prefetch) for pc1night's block cache. Lives in
cloud/ (outside io.seal()'s pc1night/*.py glob) and is installed from overnight.py.

Measured 2026-09-10 on 16 GB xlarge boxes whose train store (40-67 GB) exceeds RAM:
- v0 (package as shipped): np.asarray(memmap[sorted_rows]) = single-threaded page faults, ~2,100
  rows/s, 61 s per 32-step superblock, ~2 s/step, GPU 0%.
- v1/v2 (MADV_WILLNEED before the memmap gather, +prefetch thread): 0.48-0.50 s/step, GPU ~44%,
  iowait 0 - the disk was no longer the limit; the training thread sat at 99.9% CPU, 17% system
  time: ~131k madvise calls + ~393k page faults per superblock, ~100 us of kernel work per row,
  and the prefetch thread only relocated that CPU work (madvise holds the GIL).
- v3 (this file): rows are read with libc pread (ctypes) from a thread pool straight into the output array:
  no page faults, no readahead machinery, GIL released during each read, so a prefetch of the
  next superblock genuinely overlaps the GPU. Bytes come from the same file at the same offsets,
  so the gather is bit-identical (asserted against the memmap on a sample when
  PC1NIGHT_FASTROWS_CHECK=1, and in the deployment test).
Set PC1NIGHT_FASTROWS=0 to disable everything, PC1NIGHT_FASTROWS_PREFETCH=0 to disable prefetch.
"""
import copy, os, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

THREADS = int(os.environ.get('PC1NIGHT_FASTROWS_THREADS', '32'))

import ctypes, ctypes.util
_LIBC = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6', use_errno=True)
_PREAD = _LIBC.pread
_PREAD.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_PREAD.restype = ctypes.c_ssize_t


class FastRows:
    def __init__(self, x):
        self._x = x
        self._path = x.filename
        self._d = int(x.shape[1]) if x.ndim == 2 else None
        self._rb = self._d * x.dtype.itemsize if self._d else None
        self._off = int(x.offset)
        self._dtype = x.dtype
        self._check = os.environ.get('PC1NIGHT_FASTROWS_CHECK') == '1'
        self._lock = threading.Lock()
        self._pre = None            # [rows, thread, buf] of an in-flight or finished prefetch

    def _is_gather(self, key):
        return (self._rb is not None and isinstance(key, np.ndarray) and key.ndim == 1
                and key.dtype.kind in 'iu' and len(key) > 64)

    def _gather(self, rows):
        n = len(rows)
        out = np.empty((n, self._d), self._dtype)
        rb, off = self._rb, self._off
        base = out.ctypes.data
        offs = (off + rows.astype(np.int64) * rb).tolist()
        fd = os.open(self._path, os.O_RDONLY)
        try:
            def work(a, b):
                # libc pread straight into the array's memory: no bytes object, no view object,
                # no copy; ctypes releases the GIL for the call. ~2 us of Python per row.
                pread = _PREAD
                for j in range(a, b):
                    if pread(fd, base + j * rb, rb, offs[j]) != rb:
                        raise IOError('short read at row %d' % int(rows[j]))
            step = -(-n // THREADS)
            with ThreadPoolExecutor(THREADS) as ex:
                list(ex.map(lambda a: work(a, min(a + step, n)), range(0, n, step)))
        finally:
            os.close(fd)
        return out

    def prefetch(self, rows):
        with self._lock:
            if self._pre is not None:
                return
            slot = [np.asarray(rows).copy(), None, None]
            t = threading.Thread(target=lambda: slot.__setitem__(2, self._gather(slot[0])), daemon=True)
            slot[1] = t
            self._pre = slot
            t.start()

    def __getitem__(self, key):
        if self._is_gather(key):
            out = None
            with self._lock:
                pre, self._pre = self._pre, None
            if pre is not None:
                pre[1].join()
                if pre[2] is not None and np.array_equal(pre[0], key):
                    out = pre[2]
            if out is None:
                out = self._gather(key)
            if self._check:
                st = max(1, len(key) // 256)
                assert np.array_equal(out[::st], np.asarray(self._x[key[::st]])), 'FastRows mismatch'
            return out
        return self._x[key]

    def __getattr__(self, name):
        return getattr(self._x, name)

    def __len__(self):
        return len(self._x)

    def __array__(self, *a, **k):
        return np.asarray(self._x, *a, **k)


class _PrefetchOnly:
    """Stand-in for Store.x during the prefetch call: a gather only schedules the read."""
    def __init__(self, fr):
        self._fr = fr

    def __getitem__(self, key):
        if self._fr._is_gather(key):
            self._fr.prefetch(key)
            return None
        return self._fr[key]

    def __getattr__(self, name):
        return getattr(self._fr, name)

    def __len__(self):
        return len(self._fr)


class _PrefetchStore:
    def __init__(self, store):
        self.x = _PrefetchOnly(store.x)
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)


def install():
    if os.environ.get('PC1NIGHT_FASTROWS', '1') != '1':
        return
    import pc1night.data as D
    orig_init = D.Store.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        x = getattr(self, 'x', None)
        if isinstance(x, np.memmap) and getattr(x, 'filename', None):
            self.x = FastRows(x)
    D.Store.__init__ = __init__

    if os.environ.get('PC1NIGHT_FASTROWS_PREFETCH', '1') != '1':
        return
    orig_indices = D.TrainingView.indices

    def indices(self, step, batch, steps):
        before = getattr(self, '_block', None)
        out = orig_indices(self, step, batch, steps)
        try:
            if (os.environ.get('PC1NIGHT_BLOCK_CACHE') == '1' and self._block != before
                    and isinstance(getattr(self.st, 'x', None), FastRows)):
                nxt = (self._block + 1) * 32
                if nxt < steps:
                    c = copy.copy(self)
                    c.st = _PrefetchStore(self.st)
                    c._block = None; c._indices = None; c._rows = None; c._cache = None
                    orig_indices(c, nxt, batch, steps)     # only schedules the read; returns at once
        except Exception:
            pass
        return out
    D.TrainingView.indices = indices
