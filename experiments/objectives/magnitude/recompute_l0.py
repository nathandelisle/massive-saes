#!/usr/bin/env python3
"""Recompute L0 for the historical Pythia-1.4B TopK + beta*L1 sweep (exp_b2_l1_phase, S3 b2v2/).

Uses the UNMODIFIED pc1_v3 package staged for b2v2 (s3://.../b2v2/pkg.tgz):
  * io.ensure_acts -> data.harvest_activations: re-harvests the v3 store
    (pythia-1.4b, wikitext-103-v1 train, block index 12, 5M tokens, one document per
    128-token sequence, padding excluded, float16 store, float64 mean over all rows).
    The original store was never uploaded, so it is rebuilt with the same code.
  * io.alive: the archived firing-rate routine (unclamped final slice, divides by n).
  * saes.load_sae: loads the original checkpoints s3://.../b2v2/saes/<config_hash>.pt.

For every (beta, seed) in the archived sweep this records:
  archived_l0_measured   the value in pythia-1.4b_phase.json
  reproduced_old_routine rates.sum() from io.alive on the re-harvested store (validates the store)
  old_routine_rows       rows the unclamped loop actually encoded (expected 106,496)
  overrun_sample_l0      total nonzeros / rows actually encoded by that loop
  corrected_l0           total nonzeros over EXACTLY rows [0, 100000) / 100000, final batch clamped
plus per-row L0 histograms, and alive counts under both routines (rate > 1e-3).
"""
import hashlib, json, os, platform, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pc1_v3'))
import torch
import transformers
import datasets
import boto3
from v3.common import registry
from v3.common import io_utils as io
from v3.common.data import open_activations
from v3.common.saes import load_sae

BUCKET = 'ssup-research-079415246777'
OUT_KEY = 'pc1_sec5/pythia_l0/L0_RECOMPUTE.json'
N = registry.ALIVE_SAMPLE          # 100,000: the routine's requested sample
BATCH = 8192                       # the routine's batch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def count_rows(sae, acts, mean_t, n_rows, clamp, v1_t=None):
    total, rows = 0, 0
    hist = np.zeros(sae.d_sae + 1, dtype=np.int64)
    fire = torch.zeros(sae.d_sae, device='cuda', dtype=torch.float64)
    err = var = perr = pvar = 0.0
    for i in range(0, n_rows, BATCH):
        j = min(i + BATCH, n_rows) if clamp else i + BATCH
        x = torch.tensor(np.asarray(acts[i:j], dtype=np.float32), device='cuda') - mean_t
        hat, z = sae(x)
        nz = z != 0
        per_row = nz.sum(-1)
        total += int(per_row.sum())
        rows += x.shape[0]
        fire += nz.double().sum(0)
        hist += np.bincount(per_row.cpu().numpy(), minlength=sae.d_sae + 1)[: sae.d_sae + 1]
        e = (hat - x).double(); xd = x.double()
        err += float(e.square().sum()); var += float(xd.square().sum())
        if v1_t is not None:
            ep = e - (e @ v1_t)[:, None] * v1_t[None, :]; xp = xd - (xd @ v1_t)[:, None] * v1_t[None, :]
            perr += float(ep.square().sum()); pvar += float(xp.square().sum())
    fid = dict(fvu_total=err / var, fvu_content=(perr / pvar) if v1_t is not None else None)
    return total, rows, hist, (fire / rows).cpu().numpy(), fid


def main():
    root = os.environ['PC1_V3_ROOT']
    out_dir = os.path.join(root, 'results', 'l0_recompute')
    os.makedirs(out_dir, exist_ok=True)
    s3 = boto3.client('s3')
    archived = json.load(open(os.path.join(HERE, 'archived_pythia-1.4b_phase.json')))
    arch_rows = [r for r in archived['rows'] if r['sweep'] == 'topk_l1']
    spec = registry.MODELS['pythia-1.4b']
    env = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
               datasets=datasets.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
               archived_phase_sha256=sha256(os.path.join(HERE, 'archived_pythia-1.4b_phase.json')),
               pc1_v3_io_utils_sha256=sha256(os.path.join(HERE, 'pc1_v3/v3/common/io_utils.py')),
               pc1_v3_data_sha256=sha256(os.path.join(HERE, 'pc1_v3/v3/common/data.py')),
               pc1_v3_saes_sha256=sha256(os.path.join(HERE, 'pc1_v3/v3/common/saes.py')))
    print('[l0] env', json.dumps(env), flush=True)

    t0 = time.time()
    adir = io.ensure_acts(spec, device='cuda')
    harvest_seconds = time.time() - t0
    acts, mean = open_activations(adir)
    meta = json.load(open(os.path.join(adir, 'meta.json')))
    store = dict(act_dir=adir, shape=list(acts.shape), dtype=str(acts.dtype), meta=meta, harvest_seconds=harvest_seconds,
                 mean_sha256=hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest(),
                 first_106496_rows_sha256=hashlib.sha256(np.ascontiguousarray(acts[:106496]).tobytes()).hexdigest())
    print('[l0] store', json.dumps({k: v for k, v in store.items() if k != 'meta'}), flush=True)
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    from v3.common.pca import estimate_pcs
    _evals, _evecs, _ = estimate_pcs(adir)          # v3's PC1 (first 50k centered rows, float64), as the sweep used
    v1_t = torch.tensor(_evecs[:, 0], dtype=torch.float64, device='cuda')
    store['pc1_variance_share_first50k'] = float(_evals[0] / _evals.sum())

    rows = []
    for ar in arch_rows:
        beta = float(ar['coef'])
        for si, seed in enumerate([42, 137]):     # exp_b2_l1_phase.PAIR order
            path = io.sae_cache_path(spec, seed, arch='topk_l1', condition='standard', k=registry.TOPK_K, beta=beta)
            h = os.path.basename(path)[:-3]
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)   # sae_cache_path does not create it
                s3.download_file(BUCKET, f'b2v2/saes/{h}.pt', path)
            sae, extra = load_sae(path, device='cuda')
            cfg = extra.get('cfg', {})
            if cfg.get('seed') != seed or cfg.get('arch') != 'topk_l1' or abs(float(sae.beta) - beta) > 1e-12 or sae.k != 64:
                raise ValueError(f'checkpoint identity mismatch for beta={beta} seed={seed}: {cfg} beta={sae.beta} k={sae.k}')
            rates_old, alive_old = io.alive(sae, adir, device='cuda')          # archived routine, unmodified
            tot_c, rows_c, hist_c, rate_c, fid_c = count_rows(sae, acts, mean_t, N, clamp=True, v1_t=v1_t)
            tot_o, rows_o, hist_o, _, _ = count_rows(sae, acts, mean_t, N, clamp=False)
            nz = np.flatnonzero(hist_c)
            rec = dict(beta=beta, seed=seed, config_hash=h, checkpoint=f's3://{BUCKET}/b2v2/saes/{h}.pt',
                       checkpoint_sha256=sha256(path), checkpoint_cfg=cfg,
                       archived_l0_measured=float(ar['l0_measured'][si]), archived_l0_measured_mean=float(ar['l0_measured_mean']),
                       archived_l0_batch_seed42_only=float(ar['l0_batch']), archived_n_alive_seed42=int(ar['n_alive']),
                       reproduced_old_routine=float(rates_old.sum()), old_routine_denominator=N,
                       old_routine_rows=rows_o, overrun_sample_l0=tot_o / rows_o,
                       corrected_l0=tot_c / rows_c, corrected_rows=rows_c, corrected_total_nonzero=tot_c,
                       rows_below_k=int(hist_c[:64].sum()), min_row_l0=int(nz[0]), max_row_l0=int(nz[-1]),
                       row_l0_histogram={int(i): int(hist_c[i]) for i in nz},
                       n_alive_old_routine=int(alive_old.sum()), n_alive_corrected=int((rate_c > registry.ALIVE_FIRING_RATE).sum()),
                       fvu_total_first100k_train_rows=fid_c['fvu_total'], fvu_content_first100k_train_rows=fid_c['fvu_content'],
                       archived_fvu_is_train_log_step19000_seed42=float(ar['fvu']) if ar.get('fvu') is not None else None,
                       checkpoint_train_log_tail=extra.get('log'))
            rows.append(rec)
            print(f"[l0] beta={beta:g} seed={seed} archived={rec['archived_l0_measured']:.6f} "
                  f"reproduced_old={rec['reproduced_old_routine']:.6f} rows_old={rows_o} "
                  f"corrected={rec['corrected_l0']:.6f} over {rows_c} rows", flush=True)
            json.dump(dict(env=env, store=store, rows=rows, complete=False), open(os.path.join(out_dir, 'L0_RECOMPUTE.json'), 'w'), indent=1)
            del sae
            torch.cuda.empty_cache()
    result = dict(env=env, store=store, rows=rows, complete=True, n=N, batch=BATCH,
                  definition='corrected_l0 = sum over rows r in [0, 100000) of #nonzero(sae.encode(acts[r] - mean)) / 100000; '
                             'final batch clamped; same store, centering, float32 encode and row order as io.alive')
    p = os.path.join(out_dir, 'L0_RECOMPUTE.json')
    json.dump(result, open(p, 'w'), indent=1)
    s3.upload_file(p, BUCKET, OUT_KEY)
    print(f'[l0] DONE -> s3://{BUCKET}/{OUT_KEY}', flush=True)


if __name__ == '__main__':
    main()
