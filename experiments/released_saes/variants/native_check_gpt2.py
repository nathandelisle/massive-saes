"""Independent native-forward check for every released-SAE converter used in the audit.

The adapter's own `smoke` compares `sae.encode`+`sae.decode` against `sae(x)` -- two paths through the
SAME converted object, which cannot detect a converter/release mismatch. This script instead rebuilds each
release's documented forward pass from its RAW pinned checkpoint tensors and compares feature activations
and reconstructions against the converted SAE on real cached activations (ordinary rows, high-norm rows)
plus the synthetic probes x=0 and x=b_dec, which is where a missing pre-encoder centering shows up most.

Native equations, from each release's own code/report:
  sparsify   (Pythia)        sae_in = x - b_dec; pre = relu(encoder(sae_in)); topk(pre,k); y = acts@W_dec + b_dec
                             sparsify@c259f271 sae.py pre_acts/encode/decode. b_dec is subtracted ALWAYS.
  jumprelu   (Gemma Scope,   pre = x@W_enc + b_enc; acts = pre*(pre>threshold); y = acts@W_dec + b_dec
              Gemma Scope 2) no pre-encoder centering (released cfg apply_b_dec_to_input=false).
  jumprelu   (Llama Scope)   dataset-level constant normalization in, inverse out; report §§3.3.3-3.3.4.
  topk       (OpenAI GPT-2)  per-vector mean removal + normalization (2406.04093 §2.1) BEFORE the SAE and
                             for the reconstruction-error evaluation; pre = (x'-b_dec)@W_enc + b_enc.
  topk       (Qwen Scope)    pre = x@W_enc + b_enc; acts = topk(relu(pre),50); y = acts@W_dec + b_dec.
                             Verbatim from the release's own app.py `compute_sae_features` -- no centering.

usage: native_check.py --group gpt2 [--job-id ...] --cache cache --out results/native
"""
import argparse
import json
import pathlib
import time

import numpy as np


def rel(a, b):
    d = float(np.linalg.norm(a - b))
    return d / max(float(np.linalg.norm(b)), 1e-30)


def resolve(keys, *cands, required=True):
    low = {k.lower().lstrip('_'): k for k in keys}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise RuntimeError("cannot resolve %r among %s" % (cands[0], sorted(keys)))
    return None


def load_tensors(path):
    p = str(path)
    if p.endswith('.npz'):
        z = np.load(p)
        return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    if p.endswith('.safetensors'):
        # torch, not safetensors.numpy: Llama Scope ships bfloat16, which numpy cannot represent.
        import torch
        from safetensors.torch import load_file
        return {k: v.float().numpy().astype(np.float64) for k, v in load_file(p).items()}
    import torch
    obj = torch.load(p, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict):
        raise RuntimeError('unexpected checkpoint object %s' % type(obj))
    return {k: v.float().numpy().astype(np.float64) for k, v in obj.items() if hasattr(v, 'numpy')}


def orient(W, d_in, d_sae, want):
    """want='enc' -> [d_in,d_sae]; want='dec' -> [d_sae,d_in]. Unambiguous because d_in != d_sae."""
    target = (d_in, d_sae) if want == 'enc' else (d_sae, d_in)
    if tuple(W.shape) == target:
        return W, False
    if tuple(W.shape) == target[::-1]:
        return W.T.copy(), True
    raise RuntimeError('%s matrix has shape %s, expected %s or its transpose' % (want, W.shape, target))


def topk_rows(pre, k):
    acts = np.zeros_like(pre)
    idx = np.argpartition(-pre, k - 1, axis=-1)[:, :k]
    np.put_along_axis(acts, idx, np.take_along_axis(pre, idx, axis=-1), axis=-1)
    return acts


class Native:
    """One release's forward pass, built only from its raw checkpoint tensors."""

    def __init__(self, job, d_in, d_sae, files, log=print):
        self.job, self.d_in, self.d_sae, self.notes = job, d_in, d_sae, {}
        T = {}
        for f in files:
            for k, v in load_tensors(f).items():
                T[k if k not in T else '%s:%s' % (f.name, k)] = v
        self.keys = sorted(T)
        log('tensor keys: %s' % self.keys)
        self.kind = job.get('conversion_func') or ('sparsify' if job['loader'] == 'sparsify' else 'openai_topk')
        kenc = resolve(T, 'W_enc', 'w_enc', 'encoder.weight')
        kdec = resolve(T, 'W_dec', 'w_dec', 'decoder.weight')
        self.W_enc, f1 = orient(T[kenc], d_in, d_sae, 'enc')
        self.W_dec, f2 = orient(T[kdec], d_in, d_sae, 'dec')
        self.b_enc = T[resolve(T, 'b_enc', 'encoder.bias')].ravel()
        self.b_dec = T[resolve(T, 'b_dec', 'decoder.bias')].ravel()
        self.notes['transposed_to_canonical'] = dict(W_enc=f1, W_dec=f2)
        self.notes['source_keys'] = dict(W_enc=kenc, W_dec=kdec)
        self.notes['decoder_row_norm_median'] = float(np.median(np.linalg.norm(self.W_dec, axis=1)))
        kthr = resolve(T, 'threshold', 'jumprelu_threshold', 'log_jumprelu_threshold', required=False)
        self.threshold = None
        if kthr is not None:
            t = T[kthr].ravel()
            self.threshold = np.exp(t) if 'log' in kthr.lower() else t
            self.notes['threshold_key'] = kthr
            self.notes['threshold_median'] = float(np.median(self.threshold))
        self.k = job.get('k')
        self.scale_in = 1.0
        self.scale_out = 1.0

    def norm_in(self, x):
        if self.kind == 'openai_topk':
            c = x - x.mean(-1, keepdims=True)
            scale = np.linalg.norm(c, axis=-1, keepdims=True) / np.sqrt(self.d_in)
            # A constant vector (e.g. the x=0 probe) has no direction to normalize: the release's rule is
            # undefined there, so leave the row at zero instead of producing NaN.
            return np.divide(c, scale, out=np.zeros_like(c), where=scale > 0)
        return x * self.scale_in

    def forward(self, x, presplit=None):
        """presplit: an already-normalized input, to separate SAE math from the normalization convention."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(presplit, dtype=np.float64) if presplit is not None else self.norm_in(x)
        if self.threshold is not None:
            pre = z @ self.W_enc + self.b_enc
            acts = np.where(pre > self.threshold, pre, 0.0)
        else:
            centered = z - self.b_dec if (self.kind == 'sparsify' or self.job.get('center_topk')) else z
            pre = np.maximum(centered @ self.W_enc + self.b_enc, 0.0)
            acts = topk_rows(pre, int(self.k))
        y = acts @ self.W_dec + self.b_dec
        return acts, y, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='manifest.json')
    p.add_argument('--group', required=True)
    p.add_argument('--job-id')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='results/native')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context', type=int)
    p.add_argument('--rows', type=int, default=256)
    a = p.parse_args()
    import torch
    import huggingface_hub
    from adapters import Adapter
    m = json.loads(pathlib.Path(a.manifest).read_text())
    jobs = [j for j in m['jobs'] if j['group'] == a.group and (not a.job_id or j['job_id'] == a.job_id)]
    job = jobs[0]
    ctx = a.context or m['context_length']
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(*x):
        print('[native]', *x, flush=True)

    log('checking %s (%s, %s@%s)' % (job['job_id'], job['family'], job['repo_id'], job['repo_revision'][:8]))

    root = pathlib.Path(a.cache) / a.group / ('ctx%d' % ctx) / 'test'
    X = np.load(root / ('layer_%d.npy' % job['layer']), mmap_mode='r')
    special = np.load(root / 'special.npy')
    pool = np.flatnonzero(~special)[:20000]
    nrm = np.linalg.norm(np.asarray(X[pool], dtype=np.float64), axis=1)
    order = np.argsort(nrm)
    half = a.rows // 2
    mid = order[len(order) // 2 - half: len(order) // 2 + half]
    rows = dict(ordinary=pool[mid], high_norm=pool[order[-a.rows:]])

    adapter = Adapter(job, a.device)
    d_in = int(adapter.sae.W_dec.shape[1])
    d_sae = int(adapter.sae.W_dec.shape[0])
    files = []
    for f in job['resolved_files']:
        try:
            files.append(pathlib.Path(huggingface_hub.hf_hub_download(
                job['repo_id'], filename=f, revision=job['repo_revision'])))
        except Exception as e:
            log('could not fetch %s: %s' % (f, e))
    weight_files = [f for f in files if f.suffix in ('.npz', '.safetensors', '.pt')]
    side = {}
    for f in files:
        if f.suffix == '.json':
            try:
                side[f.name] = json.loads(f.read_text())
            except Exception as e:
                log('unreadable side file %s: %s' % (f.name, e))
    jj = dict(job)
    for cfg in side.values():
        if isinstance(cfg, dict):
            if cfg.get('k'):
                jj['k'] = cfg['k']
            elif isinstance(cfg.get('activation_fn_kwargs'), dict) and cfg['activation_fn_kwargs'].get('k'):
                jj['k'] = cfg['activation_fn_kwargs']['k']
            if cfg.get('apply_b_dec_to_input'):
                jj['center_topk'] = True
    if jj.get('k') is None:
        jj['k'] = getattr(adapter.sae.cfg, 'k', None)
    nat = Native(jj, d_in, d_sae, weight_files, log=log)
    if job.get('conversion_func') == 'llama_scope':
        hp = {}
        for kk, v in side.items():
            if 'hyper' in kk and isinstance(v, dict):
                hp = v
        log('llama hyperparams subset: ' + json.dumps({k: v for k, v in hp.items()
                                                       if any(t in k.lower() for t in ('norm', 'bias', 'act', 'sparsity'))}))
        avg = hp.get('dataset_average_activation_norm') or {}
        nat.notes['llama_hyperparams'] = {k: v for k, v in hp.items() if not isinstance(v, (list, dict)) or len(str(v)) < 200}
        if isinstance(avg, dict) and avg:
            key = next((k for k in avg if 'in' in k.lower()), list(avg)[0])
            nat.scale_in = float(np.sqrt(d_in)) / float(avg[key])
            nat.scale_out = 1.0 / nat.scale_in
            nat.notes['llama_dataset_norm'] = avg
            nat.notes['llama_scale_in'] = nat.scale_in

    report = dict(job_id=job['job_id'], group=a.group, family=job['family'], repo_id=job['repo_id'],
                  repo_revision=job['repo_revision'], sae_id=job['sae_id'],
                  conversion_func=job.get('conversion_func'), loader=job['loader'],
                  architecture=str(adapter.cfg.get('architecture')), normalization=adapter.norm,
                  d_in=d_in, d_sae=d_sae, k=jj.get('k'),
                  apply_b_dec_to_input_used=bool(getattr(adapter.sae.cfg, 'apply_b_dec_to_input', False)),
                  encoder_centering_corrected=bool(getattr(adapter, 'encoder_centering_corrected', False)),
                  wdec_orientation_corrected=bool(getattr(adapter, 'wdec_orientation_corrected', False)),
                  native=nat.notes, native_keys=nat.keys, checks={},
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    probes = {k: np.asarray(X[v], dtype=np.float64) for k, v in rows.items()}
    probes['zero'] = np.zeros((1, d_in))
    probes['b_dec'] = adapter.sae.b_dec.detach().float().cpu().numpy()[None, :].astype(np.float64)
    space = 'sae_input' if adapter.norm != 'none' else 'raw'
    for name, x in probes.items():
        got = adapter.reconstruct(x)
        x_conv, y_conv = got[space]
        with torch.inference_mode():
            z = adapter.sae.encode(torch.as_tensor(x.copy(), dtype=torch.float32, device=a.device))
            z = (z.to_dense() if z.is_sparse else z).cpu().numpy().astype(np.float64)
        acts, y_nat, x_nat = nat.forward(x)
        # Same SAE math on the CONVERTER's own normalized input: isolates the equations from the
        # normalization convention, which matters only for the OpenAI GPT-2 release.
        _, y_same, _ = nat.forward(x, presplit=x_conv)
        A, B = z > 0, acts > 0
        inter = int((A & B).sum())
        union = int((A | B).sum())
        # A probe whose normalized input is identically zero cannot exercise the release's input rule;
        # it is recorded but excluded from the verdict.
        degenerate = bool(np.all(x_nat == 0) and not np.all(x == 0) or (len(x) == 1 and np.all(x == 0)
                                                                        and nat.kind == 'openai_topk'))
        report['checks'][name] = dict(
            n_rows=int(len(x)),
            input_space_rel_diff=rel(x_nat, x_conv),
            reconstruction_rel_diff=rel(y_nat, y_conv),
            reconstruction_rel_diff_same_input=rel(y_same, y_conv),
            reconstruction_max_abs_diff=float(np.abs(y_nat - y_conv).max()),
            activation_max_abs_diff=float(np.abs(acts - z).max()),
            active_jaccard=(inter / union if union else 1.0),
            l0_native=float(B.sum(1).mean()), l0_converted=float(A.sum(1).mean()),
            mean_row_norm=float(np.linalg.norm(x, axis=1).mean()),
            degenerate_for_this_release=degenerate)
        c = report['checks'][name]
        log('  %-9s recon_rel=%.3e jaccard=%.4f L0 %.1f/%.1f' %
            (name, c['reconstruction_rel_diff'], c['active_jaccard'], c['l0_native'], c['l0_converted']))

    if a.group == 'gpt2':
        x = probes['ordinary']
        sl = adapter.transform_input(x)
        oa = nat.norm_in(x)
        report['gpt2_normalization'] = dict(
            saelens_rule='x-mean(x); divide by torch.std(x) (unbiased, ddof=1)',
            paper_rule='x-mean(x); divide by ||x-mean(x)||/sqrt(d)  (OpenAI 2406.04093 §2.1)',
            max_abs_diff=float(np.abs(sl - oa).max()),
            median_norm_ratio=float(np.median(np.linalg.norm(sl, axis=1) / np.linalg.norm(oa, axis=1))),
            expected_ddof_ratio=float(np.sqrt(d_in / (d_in - 1))))
        log('  gpt2 normalization: ' + json.dumps(report['gpt2_normalization']))

    usable = {k: c for k, c in report['checks'].items() if not c.get('degenerate_for_this_release')}
    report['probes_excluded_as_degenerate'] = sorted(set(report['checks']) - set(usable))
    worst = max(c['reconstruction_rel_diff_same_input'] for c in usable.values())
    report['max_reconstruction_rel_diff_own_normalization'] = max(
        c['reconstruction_rel_diff'] for c in usable.values())
    jac = min(c['active_jaccard'] for c in usable.values())
    report['max_reconstruction_rel_diff'] = worst
    report['min_active_jaccard'] = jac
    report['verdict'] = ('MATCH' if worst < 1e-4 and jac > 0.999 else
                         'CLOSE' if worst < 1e-4 and jac > 0.95 else
                         'NEAR' if worst < 1e-2 and jac > 0.90 else 'MISMATCH')
    (out / ('native_%s.json' % job['job_id'])).write_text(json.dumps(report, indent=1, default=str))
    log('verdict %s  worst_rel=%.3e  min_jaccard=%.4f' % (report['verdict'], worst, jac))


if __name__ == '__main__':
    main()
