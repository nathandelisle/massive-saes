"""Workstream C, part 1: interpretation contexts for one dictionary (no judge here).

Features: the top 100 by the INDEPENDENT exhaustive zero-ablation ranking (rank partition) and
100 uniformly sampled alive features (seeded by model/condition/seed). Overlap is allowed and
recorded; a feature in both sets is one record with two memberships.

Pools are document-disjoint by construction: `explain` and `ieval` are different article sets,
and neither overlaps `rank` (where the ranking came from) or `cal`.

  explanation set  top 10 activations + 5 from each of 4 nonzero-activation quartiles
                   (quartile edges fitted on the EXPLAIN pool, never on evaluation data)
  evaluation set   50 positives stratified across the same quartiles
                   50 random negatives: random valid positions where the TARGET latent is 0
                   50 hard negatives: positive contexts of the 10 selected features with the
                     highest decoder cosine to the target, each VERIFIED inactive on the target
  dense features   if the target fires on > 50% of eval positions, verified negatives may be
                   unobtainable: the feature is kept, flagged, and gets a separately specified
                   high-vs-low task (top quartile vs bottom quartile) instead of manufactured
                   negatives

Every context stores token ids (left window, the model-visible text at the target), the target
offset, activation, position, quartile bucket, article id, row id, label and split. Strings are
decoded here with the subject model's tokenizer so the judge never needs it.
"""
import json
import time
import zlib

import numpy as np
import torch

from . import common, modelio as M, partitions as P
from .common import log

LEFT = 32           # tokens of left context shown (the text available to the model)
RIGHT = 8           # two-sided display window: a separately labelled sensitivity analysis
CAP = 3000          # reservoir cap per feature per pool
N_TOP, N_PER_Q = 10, 5
N_POS, N_NEG, N_HARD = 50, 50, 50


def select_features(alive, zero_score, model, cond, seed, n_top=100, n_rand=100):
    a = np.asarray(alive, bool)
    s = np.where(a & np.isfinite(zero_score), zero_score, -np.inf)
    top = [int(x) for x in np.argsort(-s)[:n_top]]
    rng = np.random.default_rng(zlib.crc32(("%s|%s|%d" % (model, cond, seed)).encode()))
    rnd = [int(x) for x in rng.choice(np.nonzero(a)[0], size=min(n_rand, int(a.sum())), replace=False)]
    feats = sorted(set(top) | set(rnd))
    member = {f: (["top"] if f in top else []) + (["random"] if f in rnd else []) for f in feats}
    return feats, member, top, rnd


class _Res:
    """Uniform reservoir of (row, pos, act, zsel-vector) entries; random keys -> exact uniform."""

    def __init__(self, cap, nfeat, rng):
        self.cap, self.rng = cap, rng
        self.key = np.zeros(0)
        self.row = np.zeros(0, np.int64)
        self.pos = np.zeros(0, np.int64)
        self.act = np.zeros(0, np.float32)
        self.vec = np.zeros((0, nfeat), np.float16)

    def offer(self, row, pos, act, vec):
        k = self.rng.random(len(row))
        self.key = np.concatenate([self.key, k])
        self.row = np.concatenate([self.row, row])
        self.pos = np.concatenate([self.pos, pos])
        self.act = np.concatenate([self.act, act])
        self.vec = np.concatenate([self.vec, vec.astype(np.float16)])
        if len(self.key) > self.cap:
            o = np.argsort(self.key)[:self.cap]
            self.key, self.row, self.pos = self.key[o], self.row[o], self.pos[o]
            self.act, self.vec = self.act[o], self.vec[o]


@torch.no_grad()
def scan(model, spec, dic, part, feats, seed, neg_per_batch=512, max_tokens=8192):
    """One pass over a partition: per-feature positive reservoirs, random-negative reservoirs,
    top-k heaps, and firing counts."""
    rng = np.random.default_rng(seed)
    F = torch.as_tensor(feats, device=dic.device)
    nf = len(feats)
    pos_res = [_Res(CAP, nf, rng) for _ in feats]
    neg_res = [_Res(CAP, nf, rng) for _ in feats]
    fires = np.zeros(nf, np.int64)
    n_valid = 0
    for sel, ids, am in part.batches(max_tokens=max_tokens):
        h = M.prefix(model, spec, ids, am)
        B, T, d = h.shape
        z, _ = dic.encode(h.reshape(-1, d))
        zs = z[:, F].reshape(B, T, nf)
        valid = am.bool().to(zs.device)
        zs = zs * valid[..., None]
        n_valid += int(valid.sum())
        zc = zs.cpu().numpy()
        rows = part.row[sel]
        b_i, t_i, f_i = np.nonzero(zc > 0)
        fires += np.bincount(f_i, minlength=nf)
        order = np.argsort(f_i, kind="stable")
        b_i, t_i, f_i = b_i[order], t_i[order], f_i[order]
        bounds = np.searchsorted(f_i, np.arange(nf + 1))
        for j in range(nf):
            a, b = bounds[j], bounds[j + 1]
            if a == b:
                continue
            bb, tt = b_i[a:b], t_i[a:b]
            pos_res[j].offer(rows[bb], tt, zc[bb, tt, j], zc[bb, tt])
        # random valid positions as negative candidates (verified per feature)
        vb, vt = np.nonzero(valid.cpu().numpy())
        if len(vb):
            pick = rng.choice(len(vb), size=min(neg_per_batch, len(vb)), replace=False)
            bb, tt = vb[pick], vt[pick]
            vecs = zc[bb, tt]                                     # [n, nf]
            for j in range(nf):
                m = vecs[:, j] == 0
                if m.any():
                    neg_res[j].offer(rows[bb[m]], tt[m], np.zeros(int(m.sum()), np.float32), vecs[m])
    return pos_res, neg_res, fires, n_valid


def _window(row_ids, pos, tok):
    lo = max(0, pos - LEFT)
    left = row_ids[lo:pos + 1]
    right = row_ids[pos + 1:pos + 1 + RIGHT]
    return dict(left_ids=[int(x) for x in left], right_ids=[int(x) for x in right],
                target_offset=int(pos - lo),
                left_text=tok.decode(left[:-1]), target_text=tok.decode(left[-1:]),
                right_text=tok.decode(right))


def _dedupe(rows, poss, min_gap=LEFT):
    keep, seen = [], {}
    for i, (r, p) in enumerate(zip(rows, poss)):
        ok = all(abs(p - q) >= min_gap for q in seen.get(int(r), []))
        if ok:
            keep.append(i)
            seen.setdefault(int(r), []).append(int(p))
    return np.asarray(keep, np.int64)


def build(model, tok, spec, dic, feats, member, out, cond, seed, max_seqs=None):
    out.mkdir(parents=True, exist_ok=True)
    f = out / "contexts.json"
    if common.fresh(f.parent, f.name, "wsx"):
        return json.loads(f.read_text())
    t0 = time.time()
    rows_text = P._rows()
    pex = P.Partition("explain", tok, rows=rows_text, max_seqs=max_seqs)
    pev = P.Partition("ieval", tok, rows=rows_text, max_seqs=max_seqs)
    ex_pos, _, ex_fires, ex_n = scan(model, spec, dic, pex, feats, seed=1)
    ev_pos, ev_neg, ev_fires, ev_n = scan(model, spec, dic, pev, feats, seed=2)
    ids_by_row = {}
    for part in (pex, pev):
        for i, r in enumerate(part.row):
            ids_by_row[int(r)] = part.ids[i][:int(part.lengths[i])].tolist()
    art = {}
    for part in (pex, pev):
        for r, d_ in zip(part.row, part.doc):
            art[int(r)] = int(d_)
    W = dic.W_dec / dic.W_dec.norm(dim=1, keepdim=True)
    Fi = torch.as_tensor(feats, device=W.device)
    cos = (W[Fi] @ W[Fi].T).cpu().numpy()
    rng = np.random.default_rng(zlib.crc32(("ctx|%s|%d" % (cond, seed)).encode()))
    recs = []
    for j, feat in enumerate(feats):
        rec = dict(feature=int(feat), membership=member[feat],
                   explain_fire_rate=float(ex_fires[j] / max(ex_n, 1)),
                   ieval_fire_rate=float(ev_fires[j] / max(ev_n, 1)))
        pr = ex_pos[j]
        if len(pr.act) == 0:
            rec["status"] = "NO_EXPLAIN_ACTIVATIONS"
            recs.append(rec)
            continue
        qs = np.quantile(pr.act, [0.25, 0.5, 0.75])
        rec["quartile_edges_from_explain"] = [float(x) for x in qs]
        bucket = lambda a: int(np.searchsorted(qs, a, side="right"))
        # explanation set
        o = np.argsort(-pr.act)
        k = _dedupe(pr.row[o], pr.pos[o])
        top_i = o[k][:N_TOP]
        exp = []
        for i in top_i:
            exp.append(dict(kind="top", act=float(pr.act[i]), bucket=bucket(pr.act[i]),
                            row=int(pr.row[i]), pos=int(pr.pos[i]), article=art[int(pr.row[i])],
                            **_window(ids_by_row[int(pr.row[i])], int(pr.pos[i]), tok)))
        used = set((int(pr.row[i]), int(pr.pos[i])) for i in top_i)
        for qb in range(4):
            cand = [i for i in range(len(pr.act)) if bucket(pr.act[i]) == qb
                    and (int(pr.row[i]), int(pr.pos[i])) not in used]
            for i in rng.permutation(cand)[:N_PER_Q]:
                exp.append(dict(kind="quartile%d" % qb, act=float(pr.act[i]), bucket=qb,
                                row=int(pr.row[i]), pos=int(pr.pos[i]), article=art[int(pr.row[i])],
                                **_window(ids_by_row[int(pr.row[i])], int(pr.pos[i]), tok)))
        rec["explain"] = exp
        # evaluation set
        er, nr = ev_pos[j], ev_neg[j]
        ev = []
        if len(er.act):
            per_q = {qb: [i for i in range(len(er.act)) if bucket(er.act[i]) == qb] for qb in range(4)}
            take = []
            for qb in range(4):
                take += list(rng.permutation(per_q[qb])[: N_POS // 4 + (1 if qb < N_POS % 4 else 0)])
            if len(take) < N_POS:
                rest = [i for i in range(len(er.act)) if i not in set(take)]
                take += list(rng.permutation(rest)[: N_POS - len(take)])
            for i in take:
                ev.append(dict(label=1, kind="positive", act=float(er.act[i]), bucket=bucket(er.act[i]),
                               row=int(er.row[i]), pos=int(er.pos[i]), article=art[int(er.row[i])],
                               **_window(ids_by_row[int(er.row[i])], int(er.pos[i]), tok)))
        dense = rec["ieval_fire_rate"] > 0.5
        rec["dense_flag"] = bool(dense)
        ni = rng.permutation(len(nr.row))[:N_NEG]
        for i in ni:
            ev.append(dict(label=0, kind="random_negative", act=0.0, row=int(nr.row[i]),
                           pos=int(nr.pos[i]), article=art[int(nr.row[i])],
                           **_window(ids_by_row[int(nr.row[i])], int(nr.pos[i]), tok)))
        # hard negatives: positives of the most similar selected features, target verified 0
        sims = np.argsort(-cos[j])
        hard = []
        for o2 in sims:
            if o2 == j:
                continue
            r2 = ev_pos[o2]
            m = np.nonzero(r2.vec[:, j] == 0)[0]
            for i in rng.permutation(m)[: N_HARD - len(hard)]:
                hard.append(dict(label=0, kind="hard_negative", act=0.0, source_feature=int(feats[o2]),
                                 source_cos=float(cos[j, o2]), row=int(r2.row[i]), pos=int(r2.pos[i]),
                                 article=art[int(r2.row[i])],
                                 **_window(ids_by_row[int(r2.row[i])], int(r2.pos[i]), tok)))
            if len(hard) >= N_HARD:
                break
        ev += hard
        if dense:
            lo = [i for i in range(len(er.act)) if bucket(er.act[i]) == 0]
            for i in rng.permutation(lo)[:N_NEG]:
                ev.append(dict(label=0, kind="low_activation_for_dense", act=float(er.act[i]), bucket=0,
                               row=int(er.row[i]), pos=int(er.pos[i]), article=art[int(er.row[i])],
                               **_window(ids_by_row[int(er.row[i])], int(er.pos[i]), tok)))
        rec["eval"] = ev
        rec["counts"] = dict(positives=sum(e["kind"] == "positive" for e in ev),
                             random_negatives=sum(e["kind"] == "random_negative" for e in ev),
                             hard_negatives=sum(e["kind"] == "hard_negative" for e in ev),
                             low_for_dense=sum(e["kind"] == "low_activation_for_dense" for e in ev))
        rec["status"] = "OK" if rec["counts"]["positives"] >= 10 else "TOO_FEW_POSITIVES"
        recs.append(rec)
    res = dict(model=spec.key, condition=cond, seed=seed, n_features=len(recs),
               explain_tokens=int(ex_n), ieval_tokens=int(ev_n), left=LEFT, right=RIGHT,
               features=recs, seconds=time.time() - t0)
    common.write_json(f, res)
    common.pin(f.parent, f.name, "wsx")
    log("ctx", "%d features, %.0fs" % (len(recs), time.time() - t0))
    return res
