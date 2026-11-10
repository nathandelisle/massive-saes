"""Workstream B3: actual paired clean/corrupt latent patching (SVA and IOI).

Not resampling generic text and not a gradient proxy: for a recipient latent z_R, donor z_D and
coordinates S, z'_R,j = z_R,j + dose * (z_D,j - z_R,j) for j in S at every prompt position,
other coordinates and the recipient's routed PC1 held fixed. Dose 1 is literal replacement;
dose 2 is extrapolation and is labelled as such.

Eligibility is fixed from the NATIVE model (donor prefers its answer, recipient prefers its
own), never per SAE condition. Unrestricted and native-competent cohorts are both reported.
A task the native model cannot do (calibration recipient accuracy < 0.6) or with fewer than 200
eligible test pairs is UNSUPPORTED_TASK, with its counts.

Two questions, kept apart:
  1. patching-based importance -- rank every alive feature by its signed improvement of the
     donor-answer margin on DISCOVERY pairs; validate the top 50 on held-out pairs; an
     absolute-effect ranking is reported separately;
  2. usability of the GENERIC zero-ablation top 50 -- greedy budgeted edits chosen on discovery
     pairs only, scored on test pairs with foils: target switch, foil retention, joint success.
"""
import json
import time

import numpy as np
import torch

from . import common, modelio as M, tasks_ioi, tasks_sva
from .common import log

BUDGETS = [1, 3, 5, 10]
DOSES = [0.25, 0.5, 1.0, 2.0]
N_DISC, N_VAL, N_TEST = 256, 256, 1024
MIN_COMPETENCE, MIN_ELIGIBLE = 0.6, 200


def build_tasks(tok):
    out = {}
    tok.padding_side = "right"
    sva, sm = tasks_sva.build_sva(tok)
    if sva is not None:
        out["sva"] = (dict(calibration=sva["calibration"], train=sva["train"],
                           validation=sva["validation"], test=sva["test"]),
                      sm, tasks_sva.assert_disjoint(sva))
    else:
        out["sva"] = (None, sm, None)
    ioi, im = tasks_ioi.build_ioi(tok)
    out["ioi"] = (ioi, im, tasks_ioi.assert_disjoint(ioi) if ioi else None)
    return out


def _aligned(tok, rows):
    keep = []
    for it in rows:
        L = [len(tok(it[k])["input_ids"]) for k in ("prompt", "corrupt_prompt", "foil_prompt")]
        if len(set(L)) == 1:
            keep.append(it)
    return keep


class Encoded:
    """Token ids for donor / recipient / foil, grouped by length (all three equal per item)."""

    def __init__(self, tok, rows):
        self.rows = rows
        enc = lambda k: [tok(r[k])["input_ids"] for r in rows]
        self.don, self.rec, self.foil = enc("prompt"), enc("corrupt_prompt"), enc("foil_prompt")
        self.L = np.array([len(x) for x in self.rec])
        self.ans = np.array([r["answer_tok"] for r in rows])
        self.alt = np.array([r["alt_tok"] for r in rows])
        self.fans = np.array([r["foil_answer_tok"] for r in rows])
        self.falt = np.array([r["foil_alt_tok"] for r in rows])

    def groups(self, idx, max_items=128):
        idx = np.asarray(idx)
        for L in np.unique(self.L[idx]):
            g = idx[self.L[idx] == L]
            for i in range(0, len(g), max_items):
                yield g[i:i + max_items], int(L)

    def ids(self, which, sel):
        src = dict(don=self.don, rec=self.rec, foil=self.foil)[which]
        return torch.tensor([src[i] for i in sel])


@torch.no_grad()
def last_margin(suf, h, ans, alt, dtype):
    """Margin logit(ans) - logit(alt) at the LAST position, after the final soft-cap."""
    model = suf.model
    base = M._base_model(model)
    head = model.get_output_embeddings()
    cap = getattr(model.config, "final_logit_softcapping", None)
    B, T = h.shape[:2]
    am = torch.ones(B, T, dtype=torch.long)
    dummy = torch.zeros(B, T, suf.emb_dim, dtype=suf.emb_dtype, device=M.input_device(model))
    if len(suf.tail) == 0:
        hid = M._final_norm(base)(h.to(dtype))
    else:
        with suf._truncated(h.to(dtype)):
            hid = base(inputs_embeds=dummy, attention_mask=am.to(M.input_device(model)),
                       use_cache=False).last_hidden_state
    last = hid[:, -1]
    W = head.weight
    a = torch.as_tensor(ans, device=W.device)
    b = torch.as_tensor(alt, device=W.device)
    la = (last.to(W.device).float() * W[a].float()).sum(-1)
    lb = (last.to(W.device).float() * W[b].float()).sum(-1)
    if getattr(head, "bias", None) is not None:
        la, lb = la + head.bias[a].float(), lb + head.bias[b].float()
    if cap:
        la, lb = torch.tanh(la / cap) * cap, torch.tanh(lb / cap) * cap
    return (la - lb).cpu().numpy()


@torch.no_grad()
def native_margins(model, enc, idx):
    """Native donor / recipient / foil margins and eligibility."""
    out = dict(don=np.zeros(len(enc.rows)), rec=np.zeros(len(enc.rows)),
               foil=np.zeros(len(enc.rows)))
    for sel, L in enc.groups(idx):
        for w in ("don", "rec", "foil"):
            ids = enc.ids(w, sel).to(M.input_device(model))
            lg = model(ids, attention_mask=torch.ones_like(ids)).logits[:, -1].float()
            if w == "foil":
                m = (lg[torch.arange(len(sel)), torch.as_tensor(enc.fans[sel])]
                     - lg[torch.arange(len(sel)), torch.as_tensor(enc.falt[sel])])
            else:
                m = (lg[torch.arange(len(sel)), torch.as_tensor(enc.ans[sel])]
                     - lg[torch.arange(len(sel)), torch.as_tensor(enc.alt[sel])])
            out[w][sel] = m.cpu().numpy()
    return out


class SplicedPairs:
    """Spliced recipient residual, recipient / donor / foil latents, for one dictionary."""

    @torch.no_grad()
    def __init__(self, model, spec, dic, enc, idx, keep_cols=None):
        """keep_cols: store latents only for these features (memory: test sets hold thousands
        of pairs, and evaluation there only ever touches pool features)."""
        self.idx = np.asarray(idx)
        self.items = {}
        self.cols = None
        self.lookup = None
        if keep_cols is not None:
            self.cols = torch.as_tensor(np.asarray(sorted(set(int(c) for c in keep_cols))),
                                        device=dic.device)
            self.lookup = torch.full((dic.d_sae,), -1, dtype=torch.long, device=dic.device)
            self.lookup[self.cols] = torch.arange(len(self.cols), device=dic.device)
        for sel, L in enc.groups(idx, max_items=64):
            pack = {}
            for w in ("rec", "don", "foil"):
                ids = enc.ids(w, sel)
                h = M.prefix(model, spec, ids, torch.ones_like(ids))
                B, T, d = h.shape
                z, coef = dic.encode(h.reshape(-1, d))
                c = None if coef is None else coef
                if w in ("rec", "don"):
                    pack[w + "_r"] = dic.splice_fp32(z, c).reshape(B, T, d)
                    pack[w + "_h"] = h                          # model dtype: halves memory
                z = z.reshape(B, T, -1)
                pack[w + "_z"] = z if self.cols is None else z[..., self.cols]
                pack[w + "_coef"] = None if coef is None else coef.reshape(B, T)
            self.items[tuple(sel.tolist())] = (sel, L, pack)

    def batches(self):
        return self.items.values()


def _margins_of(suf, R, ans, alt, dtype, max_tokens=8192):
    n, T = R.shape[:2]
    step = max(1, max_tokens // max(T, 1))
    out = []
    for i in range(0, n, step):
        out.append(last_margin(suf, R[i:i + step], ans[i:i + step], alt[i:i + step], dtype))
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def spliced_baseline(suf, sp, enc, dtype):
    m = {}
    for sel, L, pk in sp.batches():
        mm = _margins_of(suf, pk["rec_r"], enc.ans[sel], enc.alt[sel], dtype)
        for i, s in enumerate(sel):
            m[int(s)] = float(mm[i])
    return m


@torch.no_grad()
def patch_importance(suf, dic, sp, enc, base, alive, dtype, max_tokens=8192):
    """Every alive feature that differs between donor and recipient on a pair is patched alone.
    Returns signed and absolute mean improvements over the pairs (0 where it does not differ)."""
    d_sae = dic.d_sae
    tot = np.zeros(d_sae)
    totabs = np.zeros(d_sae)
    npairs = 0
    al = torch.as_tensor(np.asarray(alive, bool), device=dic.device)
    for sel, L, pk in sp.batches():
        diff = (pk["don_z"] - pk["rec_z"])                         # [B,T,d_sae]
        changed = (diff != 0).any(1) & al[None, :]                 # [B,d_sae]
        b_idx, f_idx = torch.nonzero(changed, as_tuple=True)
        n = len(b_idx)
        T = pk["rec_r"].shape[1]
        step = max(1, max_tokens // max(T, 1))
        for i in range(0, n, step):
            bb, ff = b_idx[i:i + step], f_idx[i:i + step]
            dz = diff[bb, :, ff]                                   # [n,T]
            R = pk["rec_r"][bb] + dz[..., None] * dic.W_dec[ff][:, None, :]
            mm = last_margin(suf, R, enc.ans[sel][bb.cpu().numpy()], enc.alt[sel][bb.cpu().numpy()], dtype)
            b0 = np.array([base[int(sel[int(x)])] for x in bb.cpu().numpy()])
            eff = mm - b0
            np.add.at(tot, ff.cpu().numpy(), eff)
            np.add.at(totabs, ff.cpu().numpy(), np.abs(eff))
        npairs += len(sel)
    return tot / max(npairs, 1), totabs / max(npairs, 1), npairs


def _apply_mask(pk, dic, bsel, mask, dose, donor="don", lookup=None):
    """Recipient spliced residual with features `mask` patched from `donor` at `dose`."""
    R = pk["rec_r"][bsel].clone()
    if len(mask):
        m = torch.as_tensor(mask, device=dic.device)
        loc = m if lookup is None else lookup[m]
        if lookup is not None and bool((loc < 0).any()):
            raise KeyError("mask feature not stored in this SplicedPairs")
        dz = (pk[donor + "_z"][bsel][..., loc] - pk["rec_z"][bsel][..., loc]) * dose
        R = R + dz @ dic.W_dec[m]
    return R


@torch.no_grad()
def evaluate_mask(suf, dic, sp, enc, mask, dose, dtype, eligible):
    """Per-item target margin after the donor patch, and foil margin after the foil patch."""
    tgt, foil = {}, {}
    for sel, L, pk in sp.batches():
        keep = np.array([eligible[int(s)] for s in sel])
        if not keep.any():
            continue
        bsel = torch.as_tensor(np.nonzero(keep)[0], device=dic.device)
        s_k = sel[keep]
        Rt = _apply_mask(pk, dic, bsel, mask, dose, "don", sp.lookup)
        Rf = _apply_mask(pk, dic, bsel, mask, dose, "foil", sp.lookup)
        mt = _margins_of(suf, Rt, enc.ans[s_k], enc.alt[s_k], dtype)
        mf = _margins_of(suf, Rf, enc.fans[s_k], enc.falt[s_k], dtype)
        for i, s in enumerate(s_k):
            tgt[int(s)], foil[int(s)] = float(mt[i]), float(mf[i])
    return tgt, foil


def _rates(tgt, foil):
    ks = sorted(tgt)
    if not ks:
        return dict(n=0)
    t = np.array([tgt[k] > 0 for k in ks])
    f = np.array([foil[k] > 0 for k in ks])
    return dict(n=len(ks), target_switch=float(t.mean()), foil_retention=float(f.mean()),
                joint_success=float((t & f).mean()),
                mean_target_margin=float(np.mean([tgt[k] for k in ks])),
                mean_foil_margin=float(np.mean([foil[k] for k in ks])))


def greedy(suf, dic, sp, enc, pool, dtype, eligible, bmax=10):
    """Greedy forward selection from `pool` maximising discovery joint success (dose 1),
    ties broken by mean target margin. Discovery data only."""
    chosen, trace = [], []
    for step in range(bmax):
        best, best_key = None, None
        for j in pool:
            if j in chosen:
                continue
            tgt, foil = evaluate_mask(suf, dic, sp, enc, chosen + [int(j)], 1.0, dtype, eligible)
            r = _rates(tgt, foil)
            key = (r.get("joint_success", 0), r.get("mean_target_margin", -1e9))
            if best_key is None or key > best_key:
                best, best_key = int(j), key
        if best is None:
            break
        chosen.append(best)
        trace.append(dict(step=step + 1, added=best, joint=best_key[0], margin=best_key[1]))
    return chosen, trace


@torch.no_grad()
def controls(suf, dic, sp, enc, dtype, eligible):
    """No edit, self-donor, complete residual patch, PC1-only and perpendicular patches, and the
    clean-vs-corrupt PC1 coefficient difference."""
    res = {k: {} for k in ("none", "self", "full_residual", "pc1_only", "perp_only")}
    dcoef = []
    v1 = dic.v1
    for sel, L, pk in sp.batches():
        keep = np.array([eligible[int(s)] for s in sel])
        if not keep.any():
            continue
        b = torch.as_tensor(np.nonzero(keep)[0], device=dic.device)
        s_k = sel[keep]
        R = pk["rec_r"][b]
        cR = (pk["rec_h"][b].float() - dic.mu) @ v1
        cD = (pk["don_h"][b].float() - dic.mu) @ v1
        dcoef.append((cD - cR).abs().cpu().numpy().ravel())
        Rd = pk["don_r"][b]
        variants = dict(none=R, self=R.clone(), full_residual=Rd,
                        pc1_only=R + (cD - cR)[..., None] * v1,
                        perp_only=R + ((Rd - R) - ((Rd - R) @ v1)[..., None] * v1))
        for k, RR in variants.items():
            mm = _margins_of(suf, RR, enc.ans[s_k], enc.alt[s_k], dtype)
            for i, s in enumerate(s_k):
                res[k][int(s)] = float(mm[i])
    out = {k: dict(n=len(v), target_switch=float(np.mean([x > 0 for x in v.values()])) if v else None,
                   mean_margin=float(np.mean(list(v.values()))) if v else None)
           for k, v in res.items()}
    dc = np.concatenate(dcoef) if dcoef else np.zeros(1)
    out["pc1_coef_abs_diff_clean_vs_corrupt"] = dict(mean=float(dc.mean()), median=float(np.median(dc)),
                                                      p95=float(np.quantile(dc, .95)),
                                                      frac_zero=float((dc == 0).mean()))
    return out, res


def run(model, tok, spec, dic, alive, zero_score, pc1_cos2, cond, out, rng_seed=0):
    """All B3 measurements for one dictionary. Writes <out>/<task>.json and per-item npz."""
    out.mkdir(parents=True, exist_ok=True)
    suf = M.Suffix(model, spec)
    dtype = next(M.blocks(model, spec)[0].parameters()).dtype
    tasks = build_tasks(tok)
    summary = {}
    for tname, (splits, meta, dj) in tasks.items():
        f = out / ("%s.json" % tname)
        if common.fresh(f.parent, f.name, "wsx"):
            summary[tname] = json.loads(f.read_text())
            continue
        t0 = time.time()
        rep = dict(task=tname, meta=meta, disjointness=dj, condition=cond)
        if splits is None:
            rep["status"] = "UNSUPPORTED_TASK"
            common.write_json(f, rep)
            common.pin(f.parent, f.name, "wsx")
            continue
        splits = {k: _aligned(tok, v) for k, v in splits.items()}
        rep["aligned_counts"] = {k: len(v) for k, v in splits.items()}
        rng = np.random.default_rng(20261005)
        sel = {}
        for k, n in (("calibration", 512), ("train", 2048), ("validation", 1024), ("test", 4096)):
            rows = splits.get(k, [])
            ix = rng.permutation(len(rows))[:n]
            sel[k] = [rows[i] for i in ix]
        encs = {k: Encoded(tok, v) for k, v in sel.items()}
        nat = {k: native_margins(model, e, np.arange(len(e.rows))) for k, e in encs.items()}
        elig = {k: (nat[k]["don"] > 0) & (nat[k]["rec"] < 0) for k in encs}
        cal_acc = float(np.mean(nat["calibration"]["rec"] < 0)) if len(encs["calibration"].rows) else 0.0
        rep["native"] = {k: dict(n=int(len(e.rows)), donor_acc=float((nat[k]["don"] > 0).mean()),
                                 recipient_acc=float((nat[k]["rec"] < 0).mean()),
                                 foil_acc=float((nat[k]["foil"] > 0).mean()),
                                 eligible=int(elig[k].sum()))
                         for k, e in encs.items()}
        rep["competence_threshold"] = MIN_COMPETENCE
        if cal_acc < MIN_COMPETENCE or elig["test"].sum() < MIN_ELIGIBLE:
            rep["status"] = "UNSUPPORTED_TASK"
            rep["why"] = dict(calibration_recipient_acc=cal_acc, eligible_test=int(elig["test"].sum()))
            common.write_json(f, rep)
            common.pin(f.parent, f.name, "wsx")
            continue
        rep["status"] = "ok"
        # spliced task accuracy (outcome, not a filter)
        disc = np.nonzero(elig["train"])[0][:N_DISC]
        val = np.nonzero(elig["validation"])[0][:N_VAL]
        test_all = np.arange(len(encs["test"].rows))
        sp = {"train": SplicedPairs(model, spec, dic, encs["train"], disc)}
        # 1. patching-based importance (discovery), validated on held-out pairs
        base_disc = spliced_baseline(suf, sp["train"], encs["train"], dtype)
        signed, absval, npairs = patch_importance(suf, dic, sp["train"], encs["train"], base_disc,
                                                  alive, dtype)
        score = np.where(alive, signed, np.nan)
        top = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:50]
        sp["validation"] = SplicedPairs(model, spec, dic, encs["validation"], val, keep_cols=top)
        base_val = spliced_baseline(suf, sp["validation"], encs["validation"], dtype)
        val_eff = []
        elv = np.zeros(len(encs["validation"].rows), bool); elv[val] = True
        for j in top:
            tg, _ = evaluate_mask(suf, dic, sp["validation"], encs["validation"], [int(j)], 1.0, dtype, elv)
            val_eff.append(float(np.mean([tg[k] - base_val[k] for k in tg])) if tg else float("nan"))
        rep["patch_importance"] = dict(n_discovery_pairs=npairs, top50=[int(x) for x in top],
                                       top50_validation_effect=val_eff,
                                       top50_pc1_cos2=[float(pc1_cos2[j]) for j in top])
        common.save_npz(out / ("%s_patch_scores.npz" % tname), signed=signed, absolute=absval,
                        alive=np.asarray(alive))
        # 2. usability of the generic zero-ablation top 50
        pools = {}
        zs = np.where(np.asarray(alive, bool) & np.isfinite(zero_score), zero_score, -np.inf)
        if cond == "standard":
            pools["standard"] = np.argsort(-zs)[:50]
            ok = alive & (pc1_cos2 < 0.01)
            pools["posthoc"] = np.argsort(-np.where(ok, zero_score, -np.inf))[:50]
        else:
            pools["bypass"] = np.argsort(-zs)[:50]
        alive_ix = np.nonzero(alive)[0]
        for r_ in range(3):
            pools["random_%d" % r_] = np.random.default_rng(rng_seed * 100 + r_).choice(
                alive_ix, size=50, replace=False)
        allpool = np.unique(np.concatenate([np.asarray(v) for v in pools.values()]))
        sp["test"] = SplicedPairs(model, spec, dic, encs["test"], test_all, keep_cols=allpool)
        base_test = spliced_baseline(suf, sp["test"], encs["test"], dtype)
        rep["spliced_recipient_acc_test"] = float(np.mean([v < 0 for v in base_test.values()]))
        eld = np.zeros(len(encs["train"].rows), bool); eld[disc] = True
        elt_c = elig["test"]
        elt_u = np.ones(len(encs["test"].rows), bool)
        usab = {}
        items = {}
        for pname, pool in pools.items():
            chosen, trace = greedy(suf, dic, sp["train"], encs["train"], [int(x) for x in pool],
                                   dtype, eld, bmax=max(BUDGETS))
            usab[pname] = dict(pool=[int(x) for x in pool], greedy_trace=trace, results={})
            for b in BUDGETS:
                for dose in DOSES:
                    for cohort, el in (("eligible", elt_c), ("unrestricted", elt_u)):
                        if cohort == "unrestricted" and not (b == 5 and dose == 1.0):
                            continue
                        tg, fo = evaluate_mask(suf, dic, sp["test"], encs["test"], chosen[:b],
                                               dose, dtype, el)
                        usab[pname]["results"]["b%d_d%g_%s" % (b, dose, cohort)] = _rates(tg, fo)
                        if b == 5 and dose == 1.0 and cohort == "eligible":
                            items[pname] = (tg, fo)
            usab[pname]["mask_b5"] = chosen[:5]
            usab[pname]["mask_b10"] = chosen[:10]
        rep["usability"] = usab
        ctl, _ = controls(suf, dic, sp["test"], encs["test"], dtype, elt_c)
        rep["controls"] = ctl
        # per-item records for example-cluster bootstrap
        keys = sorted(next(iter(items.values()))[0]) if items else []
        common.save_npz(out / ("%s_items.npz" % tname), item=np.array(keys),
                        template=np.array([encs["test"].rows[k]["template_family"] for k in keys]),
                        **{"%s_target" % p: np.array([v[0][k] for k in keys]) for p, v in items.items()},
                        **{"%s_foil" % p: np.array([v[1][k] for k in keys]) for p, v in items.items()},
                        base=np.array([base_test[k] for k in keys]))
        rep["seconds"] = time.time() - t0
        common.write_json(f, rep)
        common.pin(f.parent, f.name, "wsx")
        summary[tname] = rep
        log("patch", "%s %s done in %.0fs" % (tname, cond, time.time() - t0))
    return summary
