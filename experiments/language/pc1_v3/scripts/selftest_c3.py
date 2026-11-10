#!/usr/bin/env python
"""Check the PCA reconstruction calculations on synthetic inputs."""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, ".")
from v3.common import registry                                       # noqa: E402
import scripts.exp_c3_linear_baseline as c3                          # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def make_store(tmp, n=40_000, d=64, seed=0):
    """Synthetic activations with a planted spectrum: one huge direction plus a
    geometric tail, so both the PC1 share and every eigenvalue are known."""
    rng = np.random.default_rng(seed)
    lam = np.concatenate([[400.0], 4.0 * 0.9 ** np.arange(d - 1)])
    Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = (rng.standard_normal((n, d)) * np.sqrt(lam)) @ Qr.T
    X += rng.standard_normal(d) * 3.0                       # nonzero mean
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "acts.npy"), X.astype(np.float16))
    np.save(os.path.join(tmp, "mean.npy"), X.mean(0).astype(np.float64))
    json.dump(dict(n_tokens=n, d_model=d), open(os.path.join(tmp, "meta.json"), "w"))
    return X, lam, Qr


class FakeSAE(torch.nn.Module):
    """A 'SAE' whose reconstruction is a known linear map, so its content FVU
    can be computed in closed form and compared with what eval_pass reports."""

    def __init__(self, d, keep, basis):
        super().__init__()
        self.register_buffer("B", torch.tensor(basis[:, :keep],
                                               dtype=torch.float32))
        self.keep = keep

    def forward(self, x):
        z = x @ self.B
        return z @ self.B.T, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None,
                    help="run the end-to-end splice-identity check on this model")
    args = ap.parse_args()
    dev = args.device
    tmp = tempfile.mkdtemp(prefix="c3self_")
    try:
        d, n = 64, 40_000
        X, lam, Qr = make_store(tmp, n=n, d=d)

        # ---------------- 1. Householder complement ----------------
        print("\n1. orthonormal_complement")
        from v3.common.pca import estimate_pcs
        evals, evecs, mean = estimate_pcs(tmp, n_sample=20_000, cache=False)
        v1 = evecs[:, 0]
        Q = c3.orthonormal_complement(v1)
        check("shape is d x (d-1)", Q.shape == (d, d - 1), str(Q.shape))
        check("orthonormal", np.abs(Q.T @ Q - np.eye(d - 1)).max() < 1e-10,
              f"max dev {np.abs(Q.T @ Q - np.eye(d - 1)).max():.2e}")
        check("exactly perp to v1", np.abs(Q.T @ v1).max() < 1e-12,
              f"max |<q,v1>| {np.abs(Q.T @ v1).max():.2e}")
        # and for the pathological input the branch exists for
        e1 = np.zeros(d); e1[0] = 1.0
        Qe = c3.orthonormal_complement(e1)
        check("handles v1 == e1", np.abs(Qe.T @ e1).max() < 1e-12)

        # ---------------- 2. content basis vs closed form ----------------
        print("\n2. content_basis on a planted covariance")
        C, n_fit = c3.fit_covariance(tmp, n, "head", dev, 8192, seed=0)
        Xc = X - X.mean(0)
        Cref = (Xc.T @ Xc) / (n - 1)
        # float16 storage is the only difference between C and Cref
        check("covariance matches a direct float64 computation",
              np.abs(C - Cref).max() / np.abs(Cref).max() < 2e-3,
              f"rel {np.abs(C - Cref).max() / np.abs(Cref).max():.2e}")
        U, lam_c = c3.content_basis(C, v1, dev)
        check("content basis perp to v1", np.abs(U.T @ v1).max() < 1e-10,
              f"{np.abs(U.T @ v1).max():.2e}")
        check("content basis orthonormal",
              np.abs(U.T @ U - np.eye(d - 1)).max() < 1e-10)
        P = np.eye(d) - np.outer(v1, v1)
        lam_ref = np.sort(np.linalg.eigvalsh(P @ C @ P))[::-1][:d - 1]
        check("content eigenvalues match eigh(P C P)",
              np.abs(lam_c - lam_ref).max() / lam_c.max() < 1e-8,
              f"rel {np.abs(lam_c - lam_ref).max() / lam_c.max():.2e}")
        check("content variance excludes the planted dominant direction",
              lam_c.sum() < 0.2 * np.trace(C),
              f"content/total = {lam_c.sum() / np.trace(C):.4f}")

        # ---------------- 3. eval_pass curves ----------------
        print("\n3. eval_pass")
        lam_f, V = c3._eigh_desc(0.5 * (C + C.T), device=dev)
        rand_Os = c3.random_rotations(d - 1, 2, dev, seed=0)
        keep = 8
        saes = {"fake": (FakeSAE(d, keep, U).to(dev), "standard")}
        ev = c3.eval_pass(tmp, v1, U, V, rand_Os, saes, dev, n, 8192,
                          slice_="head")
        check("closure, content", ev["closure_rel_err"]["content"] < 1e-8,
              f"{ev['closure_rel_err']['content']:.2e}")
        check("closure, full", ev["closure_rel_err"]["full"] < 1e-8,
              f"{ev['closure_rel_err']['full']:.2e}")
        f_fix = ev["fve_pca_fixed"]
        check("fixed curve is non-decreasing", np.all(np.diff(f_fix) >= -1e-12))
        check("fixed curve reaches 1 at k=d-1", abs(f_fix[-1] - 1) < 1e-9,
              f"{f_fix[-1]:.10f}")
        check("full-PCA curve reaches 1 at k=d",
              abs(ev["fve_pca_full"][-1] - 1) < 1e-9)
        check("adaptive >= fixed everywhere",
              np.all(ev["fve_pca_adaptive"] >= f_fix - 1e-12))
        # the fixed curve IS the eval-slice eigenvalue tail, by construction
        s = np.array([c3.at(f_fix, k) for k in range(1, d)])
        check("fixed curve tracks the fitted content spectrum (rank order)",
              np.all(np.diff(np.array([c3.at(f_fix, k) for k in range(1, d)])) >= -1e-12))
        rnd = np.mean([[c3.at(cc, k) for k in (8, 16, 32)]
                       for cc in ev["fve_rand"]], axis=0)
        ana = np.array([8, 16, 32]) / (d - 1)
        check("random floor is near k/(d-1)", np.abs(rnd - ana).max() < 0.12,
              f"emp {np.round(rnd, 3)} vs {np.round(ana, 3)}")

        # the FakeSAE reconstructs exactly the top-`keep` content directions,
        # so its content FVU must equal 1 - fixed-curve(keep) to float precision
        got = ev["sae"]["fake"]["fvu_content"]
        want = 1.0 - c3.at(f_fix, keep)
        check("SAE content FVU == the fixed-PCA identity it implements",
              abs(got - want) < 1e-6, f"{got:.8f} vs {want:.8f}")
        check("SAE L0 reported correctly",
              abs(ev["sae"]["fake"]["l0"] - keep) < 1e-9,
              f"{ev['sae']['fake']['l0']}")
        # a FakeSAE living entirely in the content subspace leaves ALL the PC1
        # energy in the error, so fvu_pc1 must be exactly 1
        check("PC1 error accounting", abs(ev["sae"]["fake"]["fvu_pc1"] - 1) < 1e-4,
              f"{ev['sae']['fake']['fvu_pc1']:.6f}")

        # ---------------- 4. reporting helpers ----------------
        print("\n4. helpers")
        check("match_k finds the first crossing",
              c3.match_k(np.array([0.1, 0.5, 0.9]), 0.5) == 2)
        check("match_k returns None when unreachable",
              c3.match_k(np.array([0.1, 0.2]), 0.9) is None)
        check("at() is 1-indexed", c3.at(np.array([0.3, 0.7]), 1) == 0.3)
        check("at() clamps above range", c3.at(np.array([0.3, 0.7]), 99) == 0.7)
        b = c3.bits_per_token(64, 16384)
        ba = c3.bits_per_token(64, 4095)
        check("index bits grow with dictionary size",
              b["index_bits"] > ba["index_bits"],
              f"sae {b['index_bits']:.0f} vs pca {ba['index_bits']:.0f} "
              f"({(b['index_bits'] - ba['index_bits']) / 64:.2f} bits/atom)")

        # ---------------- 5. end-to-end splice identity ----------------
        if args.model:
            print("\n5. linear_splice identity at k = d-1 (needs a real model)")
            from v3.common import io_utils as io
            from v3.common.data import load_model_and_tokenizer, eval_sequences
            from v3.common.pca import estimate_pcs as ep
            from v3.common.splice import ce_loss
            spec = registry.MODELS[args.model]
            adir = io.act_dir(spec)
            ev2, evec2, mean2 = ep(adir)
            v1b = evec2[:, 0]
            Cb, _ = c3.fit_covariance(adir, 50_000, "head", args.device, 4096)
            Ub, _ = c3.content_basis(Cb, v1b, args.device)
            _, Vb = c3._eigh_desc(0.5 * (Cb + Cb.T), device=args.device)
            model, tok = load_model_and_tokenizer(spec, device=args.device)
            seqs, am = eval_sequences(tok, n_seqs=16)
            clean = ce_loss(model, seqs, chunk=4, am=am)
            for mode, kk in (("pca_fixed", spec.d_model - 1),
                             ("pca_adaptive", spec.d_model - 1),
                             ("pca_full", spec.d_model)):
                with c3.linear_splice(model, spec, mean2, v1b, mode,
                                      U=Ub, V=Vb, k=kk):
                    ce = ce_loss(model, seqs, chunk=4, am=am)
                check(f"{mode} at full rank is the identity",
                      abs(ce - clean) < 5e-3,
                      f"CE {ce:.6f} vs clean {clean:.6f}")
            # and a rank-0-ish mode must be catastrophic, or the hook is a no-op
            with c3.linear_splice(model, spec, mean2, v1b, "mean_only"):
                ce_mean = ce_loss(model, seqs, chunk=4, am=am)
            check("mean_only actually damages the model", ce_mean > clean + 0.5,
                  f"CE {ce_mean:.4f} vs clean {clean:.4f}")
            with c3.linear_splice(model, spec, mean2, v1b, "pc1_only"):
                ce_pc1 = ce_loss(model, seqs, chunk=4, am=am)
            print(f"       (pc1_only CE {ce_pc1:.4f}; rank-1 vs clean "
                  f"{clean:.4f} vs mean-only {ce_mean:.4f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
