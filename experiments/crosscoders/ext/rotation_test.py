"""Construct alternative decoder dictionaries without changing a single code.

Run `python rotation_test.py --help`. See PROTOCOL.txt for the scientific scope.
All arrays use ROW decoder vectors. No model download or cloud operation occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_rows(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n <= 1e-14):
        raise ValueError("Zero decoder row; supply only nonzero dictionary rows")
    return a / n


def perpendicular(a, v):
    return a - (a @ v)[..., None] * v


def span(a, rtol=1e-9):
    """Orthonormal column basis for the row span; numerical rank is reported."""
    _, s, vt = np.linalg.svd(np.asarray(a, dtype=np.float64), full_matrices=False)
    r = 0 if len(s) == 0 or s[0] == 0 else int(np.sum(s > rtol * s[0]))
    return vt[:r].T, s


def copy_rotation(rows, u, vcopy, theta):
    """Rotate U into V, where rows initially have zero projection on V."""
    return rows + (rows @ u) @ ((np.cos(theta) - 1) * u + np.sin(theta) * vcopy).T


def within_rotation(rows, u, o):
    return rows + (rows @ u) @ (o.T - np.eye(u.shape[1])) @ u.T


def orthogonal_copy(u_all, pc, q, rng):
    # U_all spans every perpendicular cluster decoder, not only selected U.
    d = len(pc)
    if d - 1 - u_all.shape[1] < q:
        return None
    a = rng.normal(size=(d, q))
    for _ in range(2):
        a -= pc[:, None] * (pc @ a)[None, :]
        a -= u_all @ (u_all.T @ a)
    v, r = np.linalg.qr(a, mode="reduced")
    if np.min(np.abs(np.diag(r))) < 1e-10:
        raise ValueError("Numerically degenerate orthogonal-copy basis")
    return v


def max_cosine(query, target, absolute=True, block=512):
    q, t = normalize_rows(query), normalize_rows(target)
    best = np.full(len(q), -np.inf)
    for start in range(0, len(t), block):
        a = q @ t[start:start+block].T
        if absolute:
            a = np.abs(a)
        best = np.maximum(best, a.max(axis=1))
    return best


def validate_arrays(e, z, doc, m, d):
    if e.ndim != 2 or e.shape[1] != d or z.shape != (len(e), m):
        raise ValueError("Expected error[N,d], z_cluster[N,m]")
    if doc.shape != (len(e),) or len(e) == 0:
        raise ValueError("Expected nonempty article/document IDs")
    if not np.isfinite(e).all() or not np.isfinite(z).all() or np.min(z) < -1e-8:
        raise ValueError("Nonfinite error/code or negative code")


def load_split(root, split, m, d):
    p = Path(root) / split
    # Error is reconstruction MINUS target, with bias/mean/bypass already included.
    e = np.load(p / "error.npy", mmap_mode="r", allow_pickle=False)
    z = np.load(p / "z_cluster.npy", mmap_mode="r", allow_pickle=False)
    doc = np.load(p / "doc_id.npy", allow_pickle=False)
    validate_arrays(e, z, doc, m, d)
    return e, z, doc


def sufficient(e, z, pc, batch=2048):
    m, d = z.shape[1], e.shape[1]
    g, h = np.zeros((m, m)), np.zeros((m, d))
    total = content = 0.0
    for i in range(0, len(e), batch):
        ei, zi = np.asarray(e[i:i+batch], float), np.asarray(z[i:i+batch], float)
        g += zi.T @ zi
        h += zi.T @ ei
        total += np.square(ei).sum()
        content += np.square(perpendicular(ei, pc)).sum()
    return {"G": g/len(e), "H": h/len(e), "total": total/len(e),
            "content": content/len(e), "n": len(e)}


def loss_change(t, stats):
    return float(2 * np.sum(t * stats["H"]) + np.sum(t * (stats["G"] @ t)))


def row_geometry(old, new, ids, alive, weights, threshold):
    # Match altered original cluster against ALL atoms in the alternative dictionary.
    full = old.copy()
    full[ids] = new
    c = max_cosine(old[ids], full[alive], absolute=True)
    un = c < threshold
    weighted = None if weights.sum() <= 0 else float(weights[un].sum()/weights.sum())
    return {"fraction_unmatched_abs": float(un.mean()), "weighted_unmatched_abs": weighted,
            "mean_max_abs_cosine": float(c.mean())}, c


def candidates(rows, pc, fit_stats, rng_seeds=(11, 29, 47)):
    b = perpendicular(rows, pc)
    u_all, singular = span(b)
    rank = u_all.shape[1]
    yield "identity", rows.copy(), {"rank": rank, "family": "identity"}
    if rank == 0:
        return
    s = b @ u_all
    second = s.T @ fit_stats["G"] @ s  # uncentered second moment!
    ev, w = np.linalg.eigh((second + second.T)/2)
    weak = u_all @ w
    sizes = sorted(set([min(rank, 2), max(1, rank//4), max(1, rank//2), rank]))
    for q in sizes:
        u = weak[:, :q]
        for seed in rng_seeds:
            rng = np.random.default_rng(seed + 997*q)
            vcopy = orthogonal_copy(u_all, pc, q, rng)
            if vcopy is not None:
                for angle in (60, 90, 120):
                    name = f"copy_q{q}_seed{seed}_deg{angle}"
                    yield name, copy_rotation(rows, u, vcopy, np.deg2rad(angle)), {
                        "family": "copy", "rank": rank, "rotated_rank": q,
                        "degrees": angle, "seed": seed,
                        "output_energy_in_subspace": float(np.maximum(ev[:q], 0).sum())}
            if q >= 2:
                o, _ = np.linalg.qr(rng.normal(size=(q, q)))
                # Haar O(q), may include reflections: both preserve norms/PC1.
                yield f"within_q{q}_seed{seed}", within_rotation(rows, u, o), {
                    "family": "within", "rank": rank, "rotated_rank": q, "seed": seed}


def document_records(e, z, doc, rows, new, pc, batch=2048):
    unique, inverse = np.unique(doc, return_inverse=True)
    # n, baseline total/content, candidate total/content, disagreement, cluster Q energy
    sums = np.zeros((len(unique), 7), dtype=np.float64)
    t, b = new - rows, perpendicular(rows, pc)
    for i in range(0, len(e), batch):
        ei, zi = np.asarray(e[i:i+batch], float), np.asarray(z[i:i+batch], float)
        delta = zi @ t
        changed = ei + delta
        qout = zi @ b
        vals = np.column_stack([np.ones(len(ei)), np.square(ei).sum(1),
             np.square(perpendicular(ei, pc)).sum(1), np.square(changed).sum(1),
             np.square(perpendicular(changed, pc)).sum(1), np.square(delta).sum(1),
             np.square(qout).sum(1)])
        np.add.at(sums, inverse[i:i+len(ei)], vals)
    return unique, sums


def equivalence(sums, margin=0.01, draws=5000, seed=812, confidence=0.95):
    """Paired article bootstrap; all tokens in a drawn article travel together."""
    total = sums.sum(0)
    out = {"documents": len(sums), "tokens": int(total[0]), "margin": margin,
           "confidence": confidence, "draws": draws, "both_equivalent": False}
    if len(sums) < 20:
        out["status"] = "insufficient_documents_for_confirmatory_claim"
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty((draws, 2))
    for i in range(draws):
        s = sums[rng.integers(0, len(sums), len(sums))].sum(0)
        boots[i] = [(s[3]-s[1])/s[1] if s[1] > 1e-20 else np.nan,
                    (s[4]-s[2])/s[2] if s[2] > 1e-20 else np.nan]
    passed = []
    for j, (name, base, new) in enumerate((("total", 1, 3), ("content", 2, 4))):
        if total[base] <= 1e-20 or not np.isfinite(boots[:, j]).all():
            out[name] = {"status": "zero_baseline_error_relative_equivalence_undefined"}
            passed.append(False)
            continue
        alpha = (1-confidence)/2
        low, high = np.quantile(boots[:, j], [alpha, 1-alpha])
        ok = bool(low > -margin and high < margin)
        out[name] = {"relative_change": float((total[new]-total[base])/total[base]),
                     "ci": [float(low), float(high)], "equivalent": ok,
                     "noninferior": bool(high < margin)}
        passed.append(ok)
    c, eta, lq = total[6]/total[0], total[5]/total[0], total[2]/total[0]
    out.update(status="evaluated", both_equivalent=all(passed), cluster_perp_energy=float(c),
               output_disagreement=float(eta),
               absolute_loss_change=float((total[3]-total[1])/total[0]),
               bound_using_measured_disagreement=float(2*np.sqrt(lq*eta)+eta),
               uniform_bound=float(4*np.sqrt(lq*c)+4*c))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--margin", type=float, default=.01)
    p.add_argument("--unmatched", type=float, default=.50)
    p.add_argument("--cosine", type=float, default=.70)
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()
    if not (0 < args.margin < 1 and 0 < args.unmatched <= 1 and 0 < args.cosine < 1):
        p.error("Invalid preregistered thresholds")
    args.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.bundle / "metadata.json").read_text())
    if meta.get("error_convention") != "reconstruction_minus_target":
        raise ValueError("metadata must declare reconstruction_minus_target")
    if meta.get("encoder_depends_on_decoder") is not False:
        raise ValueError("Frozen-code construction requires an independent encoder")
    with np.load(args.bundle / "dictionary.npz", allow_pickle=False) as f:
        d = np.asarray(f["decoder"], float)
        pc = np.asarray(f["pc1"], float)
        ids = np.asarray(f["cluster_ids"], int)
        alive = np.asarray(f["alive_ids"], int)
    if d.ndim != 2 or pc.shape != (d.shape[1],):
        raise ValueError("Bad dictionary shape")
    if not np.isfinite(d).all() or not np.isfinite(pc).all():
        raise ValueError("Nonfinite dictionary")
    if not np.allclose(np.linalg.norm(d, axis=1), 1, atol=2e-4) or not np.isclose(pc@pc, 1, atol=1e-6):
        raise ValueError("This protocol expects unit decoder rows and unit PC1")
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or len(np.unique(alive)) != len(alive):
        raise ValueError("Empty/duplicate cluster or duplicate alive IDs")
    if np.any(ids < 0) or np.any(ids >= len(d)) or np.any(alive < 0) or np.any(alive >= len(d)):
        raise ValueError("Invalid feature index")
    if not set(ids).issubset(set(alive)):
        raise ValueError("All cluster IDs must be in the frozen alive set")
    loaded = {s: load_split(args.bundle, s, len(ids), d.shape[1])
              for s in ("fit", "validation", "test")}
    sets = {s: set(arr[2].tolist()) for s, arr in loaded.items()}
    if any(sets[a] & sets[b] for a,b in (("fit","validation"),("fit","test"),("validation","test"))):
        raise ValueError("Document leakage between splits")
    fit = sufficient(*loaded["fit"][:2], pc)
    val = sufficient(*loaded["validation"][:2], pc)
    eigen = np.linalg.eigvalsh((fit["G"]+fit["G"].T)/2)
    moment_diagnostics = {"eigenvalues":eigen.tolist(),
        "numerical_rank_relative_1e_9":int(np.sum(eigen>max(eigen[-1],0)*1e-9)),
        "note":"Positive definiteness precludes a distinct exact fixed-code function, not equal risk or changed-encoder alternatives."}
    (args.out / "code_second_moment.json").write_text(json.dumps(moment_diagnostics,indent=2))
    rows = d[ids]
    weights = np.diag(fit["G"]) * np.square(rows @ pc)
    records, possible = [], []
    for name, changed, detail in candidates(rows, pc, fit):
        if not np.allclose(np.linalg.norm(changed, axis=1), np.linalg.norm(rows, axis=1), atol=2e-7):
            raise AssertionError("Rotation broke decoder norms")
        if not np.allclose(changed @ pc, rows @ pc, atol=2e-7):
            raise AssertionError("Rotation changed PC1 contribution")
        geo, _ = row_geometry(d, changed, ids, alive, weights, args.cosine)
        delta = loss_change(changed-rows, val)
        rel = max(abs(delta)/val["total"], abs(delta)/val["content"]) if min(val["total"], val["content"]) > 1e-20 else float("inf")
        record = {"name": name, **detail, **geo, "validation_loss_change": delta,
                  "validation_worst_relative_change": rel if np.isfinite(rel) else None}
        records.append(record)
        if name != "identity" and geo["fraction_unmatched_abs"] >= args.unmatched and (geo["weighted_unmatched_abs"] or 0) >= args.unmatched:
            possible.append((rel, name, changed.copy(), record))
    (args.out / "validation_candidates.json").write_text(json.dumps(records, indent=2))
    if not possible:
        (args.out / "result.json").write_text(json.dumps({"status":"no_geometrically_distinct_candidate", "note":"No identifiability conclusion follows."}, indent=2))
        return
    _, name, changed, selection = min(possible, key=lambda t: (t[0],t[1]))
    # Write the frozen validation selection BEFORE evaluating test losses.
    selection.update(dictionary_sha256=digest(args.bundle/"dictionary.npz"),
                     margin=args.margin, cosine=args.cosine, min_unmatched=args.unmatched)
    (args.out / "frozen_selection.json").write_text(json.dumps(selection, indent=2))
    new_d = d.copy(); new_d[ids] = changed
    np.savez_compressed(args.out / "alternative_dictionary.npz", decoder=new_d,
                        cluster_ids=ids, pc1=pc, original_decoder=d)
    docs, sums = document_records(*loaded["test"], rows, changed, pc)
    np.savez_compressed(args.out / "test_document_sums.npz", doc_id=docs, sums=sums)
    report = equivalence(sums, args.margin, args.bootstrap)
    report.update(selected_candidate=name, geometry=selection, metadata=meta,
                  interpretation=(
                      "PRIMARY CRITERION MET: constructed alternative for this checkpoint; no global-optimum claim."
                      if (report.get("both_equivalent") and selection.get("fraction_unmatched_abs", 0) >= args.unmatched
                          and selection.get("weighted_unmatched_abs", 0) >= args.unmatched)
                      else "PRIMARY CRITERION NOT MET for the frozen candidate (see both_equivalent / geometry)."))
    (args.out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
