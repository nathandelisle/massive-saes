"""E4 real-data objective grid for one model site, end to end on one box.
Harvest (v3 protocol: wikitext-103-v1 train, one document per sequence,
add_special_tokens=True, truncation 128, padding dropped) -> PCA (first 50k
centered rows, float64) -> held-out calibration/test exports from the
validation/test splits with the identical tokenizer/model/block -> paired
init/data-order grid -> evaluation -> seed comparison -> replication summary.
Idempotent: every stage is skipped when its manifest says complete.
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; PKG = HERE.parent; EXP = PKG/'experiments'; V3 = PKG/'pc1_v3'


def run(argv, log):
    with open(log, 'a') as f:
        r = subprocess.run([sys.executable]+[str(x) for x in argv], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode: raise SystemExit(f'child failed rc={r.returncode}; see {log}')


def completed(d):
    m = Path(d)/'manifest.json'
    return m.exists() and json.loads(m.read_text()).get('status') == 'complete'


def build_splits(model_key, out_dir, counts, log):
    """make_wikitext_splits.py: validation->calibration+selection, test->test.

    Token npz files with per-document sha256, disjoint hash sets, tokenized
    exactly like the legacy 5M-token training stream. Falls back to smaller
    document counts if a split runs out of usable rows.
    """
    out_dir = Path(out_dir)
    for n_cal, n_sel, n_test in counts:
        if (out_dir/'splits.json').exists():
            return json.loads((out_dir/'splits.json').read_text())
        try:
            run([PKG/'make_wikitext_splits.py', '--model', model_key, '--n-cal', n_cal,
                 '--n-sel', n_sel, '--n-test', n_test, '--out', out_dir], log)
        except SystemExit as exc:
            print(f'[e4-real] split counts {n_cal}/{n_sel}/{n_test} failed ({exc}); trying smaller', flush=True)
            if out_dir.exists(): shutil.rmtree(out_dir)
            continue
        return json.loads((out_dir/'splits.json').read_text())
    raise SystemExit('no usable held-out document counts')


def acts_from_tokens(spec, npz_path, out_path, device='cuda', batch_seqs=32):
    """Forward the exact held-out token arrays through the hooked block.

    Identical hook, dtype and padding-drop as v3.common.data.harvest_activations,
    so held-out rows are drawn from the same distribution as the training store.
    """
    import torch
    from v3.common.data import load_model_and_tokenizer, get_blocks, _CaptureHook
    from v3.common.registry import block_index
    with np.load(npz_path) as z:
        ids_all = z['input_ids']; am_all = z['attention_mask']
    model, tok = load_model_and_tokenizer(spec, device=device)
    li = int(block_index(spec))
    hook = _CaptureHook(); h = get_blocks(model, spec.block_path)[li].register_forward_hook(hook)
    chunks = []
    with torch.no_grad():
        for s in range(0, ids_all.shape[0], batch_seqs):
            ids = torch.as_tensor(ids_all[s:s+batch_seqs]).long().to(device)
            am = torch.as_tensor(am_all[s:s+batch_seqs]).long().to(device)
            model(ids, attention_mask=am)
            acts = hook.buf.float(); keep = am.bool().to(acts.device)
            chunks.append(acts[keep].cpu().numpy().astype(np.float16))
    h.remove(); del model; torch.cuda.empty_cache()
    X = np.concatenate(chunks); np.save(out_path, X)
    return dict(source=str(npz_path), n_tokens=int(X.shape[0]), n_sequences=int(ids_all.shape[0]),
                dtype='float16', block_index=li)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--steps', type=int, default=20000); p.add_argument('--tag', default='full')
    p.add_argument('--root', default=os.environ.get('PC1_V3_ROOT', 'workdir'))
    p.add_argument('--seeds', default='42,137,0,1,2'); p.add_argument('--betas', default='0,0.03,0.1,0.3'); p.add_argument('--k', type=int, default=64)
    p.add_argument('--batch', type=int, default=4096); p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--n-tokens', type=int, default=5_000_000, help='training harvest size (v3 protocol: 5M)')
    p.add_argument('--max-heldout-tokens', type=int, default=400_000, help='cap per held-out split (validation/test are ~200k each)')
    p.add_argument('--d-sae', type=int, default=None, help='override registry d_sae (default 4*d_model)')
    a = p.parse_args()
    os.environ.setdefault('PC1_V3_ROOT', a.root); sys.path.insert(0, str(V3)); sys.path.insert(0, str(PKG))
    from v3.common.registry import MODELS
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    spec = MODELS[a.model]; root = Path(a.root).resolve()
    adir = Path(ensure_acts(spec, corpus='wikitext', n_tokens=a.n_tokens, device='cuda'))
    evals, evecs, mean = estimate_pcs(str(adir))
    pc1 = adir/'pc1.npy'
    if not pc1.exists(): np.save(pc1, np.ascontiguousarray(evecs[:, 0]).astype(np.float64))
    print(f'[e4-real] {a.model} acts={adir} PC1 var frac={float(evals[0]/evals.sum()):.4f}', flush=True)
    hold = root/'heldout'/f'{a.model}_wikitext'; hold.mkdir(parents=True, exist_ok=True); prov = hold/'provenance.json'
    if not prov.exists():
        from e123_common import HELDOUT_STATEMENT
        splits = build_splits(a.model, hold/'tokens', [(1024, 512, 1024), (512, 256, 512), (256, 256, 512)], hold/'splits.log')
        info = dict(calibration=acts_from_tokens(spec, hold/'tokens'/'calibration.npz', hold/'calibration.npy'),
                    test=acts_from_tokens(spec, hold/'tokens'/'test.npz', hold/'test.npy'),
                    selection=acts_from_tokens(spec, hold/'tokens'/'selection.npz', hold/'selection.npy'),
                    splits=splits, model=spec.hf_id, target_layer=spec.target_layer, block_path=spec.block_path,
                    tokenization=HELDOUT_STATEMENT)
        prov.write_text(json.dumps(info, indent=2)); print('[e4-real] heldout', json.dumps(
            {k: info[k] for k in ('calibration', 'test', 'selection')}), flush=True)
    seeds = [int(s) for s in a.seeds.split(',')]; betas = [float(b) for b in a.betas.split(',')]
    out = root/'results'/'e4_real'/a.model/a.tag; out.mkdir(parents=True, exist_ok=True)
    cfg = dict(v3_path=str(V3), activations=str(adir/'acts.npy'), training_split='train', mean_path=str(adir/'mean.npy'), pc1_path=str(pc1),
               d_sae=a.d_sae or spec.d_sae, conditions=['standard', 'bypass'], k_values=[a.k], seeds=seeds, betas=betas, steps=a.steps,
               batch_size=a.batch, lr=3e-4, data_order_seed=20260906, checkpoint_every=a.checkpoint_every, device='cuda', deterministic=True)
    config = out/'training_config.json'
    if config.exists() and json.loads(config.read_text()) != cfg: raise SystemExit('config changed; use a new --tag')
    config.write_text(json.dumps(cfg, indent=2))
    grid, evaluation, comparison, summary = out/'training', out/'evaluation', out/'seed_comparison', out/'summary'
    stmt = ('WikiText-103-v1 validation (calibration) and test (test) splits; training store is the first 5M tokens of the train split; '
            'no document overlap by construction of the official splits; identical tokenization/hook')
    if not completed(grid):
        run([EXP/'train_objective_grid.py', '--config', config, '--out', grid]+(['--resume'] if (grid/'manifest.json').exists() else []), out/'training.log')
    if not completed(evaluation):
        run([EXP/'evaluate_objective_grid.py', '--grid', grid, '--calibration', hold/'calibration.npy', '--test', hold/'test.npy', '--out', evaluation,
             '--heldout-statement', stmt]+(['--resume'] if (evaluation/'manifest.json').exists() else []), out/'evaluation.log')
    if not completed(comparison):
        run([EXP/'compare_grid_seeds.py', '--grid', grid, '--evaluation', evaluation, '--out', comparison]+(['--resume'] if (comparison/'manifest.json').exists() else []), out/'comparison.log')
    (out/'registry.json').write_text(json.dumps(dict(expected_seeds=seeds, cases=[dict(site=a.model, summary=str(comparison/'summary.json'))]), indent=2))
    if summary.exists() and not completed(summary): shutil.rmtree(summary)
    if not completed(summary): run([EXP/'summarize_replications.py', '--registry', out/'registry.json', '--out', summary], out/'summary.log')
    (out/'DONE').write_text(json.dumps(dict(model=a.model, tag=a.tag, steps=a.steps)))
    print(f'[e4-real] DONE {a.model} {a.tag}', flush=True)


if __name__ == '__main__': main()
