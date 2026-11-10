"""Small, local-only forward passes for E19 and E20. No SAE training or cloud launches.

profile: Pythia-160M, all raw block outputs + block-0 input + final normalized output;
         trained weights or explicitly selected random initializations.
characterize: Pythia-1.4B at an EXPLICIT ZERO-BASED BLOCK INDEX, frozen geometry.
              Saves scalar observations consumed by measure_pc1.py, not a huge activation cache.

The collector uses the backbone GPTNeoXModel, avoiding the vocabulary projection/logits.
Use the existing RX torch/transformers environment, and test one batch there before a full run.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from science_common import Moments, describe_geometry, unit, provenance, sha256, write_json


def read_tokens(path, max_tokens):
    z = np.load(path, allow_pickle=False)
    ids, mask, doc = z['input_ids'], z['attention_mask'], z['doc_id']
    if ids.ndim != 2 or mask.shape != ids.shape or doc.shape != (len(ids),):
        raise ValueError('Token cache needs input_ids/mask [sequences, positions], doc_id [sequences].')
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('input_ids must be nonnegative integers')
    if not np.issubdtype(doc.dtype, np.integer) or np.any(doc < 0):
        raise ValueError('doc_id must contain stable nonnegative ARTICLE ids')
    if not np.isin(mask, [0, 1]).all() or np.any(np.diff(mask.astype(int), axis=1) > 0):
        raise ValueError('Expected a binary RIGHT-padding mask.')
    if not np.all(mask.sum(1) > 0):
        raise ValueError('Empty sequences are not allowed.')
    take = min(int(mask.sum()), max_tokens)
    if take < 3:
        raise ValueError('Too few valid tokens.')
    # Preserve original document/paragraph order. No length sorting / token shuffling.
    selected = mask.astype(bool).copy()
    cumulative = np.cumsum(selected.ravel())
    selected.ravel()[cumulative > take] = False
    last_seq = np.flatnonzero(selected.any(1))[-1] + 1
    return ids[:last_seq], mask[:last_seq], doc[:last_seq], selected[:last_seq]


def collect(backbone, ids, mask, docs, selected, batch_seqs, device,
            mode, block=None, geometry=None):
    import torch
    nblocks = len(backbone.layers)
    state, hooks, acc, obs = {}, [], {}, []

    class FinishedSite(Exception):
        pass

    if mode == 'profile':
        keys = ['block_0_input'] + ['block_%d_output' % b for b in range(nblocks)] + ['final_norm_output']
        acc = {k: Moments() for k in keys}

        def consume(key, h):
            rows = h.detach()[torch.as_tensor(state['selected'], device=h.device)]
            acc[key].update(rows.float().cpu().numpy())

        def pre(_m, args, kwargs):
            consume('block_0_input', args[0] if args else kwargs['hidden_states'])

        hooks.append(backbone.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
        for b, module in enumerate(backbone.layers):
            def post(_m, _i, o, key='block_%d_output' % b):
                consume(key, o[0] if isinstance(o, tuple) else o)
            hooks.append(module.register_forward_hook(post))
        hooks.append(backbone.final_layer_norm.register_forward_hook(
            lambda _m, _i, o: consume('final_norm_output', o)))
    else:
        if block is None or not 0 <= block < nblocks:
            raise ValueError('Explicit zero-based --block is outside this model.')
        v1 = unit(geometry['v1'])
        mu = np.asarray(geometry['mean'], dtype=np.float64)
        if mu.shape != v1.shape or len(mu) != backbone.config.hidden_size:
            raise ValueError('Geometry does not match model width.')

        def at_site(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            # FP64 on CPU avoids cancellation in centered PC1 coefficients.
            x = h.detach()[torch.as_tensor(state['selected'], device=h.device)].float().cpu().numpy().astype(np.float64)
            obs.append(dict(score=(x-mu) @ v1, norm=np.linalg.norm(x, axis=1),
                            token_id=state['ids'][state['selected']],
                            doc_id=np.broadcast_to(state['docs'][:, None], state['ids'].shape)[state['selected']],
                            position=np.broadcast_to(np.arange(state['ids'].shape[1]), state['ids'].shape)[state['selected']]))
            raise FinishedSite()
        hooks.append(backbone.layers[block].register_forward_hook(at_site))

    start = time.time()
    try:
        with torch.inference_mode():
            for i in range(0, len(ids), batch_seqs):
                # Trim right padding; causal contexts and row order remain unchanged.
                length = int(mask[i:i+batch_seqs].sum(1).max())
                state.update(ids=ids[i:i+batch_seqs, :length], am=mask[i:i+batch_seqs, :length],
                             selected=selected[i:i+batch_seqs, :length], docs=docs[i:i+batch_seqs])
                try:
                    backbone(input_ids=torch.as_tensor(state['ids'], dtype=torch.long, device=device),
                             attention_mask=torch.as_tensor(state['am'], dtype=torch.long, device=device),
                             use_cache=False, output_hidden_states=False, return_dict=True)
                except FinishedSite:
                    if mode != 'characterize':
                        raise
                if (i // batch_seqs) % 100 == 0:
                    print('%s: %d/%d sequences, %.1fs' % (mode, min(i+batch_seqs, len(ids)), len(ids), time.time()-start), flush=True)
    finally:
        for h in hooks:
            h.remove()
    if mode == 'profile':
        expected = int(selected.sum())
        if any(a.n != expected for a in acc.values()):
            raise RuntimeError('A hook did not observe exactly the selected tokens.')
        result, arrays = {}, {}
        for key, a in acc.items():
            row, ev, q = describe_geometry(a)
            result[key] = row
            arrays[key+'__mean'] = a.mean
            arrays[key+'__eigenvalues'] = ev
            arrays[key+'__v1'] = q[:, 0]
            arrays[key+'__v2'] = q[:, 1]
        return result, arrays
    arrays = {k: np.concatenate([r[k] for r in obs]) for k in obs[0]}
    if len(arrays['score']) != int(selected.sum()):
        raise RuntimeError('Characterization hook missed tokens.')
    return dict(n_tokens=len(arrays['score']), block_index=block), arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['profile', 'characterize'])
    ap.add_argument('--weights', required=True, help='Local HF snapshot directory (no downloads)')
    ap.add_argument('--tokens', required=True, help='.npz input_ids, attention_mask, doc_id')
    ap.add_argument('--input-meta', required=True, help='JSON: model_id, model_revision, tokenizer_id, tokenizer_revision, corpus, partition, bos_rule')
    ap.add_argument('--out', required=True, help='New directory; existing measurement files are never overwritten')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    ap.add_argument('--batch-seqs', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=50000,
                    help='New measurement defaults to 50k. Use 500000 ONLY to retain the old 500k protocol.')
    ap.add_argument('--random-seeds', type=int, nargs='+', help='Profile random initializations instead of trained model. Suggested: 0 1 2')
    ap.add_argument('--block', type=int, help='ZERO-BASED block index, mandatory for characterization')
    ap.add_argument('--geometry', help='.npz mean/v1, frozen canonical training assets')
    a = ap.parse_args()
    if a.batch_seqs < 1 or a.max_tokens < 3:
        ap.error('Invalid batch size / token count')
    if not Path(a.weights).is_dir():
        ap.error('--weights must be a local HF snapshot directory')
    meta = json.loads(Path(a.input_meta).read_text())
    required = ['model_id','model_revision','tokenizer_id','tokenizer_revision','corpus','partition','bos_rule']
    if any(not isinstance(meta.get(k), str) or not meta[k].strip() for k in required):
        ap.error('Input metadata must contain nonempty strings for: '+', '.join(required))
    if a.mode == 'profile' and meta['model_id'] != 'EleutherAI/pythia-160m':
        ap.error('This targeted profile experiment is scoped to EleutherAI/pythia-160m.')
    if a.mode == 'characterize':
        if a.random_seeds or a.block is None or not a.geometry:
            ap.error('Characterization needs --block and --geometry, with trained weights.')
        if meta['model_id'] != 'EleutherAI/pythia-1.4b':
            ap.error('This targeted characterization is scoped to EleutherAI/pythia-1.4b.')
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXModel
    import transformers
    cfg = GPTNeoXConfig.from_pretrained(a.weights, local_files_only=True)
    if cfg.model_type != 'gpt_neox':
        raise ValueError('Expected a GPT-NeoX config.')
    expected = (12,768) if a.mode == 'profile' else (24,2048)
    if (cfg.num_hidden_layers, cfg.hidden_size) != expected:
        raise ValueError('Model config does not match the declared Pythia model.')
    ids, mask, docs, selected = read_tokens(a.tokens, a.max_tokens)
    if ids.max() >= cfg.vocab_size:
        raise ValueError('Token vocabulary does not match model.')
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    variants = [('random_seed_%d' % s, s) for s in a.random_seeds] if a.random_seeds else [('trained', None)]
    if len(set(k for k,s in variants)) != len(variants):
        ap.error('Duplicate random seeds')
    for label, _ in variants:
        if (root/(label+'.json')).exists() or (root/(label+'.npz')).exists():
            raise FileExistsError('Use a fresh output directory or archive the partial run: '+str(root))
    src = [a.tokens, a.input_meta, Path(a.weights)/'config.json'] + ([a.geometry] if a.geometry else [])
    prov = provenance(src)
    prov.update(torch=torch.__version__, transformers=transformers.__version__)
    geom = dict(np.load(a.geometry, allow_pickle=False)) if a.geometry else None
    if geom is not None:
        if 'block_index' not in geom or 'model_id' not in geom:
            raise ValueError('Geometry .npz must include block_index and model_id to prevent layer/cohort mistakes.')
        if int(geom['block_index']) != a.block or str(geom['model_id'].item()) != meta['model_id']:
            raise ValueError('Geometry model / zero-based block index does not match the measurement.')
    for label, seed in variants:
        t0 = time.time()
        if seed is None:
            backbone = GPTNeoXModel.from_pretrained(a.weights, local_files_only=True,
                torch_dtype=getattr(torch, a.dtype), attn_implementation='eager')
        else:
            torch.manual_seed(seed)
            # Same architecture, all weights randomly initialized. No copying trained embeddings.
            cfg._attn_implementation = 'eager'
            backbone = GPTNeoXModel(cfg).to(dtype=getattr(torch, a.dtype))
        backbone = backbone.to(a.device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        result, arrays = collect(backbone, ids, mask, docs, selected, a.batch_seqs, a.device,
                                 a.mode, a.block, geom)
        if geom:
            arrays['geometry_sha256'] = np.array(sha256(a.geometry))
        np.savez(root/(label+'.npz'), **arrays)
        report = dict(experiment='E19_profile' if a.mode == 'profile' else 'E20_observations',
                      input_metadata=meta, mode=a.mode, initialization=label, random_seed=seed,
                      dtype=a.dtype, attention='eager', device=a.device, batch_seqs=a.batch_seqs,
                      max_tokens_requested=a.max_tokens, actual_tokens=int(selected.sum()),
                      n_documents=int(len(np.unique(docs[selected.any(1)]))),
                      sampling='first valid tokens of supplied ordered cache; right padding excluded',
                      pca='exact centered covariance on ALL selected rows; FP64 Chan accumulation' if a.mode=='profile' else 'frozen supplied training mean and PC1',
                      layer_convention='block_b_output = raw output of zero-based block b; final_norm_output is a DISTINCT site',
                      result=result, provenance=prov, seconds=time.time()-t0,
                      artifact_sha256=sha256(root/(label+'.npz')))
        write_json(root/(label+'.json'), report)
        print('Wrote', root/(label+'.json'), flush=True)
        del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
