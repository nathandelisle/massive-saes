"""Build the released-SAE tables from the saved v1 and v2 measurements."""
import argparse
import csv
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results/additional/released_saes"
V1_DEFAULT = RESULTS / "v1_inputs"
HERE = pathlib.Path(__file__).resolve().parent

# Reviewed convention record. `documented` = the release states the evaluation space/token rule we apply.
CONVENTIONS = {
    'Gemma Scope': dict(
        centering='none (apply_b_dec_to_input=false)',
        scaling='fixed scalar folded into the released weights',
        token_exclusions='BOS/EOS/padding excluded',
        norm_filter='none documented',
        documented='yes',
        source='Gemma Scope §3.1, arxiv 2408.05147v2#S3.SS1',
        space='raw'),
    'Gemma Scope 2': dict(
        centering='none (apply_b_dec_to_input=false)',
        scaling='fixed scaling folded into weights (tech report §3.1)',
        token_exclusions='pad/start/end masked for evaluation (tech report §4.1)',
        norm_filter='none documented',
        documented='yes',
        source='Gemma Scope 2 technical report §§3.1, 4.1 (DeepMind PDF)',
        space='raw'),
    'Llama Scope': dict(
        centering='apply_decoder_bias_to_pre_encoder=false (pinned L6 config)',
        scaling='dataset-level constant; weights rescaled for inference (§§3.3.3-3.3.4)',
        token_exclusions='start/end tokens excluded (§3.1)',
        norm_filter='none documented',
        documented='yes',
        source='Llama Scope, arxiv 2410.20526v1 §§3.1, 3.3.3-3.3.4',
        space='raw'),
    'OpenAI GPT-2': dict(
        centering='apply_b_dec_to_input=true',
        scaling='per-vector mean removal + normalization, for SAE input AND error evaluation',
        token_exclusions='none documented; all tokens',
        norm_filter='none documented',
        documented='yes',
        source='OpenAI SAE scaling, arxiv 2406.04093v1 §2.1',
        space='sae_input'),
    'EleutherAI Pythia': dict(
        centering='b_dec subtracted before the encoder (native sparsify pre_acts) -- CORRECTED in v2',
        scaling='none',
        token_exclusions='none documented; no BOS prepended, so all == no_special',
        norm_filter='none documented',
        documented='no (cfg carries architecture/k only; no published evaluation recipe)',
        source='EleutherAI/sae-pythia-160m-32k cfg.json + sparsify@c259f271 sae.py',
        space='raw'),
    'Qwen Scope': dict(
        centering='none (release app.py: pre = hidden @ W_enc + b_enc)',
        scaling='none',
        token_exclusions='none documented beyond the norm filter',
        norm_filter='Large-norm filtering described in §2.2; multiplier and batch definition unspecified.',
        documented='Release cutoff unspecified; evaluated at 5x, 10x, and 20x the calibration median norm.',
        source='Qwen-Scope §2.2, arxiv 2605.11887v1#S2.SS2; dictionary_learning@60ec6bf5 buffer.py',
        space='raw'),
    'Qwen2.5 filtered BatchTopK': dict(
        centering='apply_b_dec_to_input=true',
        scaling='none',
        token_exclusions='release filters the first eight positions',
        norm_filter='release also applies a norm cutoff; NOT applied here',
        documented='no -- only the position half of the documented filter is reproduced',
        source='andyrdt/saes-qwen2.5-7b-instruct release notes',
        space='raw'),
}

# Which families may appear in a headline count of documented-convention checkpoints.
HEADLINE_OK = {'Gemma Scope', 'Gemma Scope 2', 'Llama Scope', 'OpenAI GPT-2'}


def read_csv(path):
    with pathlib.Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        print('skip (no rows):', path)
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            for key in ('source', 'source_file'):
                if isinstance(r.get(key), str):
                    r[key] = r[key].replace(str(RESULTS.parent.parent.parent) + '/', '')
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    print('wrote %s (%d rows)' % (path, len(rows)))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v1', default=str(V1_DEFAULT))
    p.add_argument('--v2', default=str(RESULTS / 'round2_results'))
    p.add_argument('--out', default=str(pathlib.Path.cwd() / 'generated/released_saes'))
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    import sys
    sys.path.insert(0, str(HERE))
    from metrics import article_bootstrap
    v1, v2, out = pathlib.Path(a.v1), pathlib.Path(a.v2), pathlib.Path(a.out)
    if not (v2 / "pythia_release_style" / "pythia_release_style.json").is_file() or not list(v2.rglob("qwen_filter/*.json")):
        raise FileNotFoundError("The v2 raw results are missing. Supply them with --v2; saved tables are in results/additional/released_saes/sec7_tables_v2/.")
    missing_statistics = []
    for path in v2.glob("pythia160m/ctx*/pythia160m-*.json"):
        data = json.loads(path.read_text())
        if data.get("status") == "ok":
            missing_statistics.extend(str(path.parent / row["statistics_file"])
                                      for row in data["rows"]
                                      if not (path.parent / row["statistics_file"]).is_file())
    if missing_statistics:
        raise FileNotFoundError(
            f"Missing {len(missing_statistics)} per-checkpoint statistics files. "
            "Run experiments/released_saes/download_statistics.py first.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((v1 / 'manifest.json').read_text())
    jobs = {j['job_id']: j for j in manifest['jobs']}

    figure_v1 = read_csv(v1 / 'tables' / 'figure_released_fvu.csv')
    views_v1 = read_csv(v1 / 'tables' / 'all_measured_views.csv')

    # ---------------------------------------------------------------- native forward checks
    native = []
    for f in sorted(v2.rglob('native/native_*.json')):
        d = json.loads(f.read_text())
        native.append(dict(group=d['group'], family=d['family'], job_id=d['job_id'], repo_id=d['repo_id'],
                           repo_revision=d['repo_revision'], sae_id=d['sae_id'],
                           conversion_func=d.get('conversion_func'), architecture=d.get('architecture'),
                           normalization=d.get('normalization'),
                           apply_b_dec_to_input_used=d.get('apply_b_dec_to_input_used'),
                           encoder_centering_corrected=d.get('encoder_centering_corrected'),
                           wdec_orientation_corrected=d.get('wdec_orientation_corrected'),
                           verdict=d['verdict'],
                           max_recon_rel_diff_same_input=d.get('max_reconstruction_rel_diff'),
                           max_recon_rel_diff_own_normalization=d.get('max_reconstruction_rel_diff_own_normalization'),
                           min_active_jaccard=d.get('min_active_jaccard'),
                           high_norm_recon_rel=d['checks'].get('high_norm', {}).get('reconstruction_rel_diff_same_input'),
                           b_dec_probe_jaccard=d['checks'].get('b_dec', {}).get('active_jaccard'),
                           zero_probe_jaccard=d['checks'].get('zero', {}).get('active_jaccard'),
                           gpt2_normalization=d.get('gpt2_normalization'),
                           source_file=str(f)))
    write_csv(out / 'native_forward_checks.csv', native)
    nat_by_family = {n['family']: n for n in native}

    # ---------------------------------------------------------------- corrected Pythia rows
    pyth_rows = []
    for f in sorted(v2.rglob('pythia160m/ctx*/pythia160m-*.json')):
        if f.name.endswith('.error.json'):
            continue
        d = json.loads(f.read_text())
        if d.get('status') != 'ok':
            continue
        for row in d['rows']:
            g = np.load(f.parent / row['statistics_file'])
            ci = article_bootstrap(g['group_stats'], g['axis'], draws=a.bootstrap)
            flat = dict(job_id=d['job']['job_id'], family=d['job']['family'], model=d['job']['model'],
                        layer=d['job']['layer'], sae_id=d['job']['sae_id'],
                        repo_revision=d['job']['repo_revision'], model_revision=d['job']['model_revision'],
                        protocol=d['protocol'], context_length=d['context_length'], space=row['space'],
                        population=row['population'], n_documents=row['n_documents'],
                        cache_fingerprint=d['cache_metadata']['fingerprint'], code_sha256=d['code_sha256'],
                        source=str(f), n_features=d['n_features'],
                        n_alive_any_eval_token=d['n_alive_any_eval_token'],
                        encoder_centering_corrected=d['adapter_smoke'].get('encoder_centering_corrected'),
                        apply_b_dec_to_input_used=d['adapter_smoke'].get('apply_b_dec_to_input_used'),
                        **{k: row[k] for k in ('n_rows', 'fvu_total', 'fvu_pc1', 'fvu_content',
                                               'content_over_total', 'w1_eval', 'residual_energy_on_pc1', 'l0')})
            for k in ('fvu_total', 'fvu_content', 'content_over_total', 'w1_eval'):
                if k in ci:
                    flat[k + '_lo95'], flat[k + '_hi95'] = ci[k]
            pyth_rows.append(flat)
    write_csv(out / 'pythia_corrected_views.csv', pyth_rows)

    old_p = [r for r in views_v1 if r['family'] == 'EleutherAI Pythia']
    comp = []
    for new in pyth_rows:
        old = next((r for r in old_p if r['space'] == new['space'] and r['population'] == new['population']
                    and int(r['context_length']) == new['context_length']), None)
        comp.append(dict(job_id=new['job_id'], context_length=new['context_length'], space=new['space'],
                         population=new['population'],
                         v1_apply_b_dec_to_input=False, v2_apply_b_dec_to_input=True,
                         v1_fvu_total=num(old['fvu_total']) if old else None,
                         v2_fvu_total=new['fvu_total'],
                         v1_fvu_content=num(old['fvu_content']) if old else None,
                         v2_fvu_content=new['fvu_content'],
                         v1_content_over_total=num(old['content_over_total']) if old else None,
                         v2_content_over_total=new['content_over_total'],
                         v1_w1_eval=num(old['w1_eval']) if old else None, v2_w1_eval=new['w1_eval'],
                         v1_l0=num(old['l0']) if old else None, v2_l0=new['l0']))
    write_csv(out / 'pythia_v1_vs_v2.csv', comp)

    # Pythia is one checkpoint carrying a family median, so its dependence on the evaluation token
    # population is reported explicitly rather than buried: chunked ctx128, chunked ctx512, and the
    # release-style EOS-packed ctx2049 view.
    ctx_rows = []
    for r in pyth_rows:
        ctx_rows.append(dict(view='independent chunks, ctx %d' % r['context_length'],
                             population=r['population'], context_length=r['context_length'],
                             packing='documents chunked independently, no BOS',
                             n_rows=r['n_rows'], n_documents=r['n_documents'], fvu_total=r['fvu_total'],
                             fvu_content=r['fvu_content'], content_over_total=r['content_over_total'],
                             w1_eval=r['w1_eval'], l0=r['l0'],
                             content_over_total_lo95=r.get('content_over_total_lo95'),
                             content_over_total_hi95=r.get('content_over_total_hi95'),
                             source=r['source']))
    prs = v2 / 'pythia_release_style' / 'pythia_release_style.json'
    if prs.exists():
        d = json.loads(prs.read_text())
        for r in d['rows']:
            ctx_rows.append(dict(view='release-style EOS packing, ctx %d' % d['context_length'],
                                 population=r['population'], context_length=d['context_length'],
                                 packing=d['packing'], n_rows=r['n_rows'], n_documents=r['n_documents'],
                                 fvu_total=r['fvu_total'], fvu_content=r['fvu_content'],
                                 content_over_total=r['content_over_total'], w1_eval=r['w1_eval'], l0=r['l0'],
                                 content_over_total_lo95=r.get('content_over_total_lo95'),
                                 content_over_total_hi95=r.get('content_over_total_hi95'),
                                 source=str(prs), interpretation=d['interpretation']))
    ctx_rows.sort(key=lambda r: (r['context_length'], r['population']))
    write_csv(out / 'pythia_context_sensitivity.csv', ctx_rows)

    # ---------------------------------------------------------------- Qwen norm-filter sensitivity
    qrows, qattr = [], []
    for f in sorted(v2.rglob('qwen_filter/*.json')):
        d = json.loads(f.read_text())
        if d.get('status') != 'ok':
            continue
        job = d['job']
        for row in d['rows']:
            qrows.append(dict(job_id=job['job_id'], model=job['model'], layer=job['layer'],
                              sae_id=job['sae_id'], catalog_l0=job.get('catalog_l0'),
                              context_length=d['context_length'], population=row['population'],
                              norm_threshold_multiple=row.get('norm_threshold_multiple'),
                              norm_threshold_absolute=row.get('norm_threshold_absolute'),
                              calibration_median_raw_l2_norm=d['calibration_median_raw_l2_norm'],
                              **{k: row[k] for k in ('n_rows', 'n_documents', 'calibration_rows',
                                                     'calibration_fraction', 'evaluation_rows',
                                                     'evaluation_fraction', 'evaluation_documents',
                                                     'fvu_total', 'fvu_content', 'content_over_total',
                                                     'w1_eval', 'residual_energy_on_pc1', 'l0')},
                              **{k: row.get(k) for k in ('fvu_total_lo95', 'fvu_total_hi95',
                                                         'fvu_content_lo95', 'fvu_content_hi95',
                                                         'content_over_total_lo95', 'content_over_total_hi95',
                                                         'w1_eval_lo95', 'w1_eval_hi95')},
                              threshold_rule=d['threshold_rule'],
                              exact_release_filter=d['exact_release_filter'], source=str(f)))
        for cls, v in d['attribution_common_axis'].items():
            qattr.append(dict(job_id=job['job_id'], layer=job['layer'], sae_id=job['sae_id'],
                              context_length=d['context_length'], row_class=cls,
                              axis_population=d['attribution_axis']['population'], **v))
    write_csv(out / 'qwen_norm_filter_sensitivity.csv', qrows)
    write_csv(out / 'qwen_error_variance_attribution.csv', qattr)

    # ---------------------------------------------------------------- corrected figure + family tables
    fig = []
    for r in figure_v1:
        fam = r['family']
        if fam == 'EleutherAI Pythia':
            continue                                     # replaced below by the corrected computation
        rec = dict(r)
        rec['convention_status'] = 'documented' if fam in HEADLINE_OK else CONVENTIONS[fam]['documented']
        rec['source_version'] = 'v1 (unchanged; conventions upheld on review)'
        rec['native_forward_check'] = nat_by_family.get(fam, {}).get('verdict', 'not run')
        if fam == 'Qwen Scope':
            rec['convention_status'] = 'Unfiltered activations; release norm cutoff unspecified.'
            rec['norm_filter_applied'] = 'none'
        fig.append(rec)
    for new in pyth_rows:
        job = jobs[new['job_id']]
        if (new['space'] == job['primary_space'] and new['population'] == job['primary_population']
                and new['context_length'] == manifest['context_length']):
            fig.append(dict(new, convention_status=CONVENTIONS['EleutherAI Pythia']['documented'],
                            source_version='v2 (encoder centering corrected)',
                            native_forward_check=nat_by_family.get('EleutherAI Pythia', {}).get('verdict', 'not run'),
                            convention_source=job.get('convention_source', ''),
                            evaluation_interpretation=job.get('evaluation_interpretation', '')))
    for q in qrows:
        if q['population'] == 'norm10' and q['context_length'] == manifest['context_length']:
            fig.append(dict(job_id=q['job_id'], family='Qwen Scope', model=q['model'], layer=q['layer'],
                            sae_id=q['sae_id'], context_length=q['context_length'], space='raw',
                            population=q['population'], n_rows=q['n_rows'], n_documents=q['n_documents'],
                            fvu_total=q['fvu_total'], fvu_content=q['fvu_content'],
                            content_over_total=q['content_over_total'], w1_eval=q['w1_eval'], l0=q['l0'],
                            fvu_total_lo95=q.get('fvu_total_lo95'), fvu_total_hi95=q.get('fvu_total_hi95'),
                            content_over_total_lo95=q.get('content_over_total_lo95'),
                            content_over_total_hi95=q.get('content_over_total_hi95'),
                            convention_status='Fixed calibration threshold; release norm cutoff unspecified.',
                            norm_filter_applied='10x calibration median raw L2 norm (absolute, frozen)',
                            source_version='v2 (declared norm-filter sensitivity)',
                            native_forward_check=nat_by_family.get('Qwen Scope', {}).get('verdict', 'not run'),
                            source=q['source']))
    write_csv(out / 'figure_released_fvu_v2.csv', fig)

    fam_rows = []
    for fam in sorted({r['family'] for r in fig}):
        rs = [r for r in fig if r['family'] == fam]
        for status_label, subset in (('all rows shown', rs),
                                     ('unfiltered view', [r for r in rs if r.get('norm_filter_applied') in (None, '', 'none')]),
                                     ('standardized 10x filter', [r for r in rs if r.get('norm_filter_applied', '').startswith('10x')])):
            if not subset:
                continue
            if fam == 'Qwen Scope' and status_label == 'all rows shown':
                continue          # mixing the unfiltered and filtered views into one median is meaningless
            if fam != 'Qwen Scope' and status_label != 'all rows shown':
                continue
            vals = np.array([num(r['content_over_total']) for r in subset
                             if num(r['content_over_total']) is not None], dtype=float)
            if not len(vals):
                continue
            fam_rows.append(dict(family=fam, view=status_label, checkpoints=len(subset),
                                 model_layer_sites=len({(r['model'], r['layer']) for r in subset}),
                                 median_ratio=float(np.median(vals)), max_ratio=float(vals.max()),
                                 min_ratio=float(vals.min()),
                                 **{'n_ratio_gt_%d' % t: int((vals > t).sum()) for t in (2, 5, 10)},
                                 convention_status=subset[0].get('convention_status'),
                                 eligible_for_documented_convention_headline=fam in HEADLINE_OK))
    write_csv(out / 'family_summary_v2.csv', fam_rows)

    conv = []
    for fam, c in CONVENTIONS.items():
        n = nat_by_family.get(fam, {})
        conv.append(dict(family=fam, source=c['source'], evaluation_space=c['space'],
                         centering=c['centering'], scaling=c['scaling'],
                         token_exclusions=c['token_exclusions'], norm_filter=c['norm_filter'],
                         documented_or_approximation=c['documented'],
                         native_forward_check=n.get('verdict', 'not run'),
                         native_check_job=n.get('job_id'),
                         native_max_recon_rel_diff=n.get('max_recon_rel_diff_same_input'),
                         native_min_active_jaccard=n.get('min_active_jaccard'),
                         eligible_for_documented_convention_headline=fam in HEADLINE_OK))
    write_csv(out / 'convention_table.csv', conv)

    doc_rows = [r for r in fig if r['family'] in HEADLINE_OK]
    doc_vals = np.array([num(r['content_over_total']) for r in doc_rows], dtype=float)
    headline = dict(
        documented_convention_checkpoints=len(doc_rows),
        documented_convention_n_ratio_gt_2=int((doc_vals > 2).sum()),
        documented_convention_median_ratio=float(np.median(doc_vals)),
        excluded_from_headline={
            'Qwen Scope': 'Fixed calibration threshold; release norm cutoff unspecified.',
            'EleutherAI Pythia': 'no published evaluation recipe; standardized protocol only',
            'Qwen2.5 filtered BatchTopK': 'only the position half of its documented filter is reproduced'},
        note='Counts use checkpoints with documented evaluation conventions.')
    (out / 'headline_counts.json').write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == '__main__':
    main()
