"""v2 task generator: deduplicated, genuinely disjoint splits, and a real selectivity control.

WHAT WAS WRONG IN v1 (`tasks.py`), all verified:
  * The SVA bank admitted only 4 x 8 x 6 x 2 x 2 = 768 distinct prompts, and 6,000 were sampled WITH
    replacement. The train split held 1,812 rows but 228 unique prompts, and the first 96 rows -- the
    slice the greedy mask search actually used -- contained 12 UNIQUE prompts. The 512-row test split
    held 68 unique prompts. "512 held-out examples" was therefore false.
  * `split_items` partitioned the JOINT key (template_family, entity_cluster), so the same template could
    appear in train and test with different nouns, and the same nouns under different templates. The claim
    "no entity or template straddles a split" did not hold.
  * Nothing measured selectivity. `correct()` compared the target and alternative verb only, so a feature
    that flips the verb for ANY number cue scored identically to one tracking the grammatical subject.

WHAT THIS DOES INSTEAD:
  * A much larger bank (10 templates x 40 subject lemmas x 20 distractor lemmas x 2 x 2 = 32,000 distinct
    SVA prompts) and exhaustive deduplicated enumeration -- never sampling with replacement.
  * Splits partition TEMPLATES, SUBJECT LEMMAS and DISTRACTOR LEMMAS independently; an item joins a split
    only if all three of its parts belong to it. `assert_disjoint` then checks all three intersections are
    empty, rather than inferring it from a joint key.
  * Every target item carries a FOIL: same recipient, but a donor differing only in the DISTRACTOR's
    number, with the subject number held fixed. The correct verb is unchanged for the foil, so applying
    the same mask and the same donor-patch rule tests whether the edit tracks the grammatical subject or
    merely follows any number cue. Control preservation = the foil keeps its original correct preference.
  * Balanced sampling across the whole split on (subject_plural, distractor_plural), not a prefix slice.
"""
import hashlib
import itertools
import json
import random

SVA_TEMPLATES = [
    ("to_the", "The {subj} to the {dist}"), ("near_the", "The {subj} near the {dist}"),
    ("beside", "The {subj} beside the {dist}"), ("inside", "The {subj} inside the {dist}"),
    ("behind", "The {subj} behind the {dist}"), ("under", "The {subj} under the {dist}"),
    ("across_from", "The {subj} across from the {dist}"), ("next_to", "The {subj} next to the {dist}"),
    ("in_front_of", "The {subj} in front of the {dist}"), ("opposite", "The {subj} opposite the {dist}"),
    # bank expansion 2026-10-06 (>=1,000 eligible held-out pairs per model): prepositional
    # distractors only, so no template word carries the subject's number
    ("above", "The {subj} above the {dist}"), ("below", "The {subj} below the {dist}"),
    ("outside", "The {subj} outside the {dist}"), ("past", "The {subj} past the {dist}"),
    ("beyond", "The {subj} beyond the {dist}"), ("on_top_of", "The {subj} on top of the {dist}"),
    ("at_the_back_of", "The {subj} at the back of the {dist}"),
    ("at_the_end_of", "The {subj} at the end of the {dist}"), ("close_to", "The {subj} close to the {dist}"),
    ("far_from", "The {subj} far from the {dist}"),
]
SUBJ = [("key", "keys"), ("door", "doors"), ("report", "reports"), ("author", "authors"),
        ("player", "players"), ("engine", "engines"), ("teacher", "teachers"), ("window", "windows"),
        ("letter", "letters"), ("painting", "paintings"), ("lamp", "lamps"), ("switch", "switches"),
        ("ticket", "tickets"), ("bottle", "bottles"), ("camera", "cameras"), ("printer", "printers"),
        ("manager", "managers"), ("driver", "drivers"), ("singer", "singers"), ("farmer", "farmers"),
        ("bridge", "bridges"), ("tunnel", "tunnels"), ("garden", "gardens"), ("statue", "statues"),
        ("clock", "clocks"), ("mirror", "mirrors"), ("carpet", "carpets"), ("curtain", "curtains"),
        ("speaker", "speakers"), ("monitor", "monitors"), ("folder", "folders"), ("basket", "baskets"),
        ("jacket", "jackets"), ("helmet", "helmets"), ("ladder", "ladders"), ("anchor", "anchors"),
        ("banner", "banners"), ("candle", "candles"), ("crate", "crates"), ("pillar", "pillars"),
        ("pencil", "pencils"), ("chair", "chairs"), ("table", "tables"), ("bucket", "buckets"),
        ("sign", "signs"), ("poster", "posters"), ("blanket", "blankets"), ("kettle", "kettles"),
        ("cushion", "cushions"), ("trophy", "trophies"), ("guitar", "guitars"), ("violin", "violins"),
        ("drawer", "drawers"), ("shelf", "shelves"), ("barrel", "barrels"), ("lantern", "lanterns"),
        ("bicycle", "bicycles"), ("tractor", "tractors"), ("trolley", "trolleys"), ("cart", "carts"),
        ("nurse", "nurses"), ("doctor", "doctors"), ("lawyer", "lawyers"), ("student", "students"),
        ("worker", "workers"), ("visitor", "visitors"), ("guard", "guards"), ("pilot", "pilots"),
        ("plant", "plants"), ("flower", "flowers"), ("tree", "trees"), ("fence", "fences"),
        ("gate", "gates"), ("wall", "walls"), ("tower", "towers"), ("cabin", "cabins")]
DIST = [("cabinet", "cabinets"), ("building", "buildings"), ("company", "companies"),
        ("school", "schools"), ("machine", "machines"), ("office", "offices"),
        ("station", "stations"), ("library", "libraries"), ("factory", "factories"),
        ("hospital", "hospitals"), ("theatre", "theatres"), ("museum", "museums"),
        ("warehouse", "warehouses"), ("stadium", "stadiums"), ("gallery", "galleries"),
        ("terminal", "terminals"), ("hangar", "hangars"), ("kitchen", "kitchens"),
        ("balcony", "balconies"), ("corridor", "corridors"),
        ("garage", "garages"), ("shed", "sheds"), ("barn", "barns"), ("church", "churches"),
        ("bank", "banks"), ("hotel", "hotels"), ("restaurant", "restaurants"), ("market", "markets"),
        ("airport", "airports"), ("harbor", "harbors"), ("castle", "castles"), ("palace", "palaces"),
        ("cottage", "cottages"), ("tent", "tents"), ("studio", "studios"), ("clinic", "clinics"),
        ("bakery", "bakeries"), ("pharmacy", "pharmacies"), ("laboratory", "laboratories"),
        ("classroom", "classrooms")]

SPLIT_FRACS = [("calibration", .08), ("train", .34), ("validation", .10),
               ("test", .30), ("confirmation", .18)]


def _sha(*p):
    return hashlib.sha1("|".join(map(str, p)).encode()).hexdigest()[:12]


def _single(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def partition(items, seed):
    """Split a list into the declared fractions, whole elements, deterministically."""
    r = random.Random(seed)
    xs = list(items)
    r.shuffle(xs)
    out, i = {}, 0
    n = len(xs)
    for k, f in SPLIT_FRACS:
        take = max(1, int(round(f * n)))
        out[k] = xs[i:i + take]
        i += take
    if i < n:
        out["train"] = out["train"] + xs[i:]
    return out


def build_sva(tok, seed=20260921):
    """Exhaustive, deduplicated, with per-item foils. Returns split -> [item]."""
    is_t, are_t = _single(tok, " is"), _single(tok, " are")
    if is_t is None or are_t is None:
        return None, dict(status="unsupported_task", why="' is'/' are' are not single tokens")
    tpl = partition(SVA_TEMPLATES, seed + 1)
    sub = partition(SUBJ, seed + 2)
    dis = partition(DIST, seed + 3)
    splits, meta = {}, {}
    for name, _ in SPLIT_FRACS:
        rows, seen = [], set()
        for (fam, t), (sg, pl), (dsg, dpl) in itertools.product(tpl[name], sub[name], dis[name]):
            for subj_pl in (False, True):
                for dist_pl in (False, True):
                    subj = pl if subj_pl else sg
                    dist = dpl if dist_pl else dsg
                    recip = t.format(subj=subj, dist=dist)
                    # TARGET donor: flip the SUBJECT number, hold the distractor fixed.
                    donor = t.format(subj=(sg if subj_pl else pl), dist=dist)
                    # FOIL donor: hold the SUBJECT number, flip only the DISTRACTOR number.
                    foil = t.format(subj=subj, dist=(dsg if dist_pl else dpl))
                    key = (recip, donor, foil)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(dict(
                        task="sva", item_id=_sha(name, fam, sg, dsg, subj_pl, dist_pl),
                        prompt=donor, corrupt_prompt=recip, foil_prompt=foil,
                        # after the donor patch the recipient should prefer the DONOR's verb
                        answer_tok=(is_t if subj_pl else are_t),
                        alt_tok=(are_t if subj_pl else is_t),
                        answer_str=(" is" if subj_pl else " are"),
                        alt_str=(" are" if subj_pl else " is"),
                        # the foil's correct verb follows the UNCHANGED subject number
                        foil_answer_tok=(are_t if subj_pl else is_t),
                        foil_alt_tok=(is_t if subj_pl else are_t),
                        recipient_subject_plural=subj_pl, distractor_plural=dist_pl,
                        subject_plural=subj_pl,   # legacy alias; refers to the RECIPIENT

                        template_family=fam, subject_lemma=sg, distractor_lemma=dsg, split=name))
        splits[name] = rows
        meta[name] = dict(n_items=len(rows), unique_recipients=len({r["corrupt_prompt"] for r in rows}),
                          unique_pairs=len({(r["prompt"], r["corrupt_prompt"]) for r in rows}),
                          templates=sorted({r["template_family"] for r in rows}),
                          n_subject_lemmas=len({r["subject_lemma"] for r in rows}),
                          n_distractor_lemmas=len({r["distractor_lemma"] for r in rows}))
    return splits, dict(status="ok", per_split=meta,
                        bank=dict(templates=len(SVA_TEMPLATES), subject_lemmas=len(SUBJ),
                                  distractor_lemmas=len(DIST),
                                  distinct_prompts=len(SVA_TEMPLATES) * len(SUBJ) * len(DIST) * 4))


def assert_disjoint(splits, keys=("template_family", "subject_lemma", "distractor_lemma")):
    """Every listed attribute must have empty pairwise intersections across splits."""
    report, ok = {}, True
    names = [n for n, _ in SPLIT_FRACS if splits.get(n)]
    for k in keys:
        vals = {n: {r[k] for r in splits[n]} for n in names}
        bad = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = vals[a] & vals[b]
                if inter:
                    bad.append(dict(a=a, b=b, shared=sorted(inter)[:6], n=len(inter)))
        report[k] = dict(disjoint=not bad, violations=bad,
                         sizes={n: len(v) for n, v in vals.items()})
        ok = ok and not bad
    report["all_disjoint"] = ok
    return report


def balanced_sample(rows, n, seed=20260921):
    """Sample across the WHOLE split, balanced on (subject_plural, distractor_plural).

    v1 took the first 96/128/512 rows of a group-concatenated list, which selected only the first few
    groups. This draws evenly from every cell of the crossed design and from the whole split.
    """
    r = random.Random(seed)
    cells = {}
    for it in rows:
        cells.setdefault((it.get("subject_plural"), it.get("distractor_plural")), []).append(it)
    for v in cells.values():
        r.shuffle(v)
    out, i = [], 0
    order = sorted(cells)
    while len(out) < min(n, len(rows)):
        progressed = False
        for c in order:
            if i < len(cells[c]):
                out.append(cells[c][i])
                progressed = True
                if len(out) >= min(n, len(rows)):
                    break
        if not progressed:
            break
        i += 1
    return out


def margin(logprobs, item, which="target"):
    a = item["answer_tok"] if which == "target" else item["foil_answer_tok"]
    b = item["alt_tok"] if which == "target" else item["foil_alt_tok"]
    return float(logprobs[a] - logprobs[b])


def correct(logprobs, item, which="target"):
    return margin(logprobs, item, which) > 0


def manifest(splits, meta, disjoint, tok_name, tok_rev=None):
    return dict(generator="tasks2.sva", tokenizer=tok_name, tokenizer_revision=tok_rev,
                meta=meta, disjointness=disjoint,
                items={k: [dict(item_id=i["item_id"], prompt=i["prompt"],
                                corrupt_prompt=i["corrupt_prompt"], foil_prompt=i["foil_prompt"],
                                answer_str=i["answer_str"], alt_str=i["alt_str"],
                                template_family=i["template_family"],
                                subject_lemma=i["subject_lemma"],
                                distractor_lemma=i["distractor_lemma"],
                                subject_plural=i["subject_plural"],
                                distractor_plural=i["distractor_plural"]) for i in v]
                       for k, v in splits.items()})


if __name__ == "__main__":
    import argparse
    import pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", default="protocol/task_manifest_sva.json")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    splits, meta = build_sva(tok)
    if splits is None:
        print(json.dumps(meta, indent=1))
        raise SystemExit(0)
    dj = assert_disjoint(splits)
    print(json.dumps(dict(bank=meta["bank"], per_split=meta["per_split"],
                          all_disjoint=dj["all_disjoint"]), indent=1))
    for k in ("template_family", "subject_lemma", "distractor_lemma"):
        print("%-18s disjoint=%s sizes=%s" % (k, dj[k]["disjoint"], dj[k]["sizes"]))
    ex = splits["train"][0]
    print("\nexample target : %r -> %r  (recipient %r)" % (ex["prompt"], ex["answer_str"], ex["corrupt_prompt"]))
    keep = " are" if ex["recipient_subject_plural"] else " is"
    print("example foil   : %r  (recipient subject number unchanged; correct verb stays %r)"
          % (ex["foil_prompt"], keep))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(manifest(splits, meta, dj, a.tokenizer), indent=1))
    print("\nwrote", a.out)
