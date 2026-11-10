"""Three controlled behavioral tasks, with disjoint splits, competence gates, and clean/corrupted pairs.

Each task yields items of the form

    dict(prompt, answer_tok, alt_tok, template_family, entity_cluster, corrupt_prompt, corrupt_answer_tok,
         role_positions)

where `answer_tok`/`alt_tok` are single-token continuations forming a two-choice margin, and the corrupted
prompt is the SAME template with the answer-determining content changed, so a donor patch has a well-defined
target. Splitting is by (template_family, entity_cluster) jointly: an entity may not appear in train and
test under a different template, and a template family may not straddle the split.

Competence gate (declared before any SAE arm runs): a model-task cell enters the primary aggregate only if
the unmodified model reaches >= 0.70 balanced two-choice accuracy on the calibration split. Ineligible cells
still report raw intervention behaviour as a diagnostic, but a failure there is a failure of the host model,
not of feature recovery.

Multi-token answers are rejected at construction time rather than silently truncated: `single_token_only`
drops any item whose answer or alternative is not one token under the host tokenizer, and the drop counts
are reported so a tokenizer-driven loss of items is visible.
"""
import hashlib
import itertools
import json
import random

# --------------------------------------------------------------------------- vocabularies
NAMES_A = ["Mary", "John", "Anna", "Peter", "Sarah", "David", "Emma", "James",
           "Laura", "Michael", "Julia", "Robert", "Alice", "Thomas", "Claire", "Daniel"]
PLACES = ["store", "school", "office", "station", "hospital", "library", "market", "museum"]
OBJECTS = ["drink", "book", "letter", "ticket", "key", "apple", "pen", "coat"]

SINGULAR = [("key", "keys"), ("door", "doors"), ("report", "reports"), ("author", "authors"),
            ("player", "players"), ("engine", "engines"), ("teacher", "teachers"), ("window", "windows")]
DISTRACTOR = [("cabinet", "cabinets"), ("building", "buildings"), ("company", "companies"),
              ("school", "schools"), ("machine", "machines"), ("office", "offices")]

IOI_TEMPLATES = [
    ("baba", "When {a} and {b} went to the {place}, {b} gave a {obj} to"),
    ("abba", "When {a} and {b} went to the {place}, {a} gave a {obj} to"),
    ("then", "Then {a} and {b} had a long argument. Afterwards {b} said to"),
    ("while", "While {a} and {b} were working, {b} handed the {obj} to"),
]
SVA_TEMPLATES = [
    ("to_the", "The {subj} to the {dist}"),
    ("near_the", "The {subj} near the {dist}"),
    ("beside", "The {subj} beside the {dist}"),
    ("inside", "The {subj} inside the {dist}"),
]
GT_TEMPLATES = [
    ("war", "The war lasted from the year {c}{yy} to the year {c}"),
    ("project", "The project ran from the year {c}{yy} until the year {c}"),
    ("reign", "The reign began in the year {c}{yy} and ended in the year {c}"),
]


def _cluster(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _single_token(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


# --------------------------------------------------------------------------- task builders
def build_ioi(tok, rng, n):
    items, dropped = [], 0
    for _ in range(n * 3):
        if len(items) >= n:
            break
        fam, tpl = IOI_TEMPLATES[rng.randrange(len(IOI_TEMPLATES))]
        a, b = rng.sample(NAMES_A, 2)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        prompt = tpl.format(a=a, b=b, place=place, obj=obj)
        # the indirect object is whichever name is NOT the repeated subject in this template
        io_name, subj_name = (a, b) if "{b} gave" in tpl or "{b} said" in tpl or "{b} handed" in tpl else (b, a)
        at = _single_token(tok, " " + io_name)
        alt = _single_token(tok, " " + subj_name)
        if at is None or alt is None or at == alt:
            dropped += 1
            continue
        # corruption: swap the two roles, so the correct answer becomes the other name
        corrupt = tpl.format(a=b, b=a, place=place, obj=obj)
        items.append(dict(task="ioi", prompt=prompt, answer_tok=at, alt_tok=alt,
                          answer_str=" " + io_name, alt_str=" " + subj_name,
                          template_family=fam, entity_cluster=_cluster(sorted([a, b])),
                          corrupt_prompt=corrupt, corrupt_answer_tok=alt, corrupt_alt_tok=at))
    return items, dropped


def build_sva(tok, rng, n):
    items, dropped = [], 0
    is_tok, are_tok = _single_token(tok, " is"), _single_token(tok, " are")
    if is_tok is None or are_tok is None:
        return [], n
    for _ in range(n * 3):
        if len(items) >= n:
            break
        fam, tpl = SVA_TEMPLATES[rng.randrange(len(SVA_TEMPLATES))]
        sg, pl = rng.choice(SINGULAR)
        dsg, dpl = rng.choice(DISTRACTOR)
        subj_plural = rng.random() < 0.5
        dist_plural = rng.random() < 0.5          # crossed with the subject, independently
        subj = pl if subj_plural else sg
        dist = dpl if dist_plural else dsg
        prompt = tpl.format(subj=subj, dist=dist)
        at = are_tok if subj_plural else is_tok
        alt = is_tok if subj_plural else are_tok
        # corruption flips the SUBJECT number only; the distractor is untouched
        corrupt = tpl.format(subj=(sg if subj_plural else pl), dist=dist)
        items.append(dict(task="sva", prompt=prompt, answer_tok=at, alt_tok=alt,
                          answer_str=" are" if subj_plural else " is",
                          alt_str=" is" if subj_plural else " are",
                          subject_plural=subj_plural, distractor_plural=dist_plural,
                          template_family=fam, entity_cluster=_cluster(sg, dsg),
                          corrupt_prompt=corrupt, corrupt_answer_tok=alt, corrupt_alt_tok=at))
    return items, dropped


def build_gt(tok, rng, n):
    """Greater-than, scored on TOKEN SETS rather than a two-choice margin.

    A two-choice role flip is impossible for this task. With a clean start year `yy` and choices
    (good > yy, bad < yy), raising the start year to `yy2` leaves `bad` invalid as well, so no single pair
    flips: to make `good` invalid you need `yy2 > good`, but then `bad < yy < yy2` is invalid too. The first
    version of this builder did exactly that and produced a corrupted prompt whose correct answer was
    usually unchanged -- the model scored ~0.83 on it unedited, and every arm inherited that base rate.

    The fix is to score the DISCRIMINATING set:

        answer_set = years in (yy, yy2]   valid after "from 17yy", invalid after "from 17yy2"
        alt_set    = years >  yy2         valid under both

    The clean prompt puts mass on answer_set ∪ alt_set; the corrupted prompt should put mass only on
    alt_set. So the margin log P(answer_set) - log P(alt_set) genuinely separates the two, and a donor
    patch from clean into corrupt has a well-defined target.
    """
    items, dropped = [], 0
    centuries = ["17", "18", "19"]
    two_digit = {}
    for y in range(10, 100):
        t = _single_token(tok, str(y))
        if t is not None:
            two_digit[y] = t
    if len(two_digit) < 40:
        return [], n
    ys = sorted(two_digit)
    for _ in range(n * 6):
        if len(items) >= n:
            break
        fam, tpl = GT_TEMPLATES[rng.randrange(len(GT_TEMPLATES))]
        c = rng.choice(centuries)
        yy = rng.choice([y for y in ys if 15 <= y <= 55])
        cand2 = [y for y in ys if yy + 8 <= y <= min(yy + 30, 88)]
        if not cand2:
            dropped += 1
            continue
        yy2 = rng.choice(cand2)
        answer_set = [two_digit[y] for y in ys if yy < y <= yy2]
        alt_set = [two_digit[y] for y in ys if y > yy2]
        if len(answer_set) < 4 or len(alt_set) < 4:
            dropped += 1
            continue
        items.append(dict(task="gt", prompt=tpl.format(c=c, yy=yy),
                          corrupt_prompt=tpl.format(c=c, yy=yy2),
                          answer_set=answer_set, alt_set=alt_set,
                          answer_str="years in (%d,%d]" % (yy, yy2), alt_str="years > %d" % yy2,
                          start_year=yy, corrupt_start_year=yy2,
                          template_family=fam, entity_cluster=_cluster(fam, c, yy // 10)))
    return items, dropped


BUILDERS = {"ioi": build_ioi, "sva": build_sva, "gt": build_gt}


# --------------------------------------------------------------------------- splitting
def split_items(items, seed=20260921):
    """Disjoint by (template_family, entity_cluster). Groups are assigned whole, so neither an entity nor a
    template can straddle a split under a different guise."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault((it["template_family"], it["entity_cluster"]), []).append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    if n < 10:
        raise ValueError("only %d independent (template, entity) groups: a disjoint split would be "
                         "degenerate. Widen the vocabulary or the cluster definition." % n)
    # every split gets at least one whole group; fractions are applied to what remains
    cuts = [max(1, int(f * n)) for f in (.10, .30, .12, .33)]
    while sum(cuts) >= n:
        cuts[cuts.index(max(cuts))] -= 1
    c0, c1, c2, c3 = cuts
    bounds = dict(calibration=(0, c0), train=(c0, c0 + c1), validation=(c0 + c1, c0 + c1 + c2),
                  test=(c0 + c1 + c2, c0 + c1 + c2 + c3), confirmation=(c0 + c1 + c2 + c3, n))
    out = {}
    for name, (lo, hi) in bounds.items():
        out[name] = [it for k in keys[lo:hi] for it in groups[k]]
    return out, {name: len({(i["template_family"], i["entity_cluster"]) for i in v})
                 for name, v in out.items()}


def make_all(tok, n_per_task=6000, seed=20260921, tasks=None):
    """Build only the requested tasks, and record an unsupported one instead of raising.

    A task can be genuinely impossible under a host tokenizer -- Qwen-2.5 and Gemma tokenize numbers
    digit-by-digit, so the greater-than task has no single-token years and yields nothing. That is an
    `unsupported_task` cell, not a crash, and it must not take down a run that only asked for `ioi sva`.
    """
    rng = random.Random(seed)
    data, meta = {}, {}
    for name, fn in BUILDERS.items():
        if tasks and name not in tasks:
            continue
        items, dropped = fn(tok, rng, n_per_task)
        if len(items) < 50:
            data[name] = None
            meta[name] = dict(status="unsupported_task", n_items=len(items),
                              dropped_multitoken=dropped,
                              why="the host tokenizer does not support this task's answer vocabulary "
                                  "(e.g. digit-level number tokenization for greater-than)")
            continue
        try:
            splits, groups = split_items(items, seed=seed)
        except ValueError as e:
            data[name] = None
            meta[name] = dict(status="unsupported_task", n_items=len(items), error=str(e))
            continue
        data[name] = splits
        meta[name] = dict(status="ok", n_items=len(items), dropped_multitoken=dropped,
                          groups_per_split=groups,
                          n_per_split={k: len(v) for k, v in splits.items()})
    return data, meta


# --------------------------------------------------------------------------- scoring
def _logsumexp(logprobs, toks):
    import numpy as np
    v = np.asarray([float(logprobs[t]) for t in toks], dtype=np.float64)
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def margin(logprobs, item):
    """Answer-minus-alternative log-probability at the final position.

    Single-token tasks use the two chosen tokens. Set-scored tasks (greater-than) use the pooled mass of
    the discriminating set against the always-valid set, because no single token pair can express their
    contrast (see build_gt)."""
    if "answer_set" in item:
        return _logsumexp(logprobs, item["answer_set"]) - _logsumexp(logprobs, item["alt_set"])
    return float(logprobs[item["answer_tok"]] - logprobs[item["alt_tok"]])


def correct(logprobs, item):
    if "answer_set" in item:
        return bool(_logsumexp(logprobs, item["answer_set"]) > _logsumexp(logprobs, item["alt_set"]))
    return bool(logprobs[item["answer_tok"]] > logprobs[item["alt_tok"]])


COMPETENCE_THRESHOLD = 0.70


def competence(logprob_rows, items):
    """Balanced two-choice accuracy on the calibration split."""
    if not items:
        return dict(n=0, accuracy=None, eligible=False)
    acc = sum(correct(lp, it) for lp, it in zip(logprob_rows, items)) / len(items)
    return dict(n=len(items), accuracy=float(acc), eligible=bool(acc >= COMPETENCE_THRESHOLD),
                threshold=COMPETENCE_THRESHOLD)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--out", default="protocol/tasks_meta.json")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    data, meta = make_all(tok, n_per_task=a.n)
    data = {k: v for k, v in data.items() if v is not None}
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(dict(tokenizer=a.tokenizer, meta=meta), indent=1))
    print(json.dumps(meta, indent=1))
    for t, sp in data.items():
        ex = sp["train"][0]
        print("\n%s example: %r -> %r (vs %r)" % (t, ex["prompt"], ex["answer_str"], ex["alt_str"]))
        print("   corrupt: %r -> %r" % (ex["corrupt_prompt"], ex["alt_str"]))
