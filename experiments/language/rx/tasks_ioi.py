"""IOI task bank with the same discipline as tasks_sva (v2): parts partitioned before generation.

Clean (donor)   : "When A and B went to the P, B gave a O to"   -> answer A (indirect object)
Corrupt (recip.): "When A and B went to the P, A gave a O to"   -> answer B
The two prompts differ ONLY in the repeated subject token, so they align token-for-token
whenever every name is a single token for the tokenizer (checked; others are dropped).
Patching donor -> recipient should switch the recipient to the DONOR's answer (A).

FOIL: the recipient again, with the donor differing only in the ORDER of the two names
("When B and A ..., A gave a O to" -> answer still B). Applying the same mask and donor-patch
rule from the foil donor must leave the recipient's correct answer (B) intact. An edit that
tracks "which name is repeated" passes; one that tracks "which name came first" fails.

Splits partition TEMPLATES, NAMES, PLACES and OBJECTS independently; an item belongs to a split
only if all of its parts do, and `assert_disjoint` checks every intersection is empty.
"""
import hashlib
import zlib
import itertools
import random

TEMPLATES = [
    ("went", "When {A} and {B} went to the {P}, {S} gave a {O} to"),
    ("arrived", "After {A} and {B} arrived at the {P}, {S} handed a {O} to"),
    ("worked", "While {A} and {B} were working at the {P}, {S} passed a {O} to"),
    ("met", "When {A} and {B} met at the {P}, {S} showed a {O} to"),
    ("visited", "After {A} and {B} visited the {P}, {S} offered a {O} to"),
    ("walked", "As {A} and {B} walked into the {P}, {S} threw a {O} to"),
    ("stayed", "While {A} and {B} stayed at the {P}, {S} sent a {O} to"),
    ("returned", "When {A} and {B} returned to the {P}, {S} brought a {O} to"),
    ("waited", "While {A} and {B} waited at the {P}, {S} sold a {O} to"),
    ("cleaned", "After {A} and {B} cleaned the {P}, {S} lent a {O} to"),
]
NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "James", "Emma", "David", "Laura", "Michael",
         "Anna", "Paul", "Kate", "Mark", "Lisa", "Peter", "Jane", "Tom", "Rachel", "Daniel",
         "Susan", "Kevin", "Helen", "Brian", "Linda", "Steven", "Karen", "Jason", "Emily", "Ryan",
         "Amy", "Eric", "Julia", "Adam", "Nancy", "Scott", "Grace", "Henry", "Ruth", "Frank",
         "Claire", "George", "Diana", "Jack", "Rose", "Philip", "Lucy", "Sam", "Joan", "Carl",
         "Ellen", "Simon", "Alan", "Nora", "Victor", "Irene", "Oscar", "Paula", "Ralph", "Tina"]
PLACES = ["store", "park", "school", "office", "station", "garden", "library", "market",
          "museum", "hospital", "beach", "restaurant", "theater", "church", "bank", "hotel",
          "airport", "stadium", "factory", "farm"]
OBJECTS = ["book", "drink", "ball", "letter", "key", "gift", "ticket", "phone", "coin", "map",
           "cup", "pen", "card", "bag", "box", "flower", "toy", "hat", "watch", "camera"]
SPLIT_FRACS = [("calibration", .10), ("train", .30), ("validation", .15), ("test", .45)]


def _sha(*p):
    return hashlib.sha1("|".join(map(str, p)).encode()).hexdigest()[:12]


def _single(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _partition(items, seed):
    r = random.Random(seed)
    xs = list(dict.fromkeys(items))
    r.shuffle(xs)
    out, i, n = {}, 0, len(xs)
    for k, f in SPLIT_FRACS:
        take = max(2, int(round(f * n)))
        out[k] = xs[i:i + take]
        i += take
    if i < n:
        out["train"] = out["train"] + xs[i:]
    return out


def build_ioi(tok, seed=20261005, max_per_split=6000):
    names = [n for n in dict.fromkeys(NAMES) if _single(tok, " " + n) is not None]
    if len(names) < 16:
        return None, dict(status="unsupported_task",
                          why="only %d single-token names for this tokenizer" % len(names))
    tpl = _partition(TEMPLATES, seed + 1)
    nam = _partition(names, seed + 2)
    plc = _partition(PLACES, seed + 3)
    obj = _partition(OBJECTS, seed + 4)
    splits, meta = {}, {}
    for split, _ in SPLIT_FRACS:
        r = random.Random(seed * 31 + zlib.crc32(split.encode()))
        combos = list(itertools.product(tpl[split], itertools.permutations(nam[split], 2),
                                        plc[split], obj[split]))
        r.shuffle(combos)
        rows, seen = [], set()
        for (fam, t), (A, B), P, O in combos:
            recip = t.format(A=A, B=B, P=P, O=O, S=A)
            donor = t.format(A=A, B=B, P=P, O=O, S=B)
            foil = t.format(A=B, B=A, P=P, O=O, S=A)
            key = (recip, donor)
            if key in seen:
                continue
            seen.add(key)
            a_t, b_t = _single(tok, " " + A), _single(tok, " " + B)
            rows.append(dict(task="ioi", item_id=_sha(split, fam, A, B, P, O),
                             prompt=donor, corrupt_prompt=recip, foil_prompt=foil,
                             answer_tok=a_t, alt_tok=b_t, answer_str=" " + A, alt_str=" " + B,
                             foil_answer_tok=b_t, foil_alt_tok=a_t,
                             template_family=fam, name_a=A, name_b=B, place=P, obj=O,
                             subject_plural=None, distractor_plural=None, split=split))
            if len(rows) >= max_per_split:
                break
        splits[split] = rows
        meta[split] = dict(n_items=len(rows), templates=sorted({r["template_family"] for r in rows}),
                           n_names=len(nam[split]), n_places=len(plc[split]),
                           n_objects=len(obj[split]))
    return splits, dict(status="ok", per_split=meta, n_single_token_names=len(names))


def assert_disjoint(splits, keys=("template_family", "place", "obj")):
    report, ok = {}, True
    names = [n for n, _ in SPLIT_FRACS if splits.get(n)]
    for k in keys + ("names",):
        if k == "names":
            vals = {n: {x for r in splits[n] for x in (r["name_a"], r["name_b"])} for n in names}
        else:
            vals = {n: {r[k] for r in splits[n]} for n in names}
        bad = [dict(a=a, b=b, n=len(vals[a] & vals[b])) for i, a in enumerate(names)
               for b in names[i + 1:] if vals[a] & vals[b]]
        report[k] = dict(disjoint=not bad, violations=bad)
        ok = ok and not bad
    report["all_disjoint"] = ok
    return report
