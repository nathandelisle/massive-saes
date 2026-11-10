"""Workstream C, part 2: explanations and blind detection scoring with two judge families.

Families (pinned):
  A  Qwen/Qwen2.5-32B-Instruct                @ 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
  B  mistralai/Mistral-Small-24B-Instruct-2501 @ 9527884be6e5616bdd54de542f9ae13384489724

Cross-evaluation: A explains / B judges, and B explains / A judges, on the same held-out
context ids. Judges are blind to SAE condition, seed, ranking population, PC1 alignment and
activation values: every judge prompt contains only the explanation and ONE excerpt.

Explainer output is structured JSON with a separate no-pattern flag; parse failures retry at
most twice with a fixed correction message; anything still unparsable is recorded as such and
still scored with the raw text, never dropped from a denominator. A "no pattern" explanation is
also scored. Detection score = logit("Yes") - logit("No") at the judge's first answer token.

Primary causal-context scoring shows only the LEFT context (what the model had at the target);
nothing two-sided enters the primary score.
"""
import argparse
import gc
import json
import re
import time

import numpy as np
import torch

from . import common
from .common import log

FAMILIES = {
    "qwen": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
    "mistral": ("mistralai/Mistral-Small-24B-Instruct-2501", "9527884be6e5616bdd54de542f9ae13384489724"),
}
EXPLAIN_SYS = ("You are helping interpret features of a language model. You will see short "
               "excerpts of text. In each excerpt one token is marked like <<this>>; a hidden "
               "feature of the model is active on that token.")
EXPLAIN_USER = ("Excerpts, strongest activations first:\n\n{ctx}\n\nIn one short sentence, "
                "describe what this feature responds to (the marked token and/or what precedes "
                "it). If you see no consistent pattern, say so. Respond with ONLY a JSON object: "
                '{{"explanation": "<one sentence>", "no_pattern": <true or false>}}')
RETRY = ("Your previous reply was not a valid JSON object of the required form. Reply with ONLY "
         '{"explanation": "<one sentence>", "no_pattern": <true or false>}.')
JUDGE_USER = ("A feature of a language model is described as follows:\n\"{expl}\"\n\nHere is a "
              "text excerpt. The final token is marked like <<this>>.\n\nExcerpt: {ctx}\n\n"
              "Is this feature likely to be active on the marked token? Answer with a single "
              "word: Yes or No.")


def fmt_ctx(c):
    return (c["left_text"] + "<<" + c["target_text"] + ">>").replace("\n", " ")


class Judge:
    def __init__(self, family, device="auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name, rev = FAMILIES[family]
        self.family, self.name, self.rev = family, name, rev
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=rev, torch_dtype=torch.bfloat16, device_map=device).eval()
        # first-token ids of every surface form the answer can start with; the score is
        # logsumexp(Yes forms) - logsumexp(No forms) at the first generated position
        def ids(words):
            s = set()
            for w in words:
                e = self.tok.encode(w, add_special_tokens=False)
                if e:
                    s.add(e[0])
            return sorted(s)
        self.yes, self.no = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
        assert not set(self.yes) & set(self.no), "Yes/No first tokens collide"
        self.n_prompt_tokens = 0
        self.n_gen_tokens = 0

    def _chat(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts, max_new=120, batch=8):
        out = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            new = g[:, enc["input_ids"].shape[1]:]
            self.n_prompt_tokens += int(enc["attention_mask"].sum())
            self.n_gen_tokens += int((new != self.tok.pad_token_id).sum())
            out += self.tok.batch_decode(new, skip_special_tokens=True)
        return out

    @torch.no_grad()
    def yes_no(self, prompts, batch=64):
        out = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            lg = self.model(**enc).logits[:, -1].float()
            self.n_prompt_tokens += int(enc["attention_mask"].sum())
            ls = torch.log_softmax(lg, -1)
            out += (torch.logsumexp(ls[:, self.yes], -1) - torch.logsumexp(ls[:, self.no], -1)).cpu().tolist()
        return out

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d.get("explanation"), str):
        return None
    return dict(explanation=d["explanation"].strip(), no_pattern=bool(d.get("no_pattern", False)))


def explain_all(J, feats):
    """feats: list of (key, contexts). Returns key -> record (explanation, flags, raw, retries)."""
    msgs = {k: [dict(role="system", content=EXPLAIN_SYS),
                dict(role="user", content=EXPLAIN_USER.format(
                    ctx="\n".join("- " + fmt_ctx(c) for c in ctxs)))] for k, ctxs in feats}
    out = {k: dict(raw=[], retries=0) for k, _ in feats}
    pending = list(msgs)
    for attempt in range(3):
        if not pending:
            break
        gens = J.generate([J._chat(msgs[k]) for k in pending])
        nxt = []
        for k, g in zip(pending, gens):
            out[k]["raw"].append(g)
            p = parse(g)
            if p:
                out[k].update(p, parse_ok=True)
            else:
                out[k]["retries"] += 1
                msgs[k] = msgs[k] + [dict(role="assistant", content=g), dict(role="user", content=RETRY)]
                nxt.append(k)
        pending = nxt
    for k in pending:
        out[k].update(explanation=out[k]["raw"][-1].strip()[:400], no_pattern=None, parse_ok=False)
    return out


def judge_all(J, expl, evals):
    """expl: key -> explanation text; evals: key -> list of eval contexts. Returns key -> scores."""
    keys, prompts = [], []
    for k, ctxs in evals.items():
        for c in ctxs:
            keys.append(k)
            prompts.append(J._chat([dict(role="user", content=JUDGE_USER.format(
                expl=expl[k], ctx=fmt_ctx(c)))]))
    sc = J.yes_no(prompts)
    out = {}
    for k, s in zip(keys, sc):
        out.setdefault(k, []).append(float(s))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicts", nargs="+", required=True, help="model:condition:seed ...")
    a = ap.parse_args(argv)
    out = common.RESULTS / "judge"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {}
    for spec in a.dicts:
        m, c, s = spec.split(":")
        f = common.WORK / "ctx" / ("%s_%s_s%s.json" % (m, c, s))
        common.s3_get("%s/%s/contexts/%s_s%s.json" % (common.SHARED, m, c, s), f)
        data[spec] = json.loads(f.read_text())
    feats_explain, evals = [], {}
    for spec, d in data.items():
        for r in d["features"]:
            if r.get("status") not in ("OK", "TOO_FEW_POSITIVES"):
                continue
            k = "%s|%d" % (spec, r["feature"])
            feats_explain.append((k, r["explain"]))
            evals[k] = r["eval"]
    log("judge", "%d features over %d dictionaries" % (len(feats_explain), len(data)))
    res = dict(families={k: dict(name=v[0], revision=v[1]) for k, v in FAMILIES.items()},
               dicts=a.dicts, explanations={}, scores={}, tokens={})
    order = [("qwen", "explain", None), ("mistral", "explain", None),
             ("mistral", "judge", "qwen"), ("qwen", "judge", "mistral")]
    J = None
    for fam, what, other in order:
        f = out / ("%s_%s%s.json" % (fam, what, "" if other is None else "_" + other))
        if common.fresh(f.parent, f.name, "judge"):
            obj = json.loads(f.read_text())
        else:
            if J is None or J.family != fam:
                if J is not None:
                    J.close()
                J = Judge(fam)
            if what == "explain":
                obj = explain_all(J, feats_explain)
            else:
                ex = json.loads((out / ("%s_explain.json" % other)).read_text())
                obj = judge_all(J, {k: v["explanation"] for k, v in ex.items()}, evals)
            common.write_json(f, obj)
            common.pin(f.parent, f.name, "judge")
            res["tokens"]["%s_%s" % (fam, what)] = dict(prompt=J.n_prompt_tokens, generated=J.n_gen_tokens)
            log("judge", "%s %s done (%.0fs)" % (fam, what, time.time() - t0))
    if J is not None:
        J.close()
    res["seconds"] = time.time() - t0
    common.write_json(out / "judge_meta.json", res)
    return dict(n_features=len(feats_explain))
