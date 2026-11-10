"""Model loading, the intervention site, the SAE splice, and the cached-suffix runner.

THE SUFFIX RUNNER
-----------------
Every per-feature intervention changes the residual stream only at the site (the output of
block b). Re-running blocks 0..b for each intervention recomputes an identical prefix, so the
runner caches the prefix once and runs only blocks b+1..L, the final norm and the head.

It does this WITHOUT re-implementing any model's forward pass, which is where version- and
family-specific bugs live (rotary embeddings, sliding-window masks, Gemma's soft-capping and
embedding normalizer, GPT-2's learned positions). Instead it temporarily replaces the block list
with blocks[b+1:], calls the model's own forward with a dummy `inputs_embeds`, and installs a
forward PRE-hook on the first remaining block that swaps in the cached site residual as that
block's `hidden_states`. Everything the model does to its input before the first block --
Gemma's sqrt(d) normalizer, GPT-2's position embeddings -- is thereby discarded exactly, with no
divide-then-multiply rounding, and everything after (masks, rotary, softcap, head) is the
model's own code. Correctness is not assumed: gate 7 compares it against a full forward with the
v3 splice hook, per model family, before any measurement is recorded.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common  # noqa: F401  (sets sys.path for v3)
from v3.common import registry
from v3.common import data as v3data
from v3.common import saes as v3saes


class _Stop(Exception):
    pass


def spec_of(name):
    return registry.MODELS[name]


def load(spec, device="cuda", device_map=None, dtype=None, headroom=0.80):
    """v3's loader (same dtype and attention implementation as every archived cell).
    `dtype` overrides the registry dtype ONLY for local CPU logic tests (CPU has no fp16 BLAS);
    every recorded measurement uses the registry dtype."""
    if device_map == "auto" and torch.cuda.is_available():
        # accelerate's "auto" fills every GPU to the brim, leaving no room for activations,
        # caches and the SAE; cap each device explicitly (OOM'd a 4xL40S 70B box otherwise).
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mm = {i: int(torch.cuda.get_device_properties(i).total_memory * headroom)
              for i in range(torch.cuda.device_count())}
        kw = dict(torch_dtype=dtype or getattr(torch, spec.dtype), device_map="auto",
                  max_memory=mm)
        if spec.family == "gemma":
            kw["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw).eval()
    else:
        model, tok = v3data.load_model_and_tokenizer(spec, device=device,
                                                     device_map=device_map, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _parent_and_attr(model, block_path):
    parts = block_path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def blocks(model, spec):
    return v3data.get_blocks(model, spec.block_path)


def site_index(spec, layer=None):
    return registry.block_index(spec, layer)


def input_device(model):
    return model.get_input_embeddings().weight.device


def site_device(model, spec, layer=None):
    return next(blocks(model, spec)[site_index(spec, layer)].parameters()).device


@torch.no_grad()
def prefix(model, spec, ids, am, layer=None):
    """Residual stream at the site (output of block b), stopping the forward right there."""
    out = {}
    b = blocks(model, spec)[site_index(spec, layer)]

    def hook(m, i, o):
        out["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        raise _Stop

    hd = b.register_forward_hook(hook)
    try:
        model(ids.to(input_device(model)), attention_mask=am.to(input_device(model)))
    except _Stop:
        pass
    finally:
        hd.remove()
    return out["h"]


@torch.no_grad()
def all_hidden(model, ids, am):
    """Native forward returning every hidden state and the logits (gates 1/2, layer profile)."""
    o = model(ids.to(input_device(model)), attention_mask=am.to(input_device(model)),
              output_hidden_states=True)
    return o.hidden_states, o.logits


class Suffix:
    """Run blocks b+1..L + final norm + head from a site residual. See the module docstring."""

    def __init__(self, model, spec, layer=None):
        self.model, self.spec = model, spec
        self.b = site_index(spec, layer)
        self.parent, self.attr = _parent_and_attr(model, spec.block_path)
        self.full = getattr(self.parent, self.attr)
        self.tail = nn.ModuleList(list(self.full)[self.b + 1:])
        self.emb_dim = model.get_input_embeddings().weight.shape[1]
        self.emb_dtype = model.get_input_embeddings().weight.dtype

    @contextlib.contextmanager
    def _truncated(self, h):
        holder = {"h": h}

        def pre(mod, args, kwargs):
            hh = holder["h"]
            if args:
                args = (hh.to(args[0].device) if torch.is_tensor(args[0]) else hh,) + tuple(args[1:])
            else:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hh.to(kwargs["hidden_states"].device)
            return args, kwargs

        if len(self.tail) == 0:
            raise RuntimeError("site is the last block; no suffix to run")
        hd = self.tail[0].register_forward_pre_hook(pre, with_kwargs=True)
        setattr(self.parent, self.attr, self.tail)
        try:
            yield
        finally:
            setattr(self.parent, self.attr, self.full)
            hd.remove()

    def logits(self, h, am):
        """h: [B,T,d] site residual in the MODEL's dtype. Returns [B,T,V] logits."""
        B, T = h.shape[:2]
        dummy = torch.zeros(B, T, self.emb_dim, dtype=self.emb_dtype, device=input_device(self.model))
        with self._truncated(h):
            out = self.model(inputs_embeds=dummy, attention_mask=am.to(input_device(self.model)),
                             use_cache=False)
        return out.logits


def _base_model(model):
    for name in ("model", "transformer", "gpt_neox"):
        m = getattr(model, name, None)
        if m is not None and hasattr(m, "forward") and m is not model:
            return m
    raise RuntimeError("no base model on %s" % type(model).__name__)


def _final_norm(base):
    for name in ("norm", "ln_f", "final_layer_norm"):
        m = getattr(base, name, None)
        if m is not None:
            return m
    raise RuntimeError("no final norm on %s" % type(base).__name__)


def suffix_nll(suf, h, ids, am, chunk_rows=2048, need_grad=False):
    """Per-position next-token NLL [B,T] (fp32) and validity mask, from a site residual.

    Runs the truncated BASE model (final norm included), then the LM head in row chunks with
    the family's final logit soft-cap, so a 256k-vocab model never materialises [B,T,V] fp32.
    Equal to token_nll(suf.logits(h, am), ids, am) -- checked by gate 7.
    """
    model = suf.model
    base = _base_model(model)
    head = model.get_output_embeddings()
    cap = getattr(model.config, "final_logit_softcapping", None)
    B, T = h.shape[:2]
    dummy = torch.zeros(B, T, suf.emb_dim, dtype=suf.emb_dtype, device=input_device(model))
    ctx = contextlib.nullcontext() if need_grad else torch.no_grad()
    with ctx:
        if len(suf.tail) == 0:          # site is the last block: final norm + head only
            hid = _final_norm(base)(h.to(next(_final_norm(base).parameters()).device))
        else:
            with suf._truncated(h):
                hid = base(inputs_embeds=dummy, attention_mask=am.to(input_device(model)),
                           use_cache=False).last_hidden_state
        hid = hid[:, :-1].reshape(-1, hid.shape[-1])
        tgt = ids[:, 1:].reshape(-1).to(hid.device)
        out = []
        for i in range(0, hid.shape[0], chunk_rows):
            lg = head(hid[i:i + chunk_rows].to(head.weight.device)).float()
            if cap:
                lg = torch.tanh(lg / cap) * cap
            lp = torch.log_softmax(lg, dim=-1)
            out.append(-lp.gather(-1, tgt[i:i + chunk_rows].to(lp.device).unsqueeze(-1)).squeeze(-1))
        nll = torch.cat(out).reshape(B, T - 1)
    valid = (am[:, 1:].to(nll.device) != 0)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    nll = torch.cat([nll, torch.zeros(B, 1, device=nll.device, dtype=nll.dtype)], 1)
    valid = torch.cat([valid, torch.zeros(B, 1, device=nll.device, dtype=torch.bool)], 1)
    return nll, valid


def token_nll(logits, ids, am):
    """Per-position next-token NLL in float32, [B, T]; position t predicts ids[t+1].
    Positions without a valid target (last position, padding) are 0 and masked by `valid`."""
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:].to(lp.device)
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    valid = (am[:, 1:].to(lp.device) != 0)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    pad = torch.zeros(nll.shape[0], 1, device=nll.device, dtype=nll.dtype)
    vpad = torch.zeros(nll.shape[0], 1, device=nll.device, dtype=torch.bool)
    return torch.cat([nll, pad], 1), torch.cat([valid, vpad], 1)


# ---------------------------------------------------------------------------------------
# dictionaries
# ---------------------------------------------------------------------------------------
class Dictionary:
    """One SAE plus the exact v3 splice equations for its condition.

      standard: h' = SAE(h - mu) + mu
      bypass  : h' = SAE(P_perp(h - mu)) + <h - mu, v1> v1 + mu
    The arithmetic is float32 and the result is cast to the model dtype once, exactly as the
    v3 hook does, so a spliced residual computed here equals the one the hook installs.
    """

    def __init__(self, sae, mode, mu, v1, device):
        self.sae = sae.to(device).eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)
        self.mode = mode
        self.mu = torch.as_tensor(np.asarray(mu), dtype=torch.float32, device=device)
        self.v1 = torch.as_tensor(np.asarray(v1), dtype=torch.float32, device=device)
        self.v1 = self.v1 / self.v1.norm()
        self.W_dec = self.sae.W_dec.detach().float()          # [d_sae, d]
        self.b_dec = self.sae.b_dec.detach().float()
        self.d_sae = self.W_dec.shape[0]
        self.device = device

    def encode(self, h):
        """h [N, d] (any dtype) -> (z [N, d_sae] fp32, coef [N] or None)."""
        x = h.float().to(self.device) - self.mu
        coef = None
        if self.mode == "bypass":
            coef = x @ self.v1
            x = x - coef[:, None] * self.v1
        return self.sae.encode(x), coef

    def splice_fp32(self, z, coef):
        """Spliced residual in float32 BEFORE the final cast (so deltas can be added exactly)."""
        r = z @ self.W_dec + self.b_dec + self.mu
        if coef is not None:
            r = r + coef[..., None] * self.v1
        return r


def load_dictionary(path, mode, mu, v1, device):
    sae, extra = v3saes.load_sae(str(path), device=device)
    return Dictionary(sae, mode, mu, v1, device), extra


@contextlib.contextmanager
def splice_hook(model, spec, dic, layer=None, z_transform=None):
    """Full-forward splice using THIS module's Dictionary (gate 7's reference path)."""
    b = blocks(model, spec)[site_index(spec, layer)]

    def hook(m, i, o):
        is_t = isinstance(o, tuple)
        h = o[0] if is_t else o
        B, T, d = h.shape
        z, coef = dic.encode(h.reshape(-1, d))
        if z_transform is not None:
            z = z_transform(z.reshape(B, T, -1)).reshape(B * T, -1)
        r = dic.splice_fp32(z, coef).reshape(B, T, d).to(h.dtype).to(h.device)
        return (r,) + tuple(o[1:]) if is_t else r

    hd = b.register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()
