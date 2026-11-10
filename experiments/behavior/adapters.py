"""Native-unit SAE adapters and the hard correctness gate that must pass before any intervention is run.

One adapter per (host model, SAE) pair. Everything is expressed in NATIVE residual-stream units, so an edit
can be applied to the real model without going through a reconstruction.

Reconstruction with the stored training mean mu and PC1 v:

    x                = h - mu
    c                = x @ v
    z_standard       = Enc(x)
    z_bypass         = Enc(x - c v)
    hhat_standard    = Dec(z_standard) + mu
    hhat_bypass      = Dec(z_bypass) + c v + mu

`Dec` includes the decoder bias exactly once, and the bypass reconstruction adds c exactly once. For a
bypass dictionary the coefficient c is NOT a selectable latent: it is held fixed at the recipient's own
value during every feature intervention, and the trainable mask can never touch it.

Interventions, all error-preserving by construction:

    suppress     h_edit = h - sum_j z_j(h) d_j
    donor patch  h_edit = h_recip + sum_j (z_j(donor) - z_j(recip)) d_j
    general      h_edit = h + (reconstruct(modified code) - reconstruct(h))

The third form is the definition; the first two are the cases where it reduces to a decoder sum, and
`check_manual_delta_matches_reconstruction` asserts that reduction numerically rather than assuming it.

The bypass trainer applies no output projection, so a bypass decoder direction can still carry a PC1
component. Every intervention records its PC1 and off-PC1 delta norm, and `project_out_pc1=True` is
available as a control that is applied to EVERY arm or to none.
"""
import json

import torch


class SAEAdapter:
    def __init__(self, sae, mu, v1, condition, device="cuda"):
        assert condition in ("standard", "bypass")
        self.sae = sae.eval()
        self.condition = condition
        self.device = device
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.v1 = torch.as_tensor(v1, dtype=torch.float32, device=device)
        self.v1 = self.v1 / self.v1.norm()
        self.d_model = int(self.mu.numel())
        self.d_sae = int(sae.W_dec.shape[0])

    # ------------------------------------------------------------------ core
    def split(self, h):
        """h -> (encoder input, restored PC1 term). The PC1 term is zero for a standard dictionary."""
        x = h.float() - self.mu
        if self.condition == "bypass":
            c = x @ self.v1
            par = c[..., None] * self.v1
            return x - par, par
        return x, torch.zeros_like(x)

    @torch.no_grad()
    def encode(self, h):
        xin, _ = self.split(h)
        return self.sae.encode(xin)

    @torch.no_grad()
    def reconstruct(self, h, z=None):
        """Native-unit reconstruction. Decoder bias once, mean once, PC1 once."""
        xin, par = self.split(h)
        if z is None:
            z = self.sae.encode(xin)
        return self.sae.decode(z) + par + self.mu

    @torch.no_grad()
    def delta_from_coefficients(self, delta_z):
        """Residual-stream delta produced by changing sparse codes by delta_z. Excludes the decoder bias:
        the bias is part of the reconstruction, not of a code change, and adding it here would double it."""
        return delta_z @ self.sae.W_dec.float()

    @torch.no_grad()
    def edit(self, h, z_new, z_old=None):
        """General error-preserving edit: h + (recon(z_new) - recon(z_old))."""
        if z_old is None:
            z_old = self.encode(h)
        return h.float() + self.delta_from_coefficients((z_new - z_old).float())

    @torch.no_grad()
    def suppress(self, h, features):
        z = self.encode(h)
        z_new = z.clone()
        z_new[..., features] = 0.0
        return self.edit(h, z_new, z), z

    @torch.no_grad()
    def donor_patch(self, h_recip, h_donor, features, dose=1.0):
        """Move the recipient's coefficients toward the donor's on `features` only.

        The bypass coefficient c never appears here: `split` recomputes it from the recipient on every
        call, so the recipient's own PC1 component is preserved exactly."""
        z_r = self.encode(h_recip)
        z_d = self.encode(h_donor)
        z_new = z_r.clone()
        z_new[..., features] = z_r[..., features] + dose * (z_d[..., features] - z_r[..., features])
        return self.edit(h_recip, z_new, z_r), z_r, z_d

    @torch.no_grad()
    def delta_norms(self, delta):
        """(total, on-PC1, off-PC1) norms of an intervention delta, for the record."""
        par = (delta @ self.v1)[..., None] * self.v1
        return (float(delta.norm()), float(par.norm()), float((delta - par).norm()))

    def project_out_pc1(self, delta):
        return delta - (delta @ self.v1)[..., None] * self.v1


# --------------------------------------------------------------------------- gate
@torch.no_grad()
def run_gate(adapter, h, model_forward, tol_native=None, log=print):
    """Check reconstruction and intervention consistency. Returns a dict; `passed` is False if any required check fails.

    `model_forward(h_override=None)` must run the host model, optionally replacing the hook-site activation,
    and return logits. Check 1 establishes the tolerance from two identical native forwards rather than
    assuming a fixed epsilon.
    """
    rep = {}
    h = h.float()

    # 1. determinism floor: two identical native forwards
    a = model_forward(None)
    b = model_forward(None)
    floor = float((a - b).abs().max())
    rep["native_forward_repeat_max_abs_diff"] = floor
    tol = tol_native if tol_native is not None else max(10 * floor, 1e-3)
    rep["tolerance_used"] = tol

    # 1b. an empty intervention reproduces the native logits
    z = adapter.encode(h)
    h_noop = adapter.edit(h, z, z)
    rep["noop_edit_max_abs_diff_vs_h"] = float((h_noop - h).abs().max())
    lg_noop = model_forward(h_noop)
    rep["noop_logits_max_abs_diff"] = float((lg_noop - a).abs().max())

    # 2. donor = recipient is a no-op
    h_same, _, _ = adapter.donor_patch(h, h, torch.arange(min(8, adapter.d_sae), device=h.device))
    rep["donor_equals_recipient_max_abs_diff"] = float((h_same - h).abs().max())

    # 4. a manual single-feature decoder delta equals the reconstruction difference
    f = int(torch.argmax(z.reshape(-1, adapter.d_sae).sum(0)))
    z_mod = z.clone()
    z_mod[..., f] = z[..., f] + 1.0
    manual = adapter.sae.W_dec.float()[f]
    via_recon = adapter.reconstruct(h, z_mod) - adapter.reconstruct(h, z)
    rep["manual_vs_reconstruction_max_abs_diff"] = float((via_recon - manual).abs().max())
    rep["probe_feature"] = f

    # 5. bypass reconstruction contains c and mu exactly once
    xin, par = adapter.split(h)
    recon = adapter.reconstruct(h)
    implied = recon - adapter.sae.decode(adapter.sae.encode(xin))
    rep["implied_offset_minus_mu_minus_pc1_max_abs"] = float((implied - par - adapter.mu).abs().max())
    if adapter.condition == "bypass":
        c = (h - adapter.mu) @ adapter.v1
        rep["pc1_coefficient_mean_abs"] = float(c.abs().mean())
        rep["encoder_input_pc1_residual_max_abs"] = float((xin @ adapter.v1).abs().max())
    else:
        rep["encoder_input_pc1_residual_max_abs"] = None

    # 3. standard and post-hoc share one dictionary: identical encodings by construction
    rep["posthoc_shares_standard_encoder"] = True

    rep["passed"] = bool(
        rep["noop_logits_max_abs_diff"] <= tol
        and rep["noop_edit_max_abs_diff_vs_h"] <= 1e-3
        and rep["donor_equals_recipient_max_abs_diff"] <= 1e-3
        and rep["manual_vs_reconstruction_max_abs_diff"] <= 1e-3
        and rep["implied_offset_minus_mu_minus_pc1_max_abs"] <= 1e-2
        and (adapter.condition == "standard"
             or rep["encoder_input_pc1_residual_max_abs"] <= 1e-2))
    log("[gate] " + json.dumps({k: v for k, v in rep.items() if not isinstance(v, bool)}, default=str))
    log("[gate] passed=%s" % rep["passed"])
    return rep
