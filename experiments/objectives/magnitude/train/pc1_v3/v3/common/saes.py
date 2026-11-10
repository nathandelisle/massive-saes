"""SAE architectures and the shared trainer.

All architectures share the v2 App G protocol: tied init (W_enc = W_dec^T),
decoder rows unit-normalized after every step, Adam 3e-4, batch 4096, 20k steps,
MSE on centered activations. The architecture-law experiment (exp_b1) rests on
the objective distinction:

  support-only sparsity  (TopK, BatchTopK, JumpReLU's L0)  -> tiling symmetry intact
  magnitude sparsity     (ReLU+L1, Gated's L1)             -> symmetry broken

REGISTERED PREDICTIONS (state in the paper before running):
  JumpReLU  : inherits the inversion (L0 does not price off-axis cancellation).
  Gated     : no inversion (L1 on the gate path prices it).
  Matryoshka: uncertain — nested-prefix reconstruction may force PC1 into a small
              consistent prefix. Stubbed below; run only after B1 resolves.

NOTE on JumpReLU/Gated fidelity: the straight-through kernel (JumpReLU) and the
frozen-decoder auxiliary loss (Gated) follow Rajamanoharan et al. 2024; verify
against a reference implementation (e.g. the released Gemma Scope training code
/ dictionary_learning repo) before headline runs — kernel choice can shift L0
calibration even though it should not affect the identifiability conclusion.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- 
# activation functions
# -----------------------------------------------------------------------------

def _rect(x):
    """Rectangle kernel K(z) = H(z + 1/2) - H(z - 1/2) (Rajamanoharan et al.
    2024, Eq. 11-12)."""
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLUSTE(torch.autograd.Function):
    """y = x * H(x - theta).  arXiv:2407.14435 App. J / the Gemma Scope colab:
        dy/dx     = H(x - theta)                      <- EXACT, not an STE
        dy/dtheta = -(theta/bw) * rect((x - theta)/bw)
    """

    @staticmethod
    def forward(ctx, x, theta, bandwidth):
        ctx.save_for_backward(x, theta)
        ctx.bw = bandwidth
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        x_grad = (x > theta).to(g.dtype) * g
        t_grad = -(theta / ctx.bw) * _rect((x - theta) / ctx.bw) * g
        return x_grad, t_grad.sum(dim=0), None      # reference sums over batch


class _StepSTE(torch.autograd.Function):
    """y = H(x - theta), used ONLY for the L0 term.
        dy/dx     = 0     <- the sparsity penalty must NOT reach the encoder
        dy/dtheta = -(1/bw) * rect((x - theta)/bw)

    Returning a nonzero x-gradient here (as an earlier version did) injects
    l0_coef * rect/bw into W_enc — a 1/bandwidth = 1000x amplification that
    corrupts the dictionary, which is precisely the failure this module's
    docstring assumed was impossible.
    """

    @staticmethod
    def forward(ctx, x, theta, bandwidth):
        ctx.save_for_backward(x, theta)
        ctx.bw = bandwidth
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        t_grad = -(1.0 / ctx.bw) * _rect((x - theta) / ctx.bw) * g
        return torch.zeros_like(x), t_grad.sum(dim=0), None


def topk_mask(z, k):
    vals, idx = torch.topk(z, k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, vals)
    return out


def batch_topk_mask(z, k):
    """BatchTopK (Bussmann et al. 2024, arXiv:2412.06410): keep the B*k largest
    activations across the whole batch rather than k per row.

    Index scatter, matching bartbussmann/BatchTopK, SAELens batchtopk_sae.py
    and dictionary_learning trainers/batch_top_k.py. The previous
    `z * (z >= thresh)` was wrong twice: ties at the threshold ALL survive
    (measured: 8 kept where 2 were budgeted), and because z is post-ReLU with
    many exact zeros, a batch with fewer than B*k strictly-positive entries
    gives thresh == 0, making `z >= 0` a dense passthrough.
    """
    flat = z.reshape(-1)
    n_keep = min(z.shape[0] * k, flat.numel())
    vals, idx = torch.topk(flat, n_keep, sorted=False)
    return torch.zeros_like(flat).scatter_(-1, idx, vals).reshape(z.shape)


# -----------------------------------------------------------------------------
# architectures
# -----------------------------------------------------------------------------

class BaseSAE(nn.Module):
    arch = "base"

    def __init__(self, d_model, d_sae, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_sae, d_model, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())            # [d_sae, d]
        self.W_enc = nn.Parameter(W.clone())            # tied init: enc row i = dec row i
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.d_model, self.d_sae = d_model, d_sae

    def pre(self, x):
        # Gao et al. 2024 pre-bias, as in v2 (`self.encoder(x - self.decoder.bias)`).
        # Omitting it changes the learned dictionary, so it is protocol, not taste.
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def encode(self, x):
        raise NotImplementedError

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def sparsity_loss(self, z, pre):
        return torch.zeros((), device=z.device)

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


class TopKSAE(BaseSAE):
    arch = "topk"

    def __init__(self, d_model, d_sae, k=64, seed=0):
        super().__init__(d_model, d_sae, seed)
        self.k = k

    def encode(self, x):
        return topk_mask(F.relu(self.pre(x)), self.k)


class BatchTopKSAE(TopKSAE):
    """BatchTopK with the paper's inference-time threshold.

    Without it, `splice` encodes chunk*seq_len rows at once and a feature's
    activation depends on which other TOKENS share its batch — so a per-feature
    Delta-CE is not a property of the feature and shifts with --chunk. The
    paper uses theta = E_X[min{z_ij | z_ij > 0}] with a JumpReLU at inference.
    """
    arch = "batchtopk"

    def __init__(self, d_model, d_sae, k=64, seed=0, threshold_beta=0.999):
        super().__init__(d_model, d_sae, k=k, seed=seed)
        self.threshold_beta = threshold_beta
        self.register_buffer("threshold", torch.tensor(-1.0))

    def encode(self, x):
        pre = F.relu(self.pre(x))
        if not self.training and float(self.threshold) >= 0:
            return pre * (pre > self.threshold)
        z = batch_topk_mask(pre, self.k)
        if self.training:
            with torch.no_grad():
                act = z[z > 0]
                m = act.min() if act.numel() else torch.zeros((), device=z.device)
                self.threshold.copy_(m if float(self.threshold) < 0 else
                                     self.threshold_beta * self.threshold +
                                     (1 - self.threshold_beta) * m)
        return z


class ReluL1SAE(BaseSAE):
    arch = "relu_l1"

    def __init__(self, d_model, d_sae, l1_coef=1e-3, seed=0):
        super().__init__(d_model, d_sae, seed)
        self.l1_coef = l1_coef

    def encode(self, x):
        return F.relu(self.pre(x))

    def sparsity_loss(self, z, pre):
        # decoder rows are unit-norm, so ||z||_1 == ||z * ||d_i|| ||_1
        return self.l1_coef * z.abs().sum(-1).mean()


class TopKL1SAE(TopKSAE):
    """TopK support constraint + beta * L1 magnitude penalty.
    The clean causal knob for exp_b2: support held fixed at k while the
    magnitude penalty is dialed from 0 (pure TopK, inversion predicted) up
    (symmetry broken, inversion predicted to vanish)."""
    arch = "topk_l1"

    def __init__(self, d_model, d_sae, k=64, beta=0.0, seed=0):
        super().__init__(d_model, d_sae, k=k, seed=seed)
        self.beta = beta

    def sparsity_loss(self, z, pre):
        return self.beta * z.abs().sum(-1).mean()


class JumpReLUSAE(BaseSAE):
    """JumpReLU with DATA-DEPENDENT theta and bandwidth.

    Neither theta nor the bandwidth is scale-free. Rajamanoharan et al. 2024
    App. I and Gemma Scope App. A both renormalise activations to E||x||^2 = 1
    and only then use theta = eps = 1e-3. This suite trains on RAW centered
    residual activations, where a hardcoded theta = e^-4 sits orders of
    magnitude below the pre-activation scale.

    Rescaling the input alone is not enough at this step budget (measured):
    with log-theta trained by Adam, |dlog theta| <= ~lr per step, so theta can
    move at most exp(3e-4 * 20000) = 403x over a full run and 3.3x over a 4k
    calibration probe. Gemma Scope gets away with the 1e-3 init only because it
    trains ~50x longer. Left uncalibrated, theta moved 1.00x, gated 0.17% of
    positive pre-activations, and L0 moved only 2.4x across FIVE decades of
    l0_coef — i.e. the row labelled "JumpReLU" was really a ReLU SAE, and the
    architecture-law test would have been vacuous.

    So: set theta from the pre-activation quantile giving `target_l0` active
    features, and set the bandwidth to the same scale. Widening the bandwidth
    for a short run has direct precedent in DeepMind's official notebook ("We
    use a higher bandwidth than in the paper to ensure a non-zero gradient to
    the threshold at every step").
    """

    arch = "jumprelu"

    def __init__(self, d_model, d_sae, l0_coef=1e-3, bandwidth=None,
                 theta_init=None, target_l0=64, bw_frac=1.0, seed=0):
        super().__init__(d_model, d_sae, seed)
        self.log_theta = nn.Parameter(
            torch.full((d_sae,), math.log(theta_init or 1e-3)))
        self.l0_coef = l0_coef
        self.target_l0 = target_l0
        self.bw_frac = bw_frac
        # buffers so they ride in state_dict and survive save_sae/load_sae
        self.register_buffer("bandwidth_t",
                             torch.tensor(float(bandwidth or 1e-3)))
        self.register_buffer("scale_set",
                             torch.tensor(bandwidth is not None
                                          and theta_init is not None))
        self._pre = None

    @property
    def bandwidth(self):
        return float(self.bandwidth_t)

    @torch.no_grad()
    def calibrate_from_batch(self, x):
        """One-shot data-dependent init of theta AND bandwidth. Idempotent."""
        if bool(self.scale_set):
            return
        pre = F.relu(self.pre(x)).flatten().float()
        if pre.numel() > 1_000_000:          # torch.quantile has a size limit
            pre = pre[torch.randint(0, pre.numel(), (1_000_000,),
                                    device=pre.device)]
        pos_frac = float((pre > 0).float().mean())
        want = float(self.target_l0) / self.d_sae
        # The quantile is over ALL entries, ~half of which are exact zeros
        # after the ReLU. If the requested active fraction exceeds the positive
        # fraction, the quantile lands on 0, theta collapses to the floor, the
        # rect window has zero width, theta receives no gradient, and the
        # architecture silently degenerates into a plain ReLU SAE -- precisely
        # the failure this calibration exists to prevent. Refuse instead.
        if want >= pos_frac:
            raise RuntimeError(
                f"jumprelu calibration impossible: target_l0={self.target_l0} "
                f"of d_sae={self.d_sae} needs {want:.3f} of units active, but "
                f"only {pos_frac:.3f} of pre-activations are positive. Lower "
                f"target_l0 or widen the encoder init.")
        q = min(max(1.0 - want, 0.0), 1.0 - 1e-6)
        th = float(torch.quantile(pre, q))
        if not (th > 0) or not math.isfinite(th):
            raise RuntimeError(
                f"jumprelu calibration produced theta={th}; the threshold "
                f"would not gate anything and L0 could not be calibrated.")
        self.log_theta.data.fill_(math.log(th))
        self.bandwidth_t.fill_(th * self.bw_frac)
        self.scale_set.fill_(True)

    @torch.no_grad()
    def recalibrate(self, x):
        """Re-seat theta on the CURRENT pre-activation scale.

        calibrate_from_batch fires once, at step 0, against a random encoder.
        As W_enc trains, the pre-activation distribution moves, and log-theta
        cannot follow it: Adam bounds |dlog theta| by ~lr per step, so theta is
        stranded wherever the initial scale put it. Measured consequence
        (pythia-1.4b, d_sae 8192, target L0 64): L0 settled at 12.9 and moved
        only 12.9 -> 12.1 across EIGHT decades of l0_coef -- the sparsity
        coefficient was inert and L0=64 was unreachable, so the architecture
        could not be sparsity-matched to the other rows.

        Re-seating theta periodically lets it track that scale. Measured on
        synthetic PC1-dominated data (target L0 32):

            schedule                L0 @ l0_coef=1e-4   @ 1e2
            none                          2.0            1.1
            warmup (first 50%)           21.9           19.8
            every 250, throughout        28.2           25.7
            every 100, throughout        30.9           30.1

        REPORT THIS IN THE PAPER: with the threshold on a schedule, the L0
        PENALTY has little additional effect (30.9 -> 30.1 across six decades),
        so sparsity here is set by the threshold schedule, not by the penalty.
        The row therefore tests support-only GATING -- which is what the
        architecture-law prediction is about, since nothing in the objective
        prices off-axis cancellation either way -- and NOT the L0 penalty's
        optimisation dynamics. Call it "JumpReLU (scheduled threshold)".
        The upside is that sparsity is pinned, so the architecture comparison
        is not confounded by an L0 mismatch.
        """
        self.scale_set.fill_(False)
        self.calibrate_from_batch(x)

    def encode(self, x):
        # App. J: ReLU first, so theta < eps/2 cannot let negative
        # pre-activations bias the backward pass.
        pre = F.relu(self.pre(x))
        self._pre = pre
        return _JumpReLUSTE.apply(pre, self.log_theta.exp(), self.bandwidth)

    def sparsity_loss(self, z, pre):
        # Step applies to the PRE-ACTIVATIONS (official colab + SAELens) and
        # its x-gradient is exactly zero, so this term moves theta and nothing
        # else.
        return self.l0_coef * _StepSTE.apply(
            self._pre, self.log_theta.exp(), self.bandwidth).sum(-1).mean()


class GatedSAE(BaseSAE):
    arch = "gated"

    def __init__(self, d_model, d_sae, l1_coef=1e-3, seed=0):
        super().__init__(d_model, d_sae, seed)
        self.r_mag = nn.Parameter(torch.zeros(d_sae))   # W_mag = W_enc * exp(r)
        self.b_gate = nn.Parameter(torch.zeros(d_sae))
        self.l1_coef = l1_coef
        self._pi_gate = None

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc.T
        pi_gate = pre + self.b_gate
        mag = F.relu(pre * self.r_mag.exp() + self.b_enc)
        self._pi_gate = pi_gate
        return (pi_gate > 0).to(x.dtype) * mag

    def sparsity_loss(self, z, pre):
        return self.l1_coef * F.relu(self._pi_gate).sum(-1).mean()

    def aux_loss(self, x):
        # reconstruct from relu(pi_gate) through a frozen copy of the decoder
        z_gate = F.relu(self._pi_gate)
        recon = z_gate @ self.W_dec.detach() + self.b_dec.detach()
        return F.mse_loss(recon, x, reduction="none").sum(-1).mean()


class MatryoshkaSAE(BaseSAE):
    """SKIPPED THIS ROUND — conditional on exp_b1 (see README, 'Skipped').
    Registered prediction: nested-prefix reconstruction losses force the
    dominant direction into the smallest prefix, plausibly concentrating PC1
    into few, cross-seed-consistent atoms — i.e. partial identifiability from
    an architectural prior rather than a magnitude penalty."""
    arch = "matryoshka"

    def __init__(self, *a, **kw):
        raise NotImplementedError("Conditional on exp_b1 results; see README.")


ARCHS = {
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
    "relu_l1": ReluL1SAE,
    "topk_l1": TopKL1SAE,
    "jumprelu": JumpReLUSAE,
    "gated": GatedSAE,
}


# -----------------------------------------------------------------------------
# trainer
# -----------------------------------------------------------------------------

def train_sae(batches, d_model, d_sae, arch="topk", seed=0, lr=3e-4,
              device="cuda", log_every=1000, verbose=True, tag="",
              recal_every=100, recal_until_frac=1.0, total_steps=0,
              **arch_kwargs):
    """batches: iterator of centered float32 [B, d] tensors (see data.activation_batches).
    Returns (sae, log_dict). Deterministic given seed + data order.

    `verbose` prints progress. A 20k-step run is many minutes of total silence
    otherwise, which makes a hang indistinguishable from healthy work in an
    unattended overnight queue — and a NaN loss indistinguishable from either.
    """
    import time
    torch.manual_seed(seed)
    sae = ARCHS[arch](d_model, d_sae, seed=seed, **arch_kwargs).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    log = []
    t0 = time.time()
    sae.train()                      # BatchTopK needs train mode to fit theta
    for step, x in enumerate(batches):
        if step == 0 and hasattr(sae, "calibrate_from_batch"):
            sae.calibrate_from_batch(x)     # JumpReLU data-dependent scale
        elif (recal_every and hasattr(sae, "recalibrate")
              and step <= recal_until_frac * max(total_steps, 1)
              and step % recal_every == 0):
            sae.recalibrate(x)              # track the drifting pre-act scale
        recon, z = sae(x)
        mse = F.mse_loss(recon, x, reduction="none").sum(-1).mean()
        loss = mse + sae.sparsity_loss(z, None)
        if isinstance(sae, GatedSAE):
            loss = loss + sae.aux_loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_decoder()
        if step % log_every == 0 or step < 5:
            with torch.no_grad():
                var = x.pow(2).sum(-1).mean()      # inputs are centered
                fvu = (mse / var).item()
                l0 = (z != 0).float().sum(-1).mean().item()
            rec = dict(step=step, mse=mse.item(), fvu=fvu, l0=l0)
            log.append(rec)
            if verbose and step % log_every == 0:
                el = time.time() - t0
                print(f"      [sae {tag}] step {step:6d}  fvu={fvu:.4f}  "
                      f"L0={l0:.1f}  {el / 60:.1f}m", flush=True)
            if not torch.isfinite(mse):
                raise RuntimeError(
                    f"SAE loss went non-finite at step {step} (arch={arch}, "
                    f"tag={tag}) — training diverged; do not use this SAE")
    sae.eval()                       # freeze the BatchTopK inference threshold
    return sae, log


def save_sae(sae, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = dict(arch=sae.arch, d_model=sae.d_model, d_sae=sae.d_sae)
    # target_l0 must round-trip or a reloaded jumprelu silently reverts to 64
    for k in ("k", "l1_coef", "l0_coef", "beta", "bandwidth", "target_l0",
              "bw_frac", "threshold_beta"):
        if hasattr(sae, k):
            cfg[k] = getattr(sae, k)
    torch.save(dict(state=sae.state_dict(), cfg=cfg, extra=extra or {}), path)


def load_sae(path, device="cuda"):
    blob = torch.load(path, map_location=device)
    cfg = blob["cfg"]
    arch = cfg.pop("arch")
    d_model, d_sae = cfg.pop("d_model"), cfg.pop("d_sae")
    key_map = {"k": "k", "l1_coef": "l1_coef", "l0_coef": "l0_coef",
               "beta": "beta", "bandwidth": "bandwidth",
               "target_l0": "target_l0", "bw_frac": "bw_frac",
               "threshold_beta": "threshold_beta"}
    kwargs = {key_map[k]: v for k, v in cfg.items() if k in key_map}
    sae = ARCHS[arch](d_model, d_sae, **kwargs).to(device)
    sae.load_state_dict(blob["state"])
    sae.eval()
    return sae, blob.get("extra", {})


@torch.no_grad()
def decoder_np(sae):
    W = sae.W_dec.detach().float().cpu().numpy()
    import numpy as np
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def firing_rates(sae, act_dir, n_sample=100_000, batch=8192, device="cuda"):
    """Fraction of tokens each feature fires on, over the first n_sample
    training activations (v2 'alive' definition: rate > 0.1%)."""
    import numpy as np
    from .data import open_activations
    acts, mean = open_activations(act_dir)
    n = min(n_sample, acts.shape[0])
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    counts = torch.zeros(sae.d_sae, device=device)
    for i in range(0, n, batch):
        x = torch.tensor(np.asarray(acts[i:min(i + batch, n)], dtype=np.float32),
                         device=device) - mean_t
        z = sae.encode(x)
        counts += (z != 0).float().sum(0)
    return (counts / n).cpu().numpy()
