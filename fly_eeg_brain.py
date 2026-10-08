"""Train the fly brain ITSELF to denoise EEG.

The readout is FIXED before training and never fitted to data, so every dB of cleaning must come
from inside the connectome: per-neuron leak and bias, a positive gain on every synapse (wiring and
Dale signs stay fixed) and a gain per Johnston's-organ input neuron. Trained by BPTT through the
whole 2 s epoch on the train-split MSE against the clean EEG.

Fixed readouts (--readout):
  random  a fixed +-1 projection of the descending/motor neurons (standardised once with the
          UNTRAINED brain's output statistics = a unit conversion). Pure "brain does it".
  ridge0  the ridge head of the UNTRAINED brain (lambda >= --lam-min for robust weights), frozen;
          the brain then learns on top of it. Improvement over r0 is attributable to the brain.
References computed in the same run: noisy, FIR (65 taps + ridge), untrained brain + fixed readout,
untrained brain + ridge (r0), and optionally the shuffled-wiring brain trained identically (--shuffle).

Run from the nfly folder:  uv run --no-sync python ../fly-eeg/fly_eeg_brain.py --artifact eog --out DIR
"""
import os, argparse, json, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_eeg_denoise import (load_data, metrics, fmt, paired, make_model, jo_drive, Ridge, evaluate,
                             fir_features, CENTRAL, NFLY_DATA)
from nfly.connectome import load_malecns, build_connectome
from nfly.brain import InputDrive
from nfly.brain.rnn import ConnectomeRNN, Weights


class BrainPlus(ConnectomeRNN):
    """Neuron-intrinsic tweaks on top of the connectome RNN (wiring and Dale signs untouched):
      --syn    per-neuron synaptic filter: recurrent+sensory current is low-passed (rate gamma) before the
               membrane leak -> second-order, band-pass/resonant neurons instead of a single leak
      --adapt  per-neuron spike-frequency adaptation: a slow trace of the neuron's own activity (rate beta)
               is subtracted with a learnable strength -> high-pass / derivative-like responses
      --slope  per-neuron learnable input slope (gain of the transfer function)"""
    def init_plus(self, a):
        n, dev = self.n, self.w0.device; g = torch.Generator().manual_seed(2)
        def logu(lo, hi):
            v = torch.exp(torch.rand(n, generator=g) * (np.log(hi) - np.log(lo)) + np.log(lo)); return torch.log(v / (1 - v)).to(dev)
        self.plus = dict(syn=a.syn, adapt=a.adapt, slope=a.slope)
        if a.syn: self.syn_logit = torch.nn.Parameter(logu(0.1, 0.9))
        if a.adapt:
            self.adapt_logit = torch.nn.Parameter(logu(0.005, 0.05))
            self.adapt_log_g = torch.nn.Parameter(torch.full((n,), float(np.log(a.adapt_g)), device=dev))
        if a.slope: self.log_slope = torch.nn.Parameter(torch.zeros(n, device=dev))

    def plus_params(self):
        return [getattr(self, k) for k in ("syn_logit", "adapt_logit", "adapt_log_g", "log_slope") if hasattr(self, k)]

    def forward(self, drive=None, steps=None, batch=1, h0=None, record=None):
        assert h0 is None, "BrainPlus: truncated BPTT not supported"
        steps, batch = drive.steps, drive.batch; dev = self.w0.device; P = self.plus
        h = torch.zeros(batch, self.n, device=dev); s = torch.zeros_like(h) if P["syn"] else None; q = torch.zeros_like(h) if P["adapt"] else None
        W = self.weights(); gam = torch.sigmoid(self.syn_logit) if P["syn"] else None
        bet = torch.sigmoid(self.adapt_logit) if P["adapt"] else None; ga = self.adapt_log_g.exp() if P["adapt"] else None
        nu = self.log_slope.exp() if P["slope"] else None; idx = drive.idx.to(dev)
        hist = [h if record is None else h[:, record]]
        for t in range(steps):
            u = torch.zeros(batch, self.n, device=dev); u[:, idx] = drive.drive[:, t].to(dev)
            r = self.recurrent_input(h, W.w) + u
            if s is not None: s = (1 - gam) * s + gam * r; r = s
            x = r + self.bias
            if nu is not None: x = x * nu
            if q is not None: x = x - ga * q
            x = self.act(x)
            if self.h_max is not None: x = x.clamp(max=self.h_max)
            h = (1 - W.alpha) * h + W.alpha * x
            if q is not None: q = (1 - bet) * q + bet * h
            hist.append(h if record is None else h[:, record])
        return torch.stack(hist, dim=1)


class BrainRewire(ConnectomeRNN):
    """The fly connectome as the STARTING wiring, opened up: (1) `--new-edges N` trainable synapses added (half from the
    Johnston's-organ input neurons straight to the readout neurons, half between random central neurons), starting at
    zero strength with free sign; (2) `--free-signs`: every existing synapse may flip excitatory/inhibitory."""
    def init_rewire(self, a, in_idx, out_idx):
        dev = self.w0.device; g = torch.Generator().manual_seed(7); self.n_old = self.pre.numel(); N = a.new_edges
        if N > 0:
            h = N // 2
            pre1 = in_idx[torch.randint(len(in_idx), (h,), generator=g)]; post1 = out_idx[torch.randint(len(out_idx), (h,), generator=g)]
            pre2 = torch.randint(self.n, (N - h,), generator=g); post2 = torch.randint(self.n, (N - h,), generator=g)
            self.pre = torch.cat([self.pre, torch.cat([pre1.cpu(), pre2]).to(dev)]); self.post = torch.cat([self.post, torch.cat([post1.cpu(), post2]).to(dev)])
        self.w_new = torch.nn.Parameter(torch.zeros(N, device=dev))
        self.register_buffer("w_scale", self.w0.abs().median().reshape(()))   # new synapses learn in units of a typical synapse
        print(f"  new-synapse unit = median |w0| = {float(self.w_scale):.2e}", flush=True)
        self.free_signs = a.free_signs
        if a.free_signs: self.sg = torch.nn.Parameter(self.sign.float().clone())
        self._cached = None

    def rewire_params(self):
        return [self.w_new] + ([self.sg] if self.free_signs else [])

    def edge_weights(self):
        s = self.sg if self.free_signs else self.sign
        return torch.cat([s * self.w0 * torch.exp(self.log_gain), self.w_new * self.w_scale])

    def weights(self):
        if torch.is_grad_enabled():
            return Weights(self.edge_weights(), self.alpha())
        key = (self.log_gain._version, self.w_new._version, self.alpha_logit._version, self.sg._version if self.free_signs else 0)
        if self._cached is None or self._cached[0] != key:
            self._cached = (key, Weights(self.edge_weights(), self.alpha()))
        return self._cached[1]


class BrainDenoiser(torch.nn.Module):
    """Brain + fixed linear readout. out(t) = (h_out(t) . w_out + b_out - mu0) / sd0, with the state
    aligned `lag` samples after the input (a fixed output delay of lag/256 s)."""
    def __init__(self, model, in_idx, out_idx, a):
        super().__init__()
        self.model, self.a = model, a
        self.register_buffer("in_idx", in_idx); self.register_buffer("out_idx", out_idx)
        self.in_gain = torch.nn.Parameter(torch.full((len(in_idx),), float(a.gain)))
        self.register_buffer("w_out", torch.zeros(len(out_idx))); self.register_buffer("b_out", torch.zeros(()))
        self.register_buffer("mu0", torch.zeros(())); self.register_buffer("sd0", torch.ones(()))
        self.out_scale = torch.nn.Parameter(torch.tensor(0.1 if getattr(a, "residual", False) else 1.0))
        self.out_bias = torch.nn.Parameter(torch.zeros(()))
        self.w_train = torch.nn.Parameter(torch.zeros(len(out_idx)))

    def states(self, y, idx=None):
        """Recorded activity of neurons `idx` (default: the readout wire's neurons), aligned to the input.
        With --tbptt K the unroll is split into K-step chunks and the state is detached between chunks
        (truncated BPTT: shorter, better-conditioned gradient paths; the forward dynamics are unchanged)."""
        a, T = self.a, y.shape[1]
        idx = self.out_idx if idx is None else idx
        if torch.is_tensor(y):   # differentiable path (multi-step training): torch reflect padding
            yb = torch.nn.functional.pad(y.to(self.w_out.device)[:, None, :], (a.warm, a.lag), mode="reflect")[:, 0, :]
        else:
            yb = torch.as_tensor(np.pad(y, ((0, 0), (a.warm, a.lag)), mode="reflect"), device=self.w_out.device)
        if getattr(a, "filterbank", 0) > 0:
            # cochlea-like periphery: JO neuron i gets band (i mod K) of the signal (log-spaced 1-80 Hz FFT bands),
            # delayed by ((i // K) mod (D+1)) samples; +/- halves as before. Brain itself unchanged.
            n_in = len(self.in_idx); half = n_in // 2; K = a.filterbank; D = max(0, getattr(a, "jo_delays", 0))
            Tt = yb.shape[1]; f = torch.fft.rfftfreq(Tt, d=1 / 256.0).to(yb.device); edges = np.geomspace(1.0, 80.0, K + 1)
            Y = torch.fft.rfft(yb, dim=1)
            bands = torch.stack([torch.fft.irfft(Y * ((f >= edges[k]) & (f < edges[k + 1])).to(Y.dtype), n=Tt, dim=1) for k in range(K)], -1)  # (B,T,K)
            bands = bands / (bands.std(1, keepdim=True) + 1e-6) * yb.std(1, keepdim=True)[..., None]
            i = torch.arange(n_in, device=yb.device); bi = i % K; di = (i // K) % (D + 1)
            yd = torch.gather(torch.stack([torch.roll(bands, d, dims=1) for d in range(D + 1)], -1).flatten(2), 2,
                              (bi * (D + 1) + di).view(1, 1, -1).expand(yb.shape[0], Tt, -1))
            drive = torch.cat([torch.relu(yd[:, :, :half]), torch.relu(-yd[:, :, half:])], -1) * self.in_gain
        elif getattr(a, "jo_delays", 0) > 0:
            # sensory periphery: JO neuron i sees the signal delayed by (i mod (D+1)) samples (a fixed
            # delay-line encoding, like frequency/phase-tuned JO subgroups); brain and wire stay fixed
            n_in = len(self.in_idx); half = n_in // 2
            dvec = (torch.arange(n_in, device=yb.device) % (a.jo_delays + 1))
            span = getattr(a, "jo_span", 0)
            if span and span > a.jo_delays:   # D+1 taps spread geometrically over 0..span samples, zero-filled shifts (no wrap)
                m = a.jo_delays
                while True:                                   # more geometric points until D+1 distinct integer taps
                    taps = np.unique(np.round(np.concatenate([[0], np.geomspace(1, span, m)])).astype(int))
                    if len(taps) >= a.jo_delays + 1: break
                    m += 1
                taps = taps[np.round(np.linspace(0, len(taps) - 1, a.jo_delays + 1)).astype(int)]
                Tt = yb.shape[1]
                shifted = torch.stack([torch.nn.functional.pad(yb, (int(d), 0))[:, :Tt] for d in taps], -1)
            else:
                shifted = torch.stack([torch.roll(yb, int(d), dims=1) for d in range(a.jo_delays + 1)], -1)  # (B, T, D+1)
            yd = shifted[:, :, dvec]                                                                     # (B, T, n_in)
            drive = torch.cat([torch.relu(yd[:, :, :half]), torch.relu(-yd[:, :, half:])], -1) * self.in_gain
        else:
            drive = jo_drive(yb, self.in_idx, 1.0) * self.in_gain
        if getattr(a, "envelope", False):
            # burst cue: a quarter of the JO neurons get the >30 Hz amplitude envelope (smoothed |high-pass|), delayed like the others
            n_in = len(self.in_idx); q = n_in // 4; Tt = yb.shape[1]; D = max(0, getattr(a, "jo_delays", 0))
            f = torch.fft.rfftfreq(Tt, d=1 / 256.0).to(yb.device)
            hp = torch.fft.irfft(torch.fft.rfft(yb, dim=1) * (f >= 30).to(yb.dtype), n=Tt, dim=1)
            env = torch.nn.functional.avg_pool1d(hp.abs()[:, None, :], 9, stride=1, padding=4)[:, 0, :]
            env = env / (yb.std(1, keepdim=True) + 1e-6)
            j = torch.arange(n_in - q, n_in, device=yb.device); dj = (j % (D + 1))
            E = torch.stack([torch.roll(env, int(d), dims=1) for d in range(D + 1)], -1)[:, :, dj]       # (B, T, q)
            drive = torch.cat([drive[:, :, : n_in - q], E * self.in_gain[n_in - q:]], -1)
        if not a.tbptt or not torch.is_grad_enabled():
            h = self.model(InputDrive(self.in_idx, drive), record=idx)[:, 1:]
        else:
            outs, h0 = [], None
            for s0 in range(0, drive.shape[1], a.tbptt):
                hist = self.model(InputDrive(self.in_idx, drive[:, s0:s0 + a.tbptt]), h0=h0, record=None)   # (B, k+1, N)
                outs.append(hist[:, 1:, idx]); h0 = hist[:, -1].detach()
            h = torch.cat(outs, 1)
        return h[:, a.warm + a.lag:a.warm + a.lag + T]                       # (B, T, K)

    def readout(self, h):
        w = self.w_out + (self.w_train if getattr(self.a, "train_readout", False) else 0)
        return (h @ w + self.b_out - self.mu0) / self.sd0

    def forward(self, y):
        a = self.a
        if getattr(a, "bidir", False):          # same fly brain run forward and on the time-reversed signal: past AND future context
            if torch.is_tensor(y): yy = torch.cat([y, torch.flip(y, dims=[1])], 0)
            else: yy = np.concatenate([y, y[:, ::-1]], 0)
            hh = self.states(yy); B = y.shape[0]                 # one batched unroll for both directions
            h = 0.5 * (hh[:B] + torch.flip(hh[B:], dims=[1]))
        else:
            h = self.states(y)
        out = self.readout(h)
        if getattr(a, "out_affine", False) or getattr(a, "residual", False):
            out = self.out_scale * out + self.out_bias
        if getattr(a, "residual", False):       # brain estimates the ARTIFACT: clean = noisy - brain output
            out = (y.to(out.device, out.dtype) if torch.is_tensor(y) else torch.as_tensor(y, device=out.device, dtype=out.dtype)) - out
        return out   # (B, T)

    def brain_params(self):
        ps = [self.model.alpha_logit, self.model.bias, self.in_gain]
        if self.a.train_edges: ps.append(self.model.log_gain)
        if isinstance(self.model, BrainPlus): ps += self.model.plus_params()
        if getattr(self.a, "out_affine", False) or getattr(self.a, "residual", False): ps += [self.out_scale, self.out_bias]
        if getattr(self.a, "train_readout", False): ps += [self.w_train]
        if isinstance(self.model, BrainRewire): ps += self.model.rewire_params()
        return ps


@torch.no_grad()
def feedback_mask(model, in_idx):
    """True for every edge whose target sits in an earlier layer than its source (layer = hop distance from the JO input)."""
    pre, post = model.pre.cpu().numpy(), model.post.cpu().numpy(); d = np.full(model.alpha().numel(), -1); d[in_idx.cpu().numpy()] = 0; l = 0
    while True:
        nxt = np.unique(post[d[pre] == l]); nxt = nxt[d[nxt] < 0]
        if len(nxt) == 0: break
        l += 1; d[nxt] = l
    return torch.as_tensor((d[pre] >= 0) & (d[post] >= 0) & (d[post] < d[pre]))


def cut_feedback(bd):
    """Zero the feedback synapses (log_gain -> -1e4, exp = 0); called at init, after resume and after every optimizer step."""
    if not hasattr(bd, "fb_mask"):
        bd.fb_mask = feedback_mask(bd.model, bd.in_idx).to(bd.model.log_gain.device)
        print(f"cut-feedback: {int(bd.fb_mask.sum()):,} of {bd.fb_mask.numel():,} synapses removed ({bd.fb_mask.float().mean().item():.1%})", flush=True)
    with torch.no_grad(): bd.model.log_gain[bd.fb_mask] = -1e4


def predict(bd, y, batch):
    bd.eval(); K = max(1, getattr(bd.a, "unroll", 1))
    out = []
    for b in range(0, len(y), batch):
        yk = torch.as_tensor(y[b:b + batch], device=bd.w_out.device)
        for _ in range(K):                      # same K-step cleaning as in training
            sc = yk.std(1, keepdim=True) + 1e-6
            yk = bd(yk / sc) * sc
        out.append(yk.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def ridge_on_brain(bd, x, y, a):
    """Streaming ridge on the UNTRAINED brain's readout neurons (no raw-input skip feature)."""
    K = len(bd.dn_idx); r = Ridge(K, a.device)
    for split in ("tr", "va"):
        for b in range(0, len(y[split]), a.eval_batch):
            H = bd.states(y[split][b:b + a.eval_batch], bd.dn_idx)
            r.add(split, H.reshape(-1, K), torch.as_tensor(x[split][b:b + a.eval_batch], device=a.device).reshape(-1))
    return r


@torch.no_grad()
def set_fixed_readout(bd, x, y, a, rng):
    """Freeze the readout. Returns (name, reference metrics dict or None)."""
    dev = a.device; K = len(bd.out_idx)
    if a.readout == "random":
        bd.w_out.copy_(torch.as_tensor(rng.choice([-1.0, 1.0], size=K), device=dev) / np.sqrt(K))
        proj = torch.cat([bd.states(y["tr"][b:b + a.eval_batch]) @ bd.w_out for b in range(0, min(len(y["tr"]), 4 * a.eval_batch), a.eval_batch)])
        bd.mu0.fill_(proj.mean().item()); bd.sd0.fill_(proj.std().item() + 1e-6)
        print(f"fixed random +-1 readout over {K} neurons; untrained projection mean {bd.mu0.item():.4f} sd {bd.sd0.item():.4f}", flush=True)
        return None
    r = ridge_on_brain(bd, x, y, a)
    r.fit()                                                       # unrestricted: the true r0 reference
    K = len(bd.dn_idx)
    xhat0 = np.concatenate([r.predict(bd.states(y["te"][b:b + a.eval_batch], bd.dn_idx).reshape(-1, K)).reshape(-1, y["te"].shape[1]).float().cpu().numpy()
                            for b in range(0, len(y["te"]), a.eval_batch)])
    m0 = metrics(xhat0, x["te"], y["te"], a.snr0); print(f"{'untrained brain + ridge (r0)':30s} test {fmt(m0)}   (lambda {r.lam})", flush=True)
    r.fit(lams=tuple(l for l in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1) if l >= a.lam_min))   # robust head to freeze
    bd.w_out.copy_((r.w / r.sd).float()); bd.b_out.fill_(float(r.ybar - (r.mu / r.sd) @ r.w))
    print(f"frozen ridge0 head: lambda {r.lam}, |w| max {bd.w_out.abs().max().item():.3g}", flush=True)
    return {"r0": (xhat0, m0)}


def shuffled_connectome(conn):
    g = torch.Generator().manual_seed(0)
    return build_connectome(conn.neurons, conn.pre, conn.post[torch.randperm(conn.n_edges, generator=g)], conn.syn_count, conn.sign)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="eog", choices=["eog", "emg", "both", "ecg", "line"])
    p.add_argument("--protocol", default="whole", choices=["whole", "centered"])
    p.add_argument("--n-train", type=int, default=4000); p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--extra-train", type=int, default=0); p.add_argument("--aug", type=int, default=1)
    p.add_argument("--artifact-split", default="shared", choices=["shared", "disjoint"])
    p.add_argument("--data-dir", default=None, help="train/evaluate on an independent raw cache instead of EEGdenoiseNet (see cmp/ss2016_export.py)")
    p.add_argument("--readout", default="random", choices=["random", "ridge0"])
    p.add_argument("--lam-min", type=float, default=1e-2, help="ridge0: smallest lambda allowed for the frozen head")
    p.add_argument("--shuffle", action="store_true", help="train the shuffled-wiring control brain instead")
    p.add_argument("--no-train-edges", dest="train_edges", action="store_false", help="only leak/bias/input gain (no synapse gains)")
    p.add_argument("--passes", type=int, default=30); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--eval-batch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="leak / bias / input gain"); p.add_argument("--lr-edge", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=0, help="split each batch into chunks of this size and accumulate gradients (small GPUs; identical update)")
    p.add_argument("--train-len", type=int, default=0, help="train on random crops of this many samples (0 = full epoch); eval always on full epochs")
    p.add_argument("--tbptt", type=int, default=0, help="truncated-BPTT chunk length in samples (0 = full 600-step unroll)")
    p.add_argument("--wire", default="dn", choices=["dn", "all", "fastdn"], help="neurons under the fixed readout wire: descending/motor, all non-JO, or the fastest-leak half of the DN/motor set")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay on the synapse log-gains (pulls gains back to 1)")
    p.add_argument("--cosine", action="store_true", help="cosine learning-rate decay over all passes")
    p.add_argument("--lr-bias", type=float, default=None, help="separate learning rate for the neuron biases (default: --lr)")
    p.add_argument("--warmup", type=int, default=0, help="linear learning-rate warm-up over this many steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rho", type=float, default=0.9); p.add_argument("--alpha-min", type=float, default=0.02); p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--gain", type=float, default=1.0); p.add_argument("--bias", type=float, default=0.02); p.add_argument("--h-max", type=float, default=10.0)
    p.add_argument("--jo-delays", type=int, default=0, help="JO delay-line encoding: neuron i gets the input delayed by i mod (D+1) samples (0 = plain half-wave drive)")
    p.add_argument("--jo-span", type=int, default=0, help="with --jo-delays D: spread the D+1 delay taps geometrically over 0..SPAN samples (zero-filled), e.g. 256 = 1 s for delta waves")
    p.add_argument("--lag", type=int, default=24); p.add_argument("--warm", type=int, default=64)
    p.add_argument("--syn", action="store_true", help="BrainPlus: per-neuron synaptic filter (second-order neurons)")
    p.add_argument("--adapt", action="store_true", help="BrainPlus: per-neuron spike-frequency adaptation")
    p.add_argument("--adapt-g", type=float, default=0.3, help="initial adaptation strength")
    p.add_argument("--slope", action="store_true", help="BrainPlus: per-neuron learnable transfer slope")
    p.add_argument("--filterbank", type=int, default=0, help="K log-spaced 1-80 Hz bands feeding interleaved JO groups (with --jo-delays: band x delay)")
    p.add_argument("--bidir", action="store_true", help="run the fly brain forward and time-reversed, average the states (full-window context)")
    p.add_argument("--residual", action="store_true", help="brain estimates the artifact; output = noisy - brain (learned scale/offset)")
    p.add_argument("--out-affine", action="store_true", help="learnable output scale + offset on the fixed readout")
    p.add_argument("--train-readout", action="store_true", help="make the linear readout over the wire neurons trainable (added to the fixed one)")
    p.add_argument("--spec-loss", type=float, default=0.0, help="weight of the spectral (|FFT|) loss term")
    p.add_argument("--cc-loss", type=float, default=0.0, help="weight of the (1 - CC) loss term")
    p.add_argument("--snr-lo", type=float, default=-7.0, help="training/val SNR range low end (test stays -7..2)")
    p.add_argument("--snr-hi", type=float, default=2.0, help="training/val SNR range high end")
    p.add_argument("--unroll", type=int, default=1, help="diffusion-style K-step cleaning during training (loss on every step)")
    p.add_argument("--new-edges", type=int, default=0, help="add N trainable synapses (half JO->readout, half random central), zero-init, free sign")
    p.add_argument("--free-signs", action="store_true", help="let every existing synapse flip excitatory/inhibitory")
    p.add_argument("--envelope", action="store_true", help="burst cue: a quarter of JO neurons get the >30 Hz amplitude envelope")
    p.add_argument("--cut-feedback", action="store_true", help="remove every synapse that points back to an earlier layer (layer = hop distance from the JO input), kept at zero during training")
    p.add_argument("--no-ref-ridge", action="store_true", help="skip the untrained-brain ridge reference (random readout mode)")
    p.add_argument("--no-fir", action="store_true")
    p.add_argument("--val-max", type=int, default=0, help="validate each pass on a fixed random subset of this many val epochs")
    p.add_argument("--init", type=Path, default=None, help="warm start from a brain_params.pt (same architecture flags)")
    p.add_argument("--eval-only", action="store_true", help="with --init: evaluate the loaded brain on the test set and exit")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.snr0 = None
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    T0 = time.time()

    if a.data_dir:                                                     # independent dataset in raw-cache format ({tr,va,te}_{clean,noisy}.npy)
        x = {k: np.load(Path(a.data_dir) / f"{k}_clean.npy") for k in ("tr", "va", "te")}; y = {k: np.load(Path(a.data_dir) / f"{k}_noisy.npy") for k in x}; snr = None
        if a.n_train: x["tr"], y["tr"] = x["tr"][: a.n_train], y["tr"][: a.n_train]
        if a.n_test: x["te"], y["te"] = x["te"][: a.n_test], y["te"][: a.n_test]
    else:
        x, y, snr = load_data(a, np.random.default_rng(0))
    if a.val_max and len(x["va"]) > a.val_max:                         # cheaper per-pass validation (fixed subset, test set untouched)
        vi = np.random.default_rng(1).choice(len(x["va"]), a.val_max, replace=False); x["va"], y["va"] = x["va"][vi], y["va"][vi]
    snr0 = snr["te"] if snr else None; a.snr0 = snr0   # centered protocol: nominal input SNR per test epoch (SPAR-EEG score)
    print(f"EEGdenoiseNet {a.artifact} ({a.protocol}): train {len(x['tr'])} val {len(x['va'])} test {len(x['te'])} epochs of {x['tr'].shape[1]} samples", flush=True)
    res, xh = {"noisy": metrics(y["te"], x["te"], y["te"], snr0)}, {}
    print(f"{'noisy input':30s} test {fmt(res['noisy'])}", flush=True)
    if not a.no_fir:
        a.save_states, a.batch_eval_backup = None, a.batch; a.batch = a.eval_batch
        xh["fir"], res["fir"], _ = evaluate("linear FIR (65 taps, no brain)", lambda yb: fir_features(yb, a.device), x, y, a, snr0)
        a.batch = a.batch_eval_backup

    conn = load_malecns(NFLY_DATA)
    keep = conn.neurons.super_class.isin(CENTRAL) | conn.neurons.cell_type.str.startswith("JO", na=False)
    conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy())))
    if a.shuffle:
        conn = shuffled_connectome(conn); print("*** shuffled-wiring control ***", flush=True)
    ne = conn.neurons
    is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    in_idx = torch.as_tensor(np.flatnonzero(is_jo))
    dn_idx = torch.as_tensor(np.flatnonzero(ne.super_class.isin(["descending_neuron"]).to_numpy() | (ne.flow == "efferent").to_numpy()))
    out_idx = dn_idx if a.wire == "dn" else torch.as_tensor(np.flatnonzero(~is_jo))
    model = make_model(conn, a, a.device)
    if a.new_edges > 0 or a.free_signs:
        model.__class__ = BrainRewire; model.init_rewire(a, in_idx.to(a.device), out_idx.to(a.device))
        print(f"BrainRewire: +{a.new_edges} trainable synapses, free signs={a.free_signs}", flush=True)
    if a.syn or a.adapt or a.slope:
        model.__class__ = BrainPlus; model.init_plus(a); print(f"BrainPlus neuron tweaks: {model.plus}", flush=True)
    if a.wire == "fastdn":                                   # readout only from the fastest DN/motor neurons (short time constants -> sharper output)
        al = model.alpha().detach().cpu()[dn_idx]; out_idx = dn_idx[torch.argsort(al, descending=True)[: len(dn_idx) // 2]]
    print(f"{conn.summary()}\ninput: {len(in_idx)} JO neurons; fixed readout wire over {len(out_idx)} neurons ({a.wire}); ridge reference on {len(dn_idx)} DN/motor neurons", flush=True)

    for q in model.parameters(): q.requires_grad_(False)
    bd = BrainDenoiser(model, in_idx, out_idx, a).to(a.device)
    bd.register_buffer("dn_idx", dn_idx.to(a.device))
    if a.cut_feedback: cut_feedback(bd)
    ref = set_fixed_readout(bd, x, y, a, rng)
    if ref: xh["r0"], res["r0"] = ref["r0"]
    elif not a.no_ref_ridge:
        r = ridge_on_brain(bd, x, y, a).fit(); K = len(dn_idx)
        with torch.no_grad():
            xh["r0"] = np.concatenate([r.predict(bd.states(y["te"][b:b + a.eval_batch], dn_idx).reshape(-1, K)).reshape(-1, y["te"].shape[1]).float().cpu().numpy()
                                       for b in range(0, len(y["te"]), a.eval_batch)])
        res["r0"] = metrics(xh["r0"], x["te"], y["te"], snr0); print(f"{'untrained brain + ridge (r0)':30s} test {fmt(res['r0'])}   (lambda {r.lam})", flush=True)
    if a.init:   # warm start from a saved brain_params.pt (also restores its fixed readout w_out/mu0/sd0)
        sd = torch.load(a.init, map_location=a.device, weights_only=False)
        r_ = bd.load_state_dict(sd, strict=False); print(f"init from {a.init}: loaded {len(sd)} tensors, unexpected {r_.unexpected_keys}", flush=True)
    xh["untrained"] = predict(bd, y["te"], a.eval_batch); res["untrained"] = metrics(xh["untrained"], x["te"], y["te"], snr0)
    if a.eval_only:
        print(f"{'init brain (eval only)':30s} test {fmt(res['untrained'])}", flush=True)
        json.dump({k: {m: float(v.mean()) for m, v in r.items()} for k, r in res.items()}, open(a.out / "eval.json", "w"), indent=1); return
    print(f"{'untrained brain + fixed readout':30s} test {fmt(res['untrained'])}   ({time.time() - T0:.0f}s)", flush=True)

    # ---- train the brain ----
    params = bd.brain_params()
    for q in params: q.requires_grad_(True)
    groups = [{"params": [bd.model.alpha_logit, bd.in_gain], "lr": a.lr},
              {"params": [bd.model.bias], "lr": a.lr if a.lr_bias is None else a.lr_bias}]
    if isinstance(bd.model, BrainPlus): groups.append({"params": bd.model.plus_params(), "lr": a.lr})
    if a.train_edges: groups.append({"params": [bd.model.log_gain], "lr": a.lr_edge, "weight_decay": a.wd})
    if a.out_affine or a.residual: groups.append({"params": [bd.out_scale, bd.out_bias], "lr": a.lr})
    if a.train_readout: groups.append({"params": [bd.w_train], "lr": a.lr * 0.1})
    if isinstance(bd.model, BrainRewire): groups.append({"params": bd.model.rewire_params(), "lr": a.lr_edge})
    opt = torch.optim.Adam(groups)
    steps_per_pass = -(-len(x["tr"]) // a.batch)
    total = a.passes * steps_per_pass
    def _lr_mult(step):   # linear warm-up, then optional cosine decay to zero
        w = min(1.0, (step + 1) / a.warmup) if a.warmup else 1.0
        c = 0.5 * (1 + np.cos(np.pi * min(1.0, step / max(1, total)))) if a.cosine else 1.0
        return w * c
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult) if (a.cosine or a.warmup) else None
    n_par = sum(q.numel() for q in params)
    print(f"training {n_par:,} brain parameters (edges={a.train_edges}) for up to {a.passes} passes, batch {a.batch}", flush=True)
    ck = a.out / "ckpt.pt"; hist, start, best = [], 0, {"val_gain": -np.inf, "pass": -1, "state": None}
    if ck.exists():
        c = torch.load(ck, map_location=a.device, weights_only=False)
        bd.load_state_dict(c["state"], strict=False)                      # strict=False: older ckpts lack dn_idx
        if a.cut_feedback: cut_feedback(bd)
        try:
            opt.load_state_dict(c["opt"])
        except ValueError:                                                 # older ckpts: 2 groups [alpha,bias,in_gain],[log_gain] -> now 3 groups [alpha,in_gain],[bias],[log_gain]
            o = c["opt"]; st = o["state"]; remap = {0: 0, 1: 2, 2: 1, 3: 3}
            if len(o["param_groups"]) == 2 and set(st) <= set(remap):
                new_state = {remap[k]: v for k, v in st.items()}
                groups_new = opt.state_dict()["param_groups"]
                for g_new, g_old in zip(groups_new, [o["param_groups"][0], o["param_groups"][0], o["param_groups"][-1]]):
                    for key in ("betas", "eps", "weight_decay", "step" if "step" in g_old else "eps"):
                        if key in g_old: g_new[key] = g_old[key]
                opt.load_state_dict({"state": new_state, "param_groups": groups_new}); print("optimizer state remapped from the 2-group layout", flush=True)
            else:
                print("WARNING: optimizer state incompatible -- Adam moments reset", flush=True)
        hist, start, best = c["hist"], c["pass"] + 1, c["best"]
        if sched is not None and "sched" in c: sched.load_state_dict(c["sched"])
        print(f"resumed from pass {start} (best val gain {best['val_gain']:+.2f} dB @ pass {best['pass'] + 1})", flush=True)
    n, g = len(x["tr"]), torch.Generator().manual_seed(a.seed)
    mid, cm = a.out / "ckpt_mid.pt", None          # mid-pass checkpoint: 1 h debug lanes are shorter than one slow pass
    if mid.exists():
        cm = torch.load(mid, map_location=a.device, weights_only=False)
        if cm["pass"] != start: cm = None
        else:
            bd.load_state_dict(cm["state"], strict=False); opt.load_state_dict(cm["opt"])
            if sched is not None and "sched" in cm: sched.load_state_dict(cm["sched"])
            print(f"resumed mid-pass: pass {start + 1} step {cm['k']}", flush=True)
    for ps in range(start, a.passes):
        bd.train(); perm, tot, t0, k = torch.randperm(n, generator=g).numpy(), 0.0, time.time(), 0
        if cm is not None and ps == cm["pass"]:
            perm, tot, k = cm["perm"], cm["tot"], cm["k"]; g.set_state(cm["g"].cpu()); t0 = time.time() - cm["sec"]
        for i0 in range(k * a.batch, n, a.batch):
            i = np.sort(perm[i0:i0 + a.batch]); yb, xb = y["tr"][i], x["tr"][i]
            if a.train_len and a.train_len < yb.shape[1]:            # random crop (same offset for the batch): ~T/train_len x faster per step
                s0 = int(g.integers(0, yb.shape[1] - a.train_len + 1)) if hasattr(g, "integers") else int(torch.randint(0, yb.shape[1] - a.train_len + 1, (1,), generator=g))
                yb, xb = yb[:, s0:s0 + a.train_len], xb[:, s0:s0 + a.train_len]
            opt.zero_grad(set_to_none=True)
            mb = a.micro_batch if a.micro_batch and a.micro_batch < len(i) else len(i)   # gradient accumulation: same optimizer steps, less GPU memory
            xt, loss = torch.as_tensor(xb, device=a.device), 0.0
            for c0 in range(0, len(i), mb):
                xc = xt[c0:c0 + mb]
                if a.unroll > 1:   # diffusion-style: clean in K steps, loss on every step (deep supervision), renormalising between steps
                    yk = torch.as_tensor(yb[c0:c0 + mb], device=a.device); lsteps = 0
                    for k in range(a.unroll):
                        sc = yk.detach().std(1, keepdim=True) + 1e-6
                        yk = bd(yk / sc) * sc
                        lsteps = lsteps + ((yk - xc) ** 2).sum() / xt.numel()
                    pr = yk; lc_extra = lsteps / a.unroll
                else:
                    pr, lc_extra = bd(yb[c0:c0 + mb]), None
                lc = ((pr - xc) ** 2).sum() / xt.numel() if lc_extra is None else lc_extra
                if a.spec_loss > 0:   # spectral term (RRMSE_s-like, on |FFT|)
                    P, X = torch.fft.rfft(pr, dim=1).abs(), torch.fft.rfft(xc, dim=1).abs()
                    lc = lc + a.spec_loss * (((P - X) ** 2).mean(1) / ((X ** 2).mean(1) + 1e-6)).sum() / len(xt)
                if a.cc_loss > 0:     # 1 - correlation with the clean EEG
                    pc, xcc = pr - pr.mean(1, keepdim=True), xc - xc.mean(1, keepdim=True)
                    cc = (pc * xcc).sum(1) / (pc.norm(dim=1) * xcc.norm(dim=1) + 1e-6)
                    lc = lc + a.cc_loss * (1 - cc).sum() / len(xt)
                lc.backward(); loss = loss + lc.detach()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip).item(); opt.step()
            if a.cut_feedback: cut_feedback(bd)
            if sched is not None: sched.step()
            tot += loss.item() * len(i); k += 1
            if k % 10 == 1: print(f"  pass {ps + 1} step {k}/{-(-n // a.batch)} loss {loss.item():.4f} gradnorm {gn:.3g} {(time.time() - t0) / k:.2f}s/step", flush=True)
            if k % 50 == 0:
                torch.save({"state": bd.state_dict(), "opt": opt.state_dict(), "pass": ps, "k": k, "tot": tot, "perm": perm, "g": g.get_state(),
                            "sec": time.time() - t0, **({"sched": sched.state_dict()} if sched is not None else {})}, str(mid) + ".tmp")
                os.replace(str(mid) + ".tmp", mid)
            if k % 25 == 0 and (a.out / "STOP").exists(): break               # early stop requested by the trend watcher
        xv = predict(bd, y["va"], a.eval_batch); mv = metrics(xv, x["va"], y["va"])
        al = bd.model.alpha().detach(); gain = bd.model.log_gain.exp()
        rec = dict(pass_=ps + 1, train_rmse=float(np.sqrt(tot / n)), val_rmse=float(np.sqrt(((xv - x["va"]) ** 2).mean())),
                   val_gain=float(mv["snr_gain"].mean()), val_cc=float(mv["cc"].mean()), sec=time.time() - t0,
                   alpha_med=float(al.median()), bias_mean=float(bd.model.bias.mean()), gain_med=float(gain.median()),
                   gain_min=float(gain.min()), gain_max=float(gain.max()), in_gain_mean=float(bd.in_gain.mean()))
        hist.append(rec)
        print(f"pass {ps + 1}/{a.passes}: train RMSE {rec['train_rmse']:.4f}  val RMSE {rec['val_rmse']:.4f}  val SNR gain {rec['val_gain']:+.2f} dB  CC {rec['val_cc']:.3f}  "
              f"alpha med {rec['alpha_med']:.3f} gain [{rec['gain_min']:.2f},{rec['gain_max']:.2f}]  {rec['sec']:.0f}s", flush=True)
        if rec["val_gain"] > best["val_gain"]:
            best = {"val_gain": rec["val_gain"], "pass": ps, "state": {k_: v.detach().cpu().clone() for k_, v in bd.state_dict().items()}}
        torch.save({"state": bd.state_dict(), "opt": opt.state_dict(), "pass": ps, "hist": hist, "best": best,
                    **({"sched": sched.state_dict()} if sched is not None else {})}, ck)
        json.dump(hist, open(a.out / "history.json", "w"), indent=1)
        if mid.exists(): mid.unlink()
        if ps - best["pass"] >= a.patience: print("early stop", flush=True); break
        if (a.out / "STOP").exists(): print(f"STOP file: ending after pass {ps + 1} ({(a.out / 'STOP').read_text().strip()})", flush=True); break

    if best["state"] is not None: bd.load_state_dict(best["state"], strict=False)
    xh["brain"] = predict(bd, y["te"], a.eval_batch); res["brain"] = metrics(xh["brain"], x["te"], y["te"], snr0)
    name = "TRAINED brain + fixed readout" + (" (shuffled)" if a.shuffle else "")
    print(f"{name:30s} test {fmt(res['brain'])}   (best pass {best['pass'] + 1}, {time.time() - T0:.0f}s)", flush=True)
    for other in ("untrained", "r0", "fir"):
        if other in res: paired(res, "brain", other)
    summary = {k: {m: float(v.mean()) for m, v in r.items()} for k, r in res.items()}
    if snr0 is not None:
        for k, r in res.items():
            summary[k]["region_dsnr"] = float(np.mean([np.median(r["dsnr_region"][r["snr0"] == l]) for l in np.unique(r["snr0"])]))
    summary.update(args=vars(a) | {"out": str(a.out)}, best_pass=best["pass"] + 1, n_params=n_par, elapsed_s=time.time() - T0)
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    np.savez(a.out / "results.npz", clean=x["te"], noisy=y["te"], **{f"xhat_{k}": v for k, v in xh.items()},
             **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()})
    torch.save({k_: v for k_, v in bd.state_dict().items() if k_ in ("model.alpha_logit", "model.bias", "model.log_gain", "in_gain", "w_out", "b_out", "mu0", "sd0",
                                                                "model.syn_logit", "model.adapt_logit", "model.adapt_log_g", "model.log_slope",
                                                                "out_scale", "out_bias", "w_train", "model.w_new", "model.sg", "model.pre", "model.post")}, a.out / "brain_params.pt")
    print("saved", a.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
