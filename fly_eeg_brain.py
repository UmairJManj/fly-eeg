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
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_eeg_denoise import (load_data, metrics, fmt, paired, make_model, jo_drive, Ridge, evaluate,
                             fir_features, CENTRAL, NFLY_DATA)
from nfly.connectome import load_malecns, build_connectome
from nfly.brain import InputDrive


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

    def states(self, y, idx=None):
        """Recorded activity of neurons `idx` (default: the readout wire's neurons), aligned to the input.
        With --tbptt K the unroll is split into K-step chunks and the state is detached between chunks
        (truncated BPTT: shorter, better-conditioned gradient paths; the forward dynamics are unchanged)."""
        a, T = self.a, y.shape[1]
        idx = self.out_idx if idx is None else idx
        yb = torch.as_tensor(np.pad(y, ((0, 0), (a.warm, a.lag)), mode="reflect"), device=self.w_out.device)
        drive = jo_drive(yb, self.in_idx, 1.0) * self.in_gain
        if not a.tbptt or not torch.is_grad_enabled():
            h = self.model(InputDrive(self.in_idx, drive), record=idx)[:, 1:]
        else:
            outs, h0 = [], None
            for s0 in range(0, drive.shape[1], a.tbptt):
                hist = self.model(InputDrive(self.in_idx, drive[:, s0:s0 + a.tbptt]), h0=h0, record=None)   # (B, k+1, N)
                outs.append(hist[:, 1:, idx]); h0 = hist[:, -1].detach()
            h = torch.cat(outs, 1)
        return h[:, a.warm + a.lag:a.warm + a.lag + T]                       # (B, T, K)

    def forward(self, y):
        return (self.states(y) @ self.w_out + self.b_out - self.mu0) / self.sd0   # (B, T)

    def brain_params(self):
        ps = [self.model.alpha_logit, self.model.bias, self.in_gain]
        if self.a.train_edges: ps.append(self.model.log_gain)
        return ps


@torch.no_grad()
def predict(bd, y, batch):
    bd.eval()
    return np.concatenate([bd(y[b:b + batch]).float().cpu().numpy() for b in range(0, len(y), batch)])


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
    p.add_argument("--artifact", default="eog", choices=["eog", "emg", "both"])
    p.add_argument("--protocol", default="whole", choices=["whole", "centered"])
    p.add_argument("--n-train", type=int, default=4000); p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--extra-train", type=int, default=0); p.add_argument("--aug", type=int, default=1)
    p.add_argument("--readout", default="random", choices=["random", "ridge0"])
    p.add_argument("--lam-min", type=float, default=1e-2, help="ridge0: smallest lambda allowed for the frozen head")
    p.add_argument("--shuffle", action="store_true", help="train the shuffled-wiring control brain instead")
    p.add_argument("--no-train-edges", dest="train_edges", action="store_false", help="only leak/bias/input gain (no synapse gains)")
    p.add_argument("--passes", type=int, default=30); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--eval-batch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="leak / bias / input gain"); p.add_argument("--lr-edge", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--tbptt", type=int, default=0, help="truncated-BPTT chunk length in samples (0 = full 600-step unroll)")
    p.add_argument("--wire", default="dn", choices=["dn", "all"], help="neurons under the fixed readout wire: descending/motor or all non-JO")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay on the synapse log-gains (pulls gains back to 1)")
    p.add_argument("--cosine", action="store_true", help="cosine learning-rate decay over all passes")
    p.add_argument("--lr-bias", type=float, default=None, help="separate learning rate for the neuron biases (default: --lr)")
    p.add_argument("--warmup", type=int, default=0, help="linear learning-rate warm-up over this many steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rho", type=float, default=0.9); p.add_argument("--alpha-min", type=float, default=0.02); p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--gain", type=float, default=1.0); p.add_argument("--bias", type=float, default=0.02); p.add_argument("--h-max", type=float, default=10.0)
    p.add_argument("--lag", type=int, default=24); p.add_argument("--warm", type=int, default=64)
    p.add_argument("--no-ref-ridge", action="store_true", help="skip the untrained-brain ridge reference (random readout mode)")
    p.add_argument("--no-fir", action="store_true")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.snr0 = None
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    T0 = time.time()

    x, y, snr = load_data(a, np.random.default_rng(0)); snr0 = snr["te"] if snr else None; a.snr0 = snr0   # centered protocol: nominal input SNR per test epoch (SPAR-EEG score)
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
    print(f"{conn.summary()}\ninput: {len(in_idx)} JO neurons; fixed readout wire over {len(out_idx)} neurons ({a.wire}); ridge reference on {len(dn_idx)} DN/motor neurons", flush=True)

    model = make_model(conn, a, a.device)
    for q in model.parameters(): q.requires_grad_(False)
    bd = BrainDenoiser(model, in_idx, out_idx, a).to(a.device)
    bd.register_buffer("dn_idx", dn_idx.to(a.device))
    ref = set_fixed_readout(bd, x, y, a, rng)
    if ref: xh["r0"], res["r0"] = ref["r0"]
    elif not a.no_ref_ridge:
        r = ridge_on_brain(bd, x, y, a).fit(); K = len(dn_idx)
        with torch.no_grad():
            xh["r0"] = np.concatenate([r.predict(bd.states(y["te"][b:b + a.eval_batch], dn_idx).reshape(-1, K)).reshape(-1, y["te"].shape[1]).float().cpu().numpy()
                                       for b in range(0, len(y["te"]), a.eval_batch)])
        res["r0"] = metrics(xh["r0"], x["te"], y["te"], snr0); print(f"{'untrained brain + ridge (r0)':30s} test {fmt(res['r0'])}   (lambda {r.lam})", flush=True)
    xh["untrained"] = predict(bd, y["te"], a.eval_batch); res["untrained"] = metrics(xh["untrained"], x["te"], y["te"], snr0)
    print(f"{'untrained brain + fixed readout':30s} test {fmt(res['untrained'])}   ({time.time() - T0:.0f}s)", flush=True)

    # ---- train the brain ----
    params = bd.brain_params()
    for q in params: q.requires_grad_(True)
    groups = [{"params": [bd.model.alpha_logit, bd.in_gain], "lr": a.lr},
              {"params": [bd.model.bias], "lr": a.lr if a.lr_bias is None else a.lr_bias}]
    if a.train_edges: groups.append({"params": [bd.model.log_gain], "lr": a.lr_edge, "weight_decay": a.wd})
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
    for ps in range(start, a.passes):
        bd.train(); perm, tot, t0, k = torch.randperm(n, generator=g).numpy(), 0.0, time.time(), 0
        for i0 in range(0, n, a.batch):
            i = np.sort(perm[i0:i0 + a.batch])
            pred = bd(y["tr"][i]); loss = ((pred - torch.as_tensor(x["tr"][i], device=a.device)) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip).item(); opt.step()
            if sched is not None: sched.step()
            tot += loss.item() * len(i); k += 1
            if k % 10 == 1: print(f"  pass {ps + 1} step {k}/{-(-n // a.batch)} loss {loss.item():.4f} gradnorm {gn:.3g} {(time.time() - t0) / k:.2f}s/step", flush=True)
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
        if ps - best["pass"] >= a.patience: print("early stop", flush=True); break

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
    torch.save({k_: v for k_, v in bd.state_dict().items() if k_ in ("model.alpha_logit", "model.bias", "model.log_gain", "in_gain", "w_out", "b_out", "mu0", "sd0")}, a.out / "brain_params.pt")
    print("saved", a.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
