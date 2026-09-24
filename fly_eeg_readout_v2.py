"""Readout suite v2 on cached fly features (fly_eeg_denoise.py --save-states DIR).
The brain is fixed. Candidates that listen to the descending / motor neurons:
  ridge{5,9,17}   ridge over 5 / 9 / 17 time offsets, lambda grid down to 1e-7
  mlp{9,17}       2-layer MLP over 9 / 17 offsets, hidden 1024, long cosine schedule, seed ensemble
  tcn             per-sample linear embedding + dilated temporal conv stack (learned filters over
                  +-62 samples), seed ensemble
Writes DIR/readouts_v2_<tag>.json (per-method metrics incl. per-level region dSNR medians) and
DIR/readouts_v2_<tag>.npz (test predictions of the best method).
Run from the nfly folder:  uv run --no-sync python ../fly-eeg/fly_eeg_readout_v2.py DIR --configs ridge9,mlp9,tcn
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from fly_eeg_denoise import metrics, fmt, Ridge, LEVELS


def load(d):
    S = {k: np.load(d / f"{k}_states.npy", mmap_mode="r") for k in ("tr", "va", "te")}
    x = {k: np.load(d / f"{k}_clean.npy") for k in S}
    y = {k: np.load(d / f"{k}_noisy.npy") for k in S}
    snr0 = np.load(d / "te_snr.npy") if (d / "te_snr.npy").exists() else None
    return S, x, y, snr0


def shifted(X, shifts):
    T = X.shape[1]
    idx = torch.arange(T, device=X.device)
    return torch.cat([X[:, (idx + d).clamp(0, T - 1)] for d in shifts], -1)


def summary(m):
    """Scalar summary: means of RRMSE / CC / SNR gain and the SPAR-EEG score (mean over levels of
    the per-level median artifact-region dSNR) plus the per-level medians themselves."""
    out = {k: float(np.mean(m[k])) for k in ("rrmse", "cc", "snr_gain")}
    out["rrmse_median"] = float(np.median(m["rrmse"]))
    if "dsnr_region" in m:
        lv = {int(l): float(np.median(m["dsnr_region"][m["snr0"] == l])) for l in LEVELS}
        out["region_dsnr"] = float(np.mean(list(lv.values())))
        out["per_level"] = lv
    out["score"] = out.get("region_dsnr", out["snr_gain"])     # SPAR-EEG score when levels exist, else whole-epoch SNR gain
    return out


def batches(S, split, dev, batch=25):
    for b in range(0, len(S[split]), batch):
        X = S[split][b:b + batch]
        yield b, (X.to(dev) if torch.is_tensor(X) else torch.as_tensor(np.array(X), device=dev)).float()


def ridge_eval(S, x, y, shifts, dev, snr0):
    F = S["tr"].shape[-1] * len(shifts)
    r = Ridge(F, dev)
    for split in ("tr", "va"):
        for b, X in batches(S, split, dev, max(4, 25 // len(shifts))):
            r.add(split, shifted(X, shifts).reshape(-1, F), torch.as_tensor(x[split][b:b + len(X)], device=dev).reshape(-1))
    r.fit(lams=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
    xhat = np.concatenate([r.predict(shifted(X, shifts).reshape(-1, F)).reshape(len(X), -1).float().cpu().numpy()
                           for _, X in batches(S, "te", dev, max(4, 25 // len(shifts)))])
    return xhat, metrics(xhat, x["te"], y["te"], snr0), dict(lam=r.lam, val_rmse=float(np.sqrt(r.val_mse)))


class Std(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, X):
        return ((X - self.mu) / self.sd).clamp(-10, 10)


class MLP(nn.Module):
    def __init__(self, mu, sd, F, hidden, shifts):
        super().__init__()
        self.shifts, self.std = shifts, Std(mu.repeat(len(shifts)), sd.repeat(len(shifts)))
        self.net = nn.Sequential(nn.Linear(F, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, X):                      # X (B, T, K) -> (B, T)
        return self.net(self.std(shifted(X, self.shifts))).squeeze(-1)

    def rows(self, R):                         # R (B, n_shifts, K) gathered rows -> (B,)
        return self.net(self.std(R.reshape(len(R), -1))).squeeze(-1)


class TCN(nn.Module):
    """Linear embedding of the K neurons per sample, then dilated non-causal conv blocks with
    residuals. Kernel 5, dilations 1,2,4,8,16 -> receptive field +-62 samples."""
    def __init__(self, mu, sd, K, width=256, dil=(1, 2, 4, 8, 16), k=5, drop=0.1):
        super().__init__()
        self.std = Std(mu, sd)
        self.emb = nn.Linear(K, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, k, padding=d * (k - 1) // 2, dilation=d), nn.GELU(),
                                                   nn.Dropout(drop), nn.Conv1d(width, width, 1)) for d in dil])
        self.norms = nn.ModuleList([nn.GroupNorm(1, width) for _ in dil])
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, X):                      # X (B, T, K) -> (B, T)
        h = self.emb(self.std(X)).transpose(1, 2)     # (B, W, T)
        for blk, nm in zip(self.blocks, self.norms):
            h = h + blk(nm(h))
        return self.head(h).squeeze(1)


class FCNN(nn.Module):
    """EEGdenoiseNet-style fully-connected denoiser: flatten -> 3 x Dense(T) ReLU + dropout -> T."""
    def __init__(self, mu, sd, K, T=512, drop=0.3):
        super().__init__(); self.std = Std(mu, sd); self.T, self.K = T, K
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(T * K, T), nn.ReLU(), nn.Dropout(drop), nn.Linear(T, T), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(T, T), nn.ReLU(), nn.Dropout(drop), nn.Linear(T, T))
    def forward(self, X): return self.net(self.std(X))


class SimpleCNN(nn.Module):
    """EEGdenoiseNet-style simple CNN: 4 x Conv1d(64, k=3) ReLU, flatten -> Dense(T)."""
    def __init__(self, mu, sd, K, T=512):
        super().__init__(); self.std = Std(mu, sd)
        self.conv = nn.Sequential(nn.Conv1d(K, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * T, T))
    def forward(self, X): return self.head(self.conv(self.std(X).transpose(1, 2)))


class ComplexCNN(nn.Module):
    """EEGdenoiseNet-style complex CNN: stride-2 conv stages 32-64-128-256 (2 convs each, BN, ReLU), flatten -> Dense(T)."""
    def __init__(self, mu, sd, K, T=512):
        super().__init__(); self.std = Std(mu, sd); layers, c_in = [], K
        for c in (32, 64, 128, 256):
            layers += [nn.Conv1d(c_in, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c, 5, padding=2, stride=2), nn.BatchNorm1d(c), nn.ReLU()]; c_in = c
        self.conv = nn.Sequential(*layers); self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256 * (T // 16), T))
    def forward(self, X): return self.head(self.conv(self.std(X).transpose(1, 2)))


class RNNDenoiser(nn.Module):
    """EEGdenoiseNet-style RNN: bidirectional LSTM(64) over samples -> linear per sample."""
    def __init__(self, mu, sd, K, hidden=64):
        super().__init__(); self.std = Std(mu, sd); self.rnn = nn.LSTM(K, hidden, batch_first=True, bidirectional=True); self.head = nn.Linear(2 * hidden, 1)
    def forward(self, X): return self.head(self.rnn(self.std(X))[0]).squeeze(-1)


class XfmrDenoiser(nn.Module):
    """EEGDnet / DenoiseFormer-style transformer: 16-sample patches -> d=128, 4 encoder layers, 4 heads -> unpatch."""
    def __init__(self, mu, sd, K, T=512, patch=16, d=128, layers=4, heads=4, drop=0.1):
        super().__init__(); self.std = Std(mu, sd); self.patch, self.T = patch, T
        self.emb = nn.Linear(patch * K, d); self.pos = nn.Parameter(torch.zeros(1, T // patch, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers); self.head = nn.Linear(d, patch)
    def forward(self, X):
        B, T, K = X.shape; Z = self.std(X).reshape(B, T // self.patch, self.patch * K)
        return self.head(self.enc(self.emb(Z) + self.pos)).reshape(B, T)


BASELINES = {"fcnn": FCNN, "scnn": SimpleCNN, "ccnn": ComplexCNN, "rnn": RNNDenoiser, "xfmr": XfmrDenoiser}


def train_net(make, S, x, y, dev, steps, batch, lr, wd, seed, snr0, name, whole_epochs):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = make().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    Str, xtr = S["tr"], torch.as_tensor(x["tr"], device=dev)
    N, T = xtr.shape

    def predict(split):
        model.eval()
        with torch.no_grad():
            out = np.concatenate([model(X).cpu().numpy() for _, X in batches(S, split, dev)])
        model.train()
        return out

    best, t0 = (np.inf, None), time.time()
    for step in range(1, steps + 1):
        idev = Str.device                       # index on the cache's device, compute on dev
        e = torch.randint(N, (batch,), generator=g).to(idev)
        if whole_epochs:
            pred, tgt = model(Str[e].to(dev, non_blocking=True).float()), xtr[e.to(dev)]
        else:                                   # random (epoch, time) rows for the MLP, gathered at the shifts
            t = torch.randint(T, (batch,), generator=g).to(idev)
            tt = (t[:, None] + torch.as_tensor(model.shifts, device=idev)).clamp(0, T - 1)
            pred, tgt = model.rows(Str[e[:, None], tt].to(dev).float()), xtr[e.to(dev), t.to(dev)]
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            v = float(np.sqrt(((predict("va") - x["va"]) ** 2).mean()))
            if v < best[0]:
                best = (v, {k: t_.detach().clone() for k, t_ in model.state_dict().items()})
            print(f"  [{name} seed {seed}] step {step}: train RMSE {loss.sqrt().item():.3f}  val RMSE {v:.3f}  {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best[1])
    xhat = predict("te")
    train_net.last_model = model                      # exposed for --save-model
    return xhat, metrics(xhat, x["te"], y["te"], snr0), dict(val_rmse=best[0], params=sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--configs", default="ridge5,ridge9,ridge17,mlp9,tcn")
    p.add_argument("--tag", default="r1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--mlp-steps", type=int, default=30000)
    p.add_argument("--mlp-hidden", type=int, default=1024)
    p.add_argument("--tcn-steps", type=int, default=12000)
    p.add_argument("--tcn-width", type=int, default=256)
    p.add_argument("--tcn-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--gpu-cache-gb", type=float, default=20.0, help="train/val caches larger than this stay in host RAM")
    p.add_argument("--save-model", type=Path, default=None, help="save every trained TCN (state + feature standardisation + config) here, for fly_apply.py")
    a = p.parse_args()
    dev = a.device
    S, x, y, snr0 = load(a.dir)
    K = S["tr"].shape[-1]
    print(f"states {S['tr'].shape} train, {S['va'].shape} val, {S['te'].shape} test", flush=True)
    # train / val as fp16 tensors: on the GPU when they fit (--gpu-cache-gb), else in host RAM; test streams from the memmap
    t0 = time.time()
    S = dict(S)
    for k in ("tr", "va"):
        gb = S[k].nbytes / 1e9
        S[k] = torch.from_numpy(np.ascontiguousarray(S[k])) if gb > a.gpu_cache_gb else torch.as_tensor(np.array(S[k]), device=dev)
        print(f"{k} {gb:.1f} GB on {S[k].device} in {time.time() - t0:.0f}s", flush=True)
    mu = torch.stack([X.reshape(-1, K).mean(0) for _, X in batches(S, "tr", dev)]).mean(0)
    sd = torch.stack([X.reshape(-1, K).var(0) for _, X in batches(S, "tr", dev)]).mean(0).sqrt()
    sd = sd.clamp_min(1e-2 * sd.median())
    SH = {5: [-24, -12, 0, 12, 24], 9: list(range(-32, 33, 8)), 17: list(range(-32, 33, 4))}

    res, preds = {"noisy": summary(metrics(y["te"], x["te"], y["te"], snr0))}, {}
    for cfg in a.configs.split(","):
        t0 = time.time()
        if cfg.startswith("ridge"):
            xhat, m, info = ridge_eval(S, x, y, SH[int(cfg[5:])], dev, snr0)
            res[cfg], preds[cfg] = {**summary(m), **info}, xhat
        else:
            ens, infos = [], []
            for seed in range(a.seeds):
                if cfg.split("+")[0] in BASELINES:      # published single-channel architectures (EEGdenoiseNet / EEGDnet style)
                    o = {"s": a.tcn_steps, "b": a.tcn_batch}
                    for tok in cfg.split("+")[1:]:
                        o[tok[0]] = type(o[tok[0]])(tok[1:])
                    cls = BASELINES[cfg.split("+")[0]]
                    make = lambda: cls(mu, sd, K)
                    xhat, m, info = train_net(make, S, x, y, dev, int(o["s"]), int(o["b"]), a.lr, a.wd, seed, snr0, cfg, True)
                elif cfg.startswith("mlp"):
                    sh = SH[int(cfg[3:])]
                    make = lambda: MLP(mu, sd, K * len(sh), a.mlp_hidden, sh)
                    xhat, m, info = train_net(make, S, x, y, dev, a.mlp_steps, 1024, a.lr, a.wd, seed, snr0, cfg, False)
                else:                       # tcn[+wW+dD+pP+bB+sS]: width, n dilation blocks, dropout, batch, steps
                    o = {"w": a.tcn_width, "d": 5, "p": 0.1, "b": a.tcn_batch, "s": a.tcn_steps}
                    for tok in cfg.split("+")[1:]:
                        o[tok[0]] = type(o[tok[0]])(tok[1:])
                    dil = tuple(2 ** i for i in range(int(o["d"])))
                    make = lambda: TCN(mu, sd, K, int(o["w"]), dil, drop=float(o["p"]))
                    xhat, m, info = train_net(make, S, x, y, dev, int(o["s"]), int(o["b"]), a.lr, a.wd, seed, snr0, cfg, True)
                    if a.save_model:
                        a.save_model.mkdir(parents=True, exist_ok=True)
                        torch.save({"state": {k_: v.cpu() for k_, v in train_net.last_model.state_dict().items()}, "mu": mu.cpu(), "sd": sd.cpu(), "K": K,
                                    "width": int(o["w"]), "dil": dil, "drop": float(o["p"]), "cfg": cfg, "seed": seed, "val_rmse": info["val_rmse"]},
                                   a.save_model / f"{cfg.replace('+', '_')}_seed{seed}.pt")
                ens.append(xhat)
                infos.append({**summary(m), **info})
                print(f"{cfg} seed {seed:34d} test {fmt(m)}", flush=True)
            xhat = np.mean(ens, 0)
            m = metrics(xhat, x["te"], y["te"], snr0)
            res[cfg] = {**summary(m), "seeds": infos, "params": infos[0]["params"], "single_mean_score": float(np.mean([i["score"] for i in infos]))}
            preds[cfg] = xhat
        res[cfg]["seconds"] = time.time() - t0
        print(f"{cfg:34s} test {fmt(m)}   score {res[cfg]['score']:+.2f} dB   ({res[cfg]['seconds']:.0f}s)", flush=True)

    nets = [k for k in preds if not k.startswith("ridge")]
    if len(nets) > 1:                          # ensemble over every trained-net config (all seeds)
        xhat = np.mean([preds[k] for k in nets], 0)
        m = metrics(xhat, x["te"], y["te"], snr0)
        res["ens_all"], preds["ens_all"] = {**summary(m), "members": nets}, xhat
        print(f"{'ens_all':34s} test {fmt(m)}", flush=True)
    best = max((k for k in res if k != "noisy"), key=lambda k: res[k]["score"])
    res["best"] = best
    print("\nmethod           RRMSE    CC   SNRgain  regionDSNR")
    for k, r in res.items():
        if isinstance(r, dict):
            print(f"{k:14s} {r['rrmse']:7.3f} {r['cc']:6.3f} {r['snr_gain']:+8.2f} {r.get('region_dsnr', float('nan')):+10.2f}")
    print(f"best: {best}")
    (a.dir / f"readouts_v2_{a.tag}.json").write_text(json.dumps(res, indent=1))
    np.savez(a.dir / f"readouts_v2_{a.tag}.npz", clean=x["te"], noisy=y["te"], best=preds[best], best_name=best)


if __name__ == "__main__":
    main()
