"""Diffusion-style test (no retraining): apply a trained residual fly brain repeatedly. Each step estimates the remaining
artifact and removes a fraction alpha of it: y <- y - alpha * (y - brain(y)). Reports accuracy (CC with clean EEG) per step.
  python iterate_eval.py BRAIN_RUN_DIR [--steps 4] [--alphas 1.0 0.5]"""
import argparse, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/fly-eeg")
from fly_brain_apply import load_brain

def cc(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--steps", type=int, default=4)
p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5]); p.add_argument("--n", type=int, default=500)
a = p.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(Path(a.run) / "results.npz"); x, y0 = z["clean"][: a.n], z["noisy"][: a.n]
bd = load_brain(a.run, dev)

@torch.no_grad()
def clean(y):
    sc = y.std(1, keepdims=True) + 1e-8
    return np.concatenate([bd((y[b:b + 50] / sc[b:b + 50]).astype(np.float32)).float().cpu().numpy() for b in range(0, len(y), 50)]) * sc

print(f"noisy input        accuracy (CC) {100 * cc(y0, x).mean():.1f}%", flush=True)
for al in a.alphas:
    y = y0.copy()
    for k in range(1, a.steps + 1):
        y = y - al * (y - clean(y))
        snr = 10 * np.log10((x ** 2).sum(1) / ((y - x) ** 2).sum(1)) - 10 * np.log10((x ** 2).sum(1) / ((y0 - x) ** 2).sum(1))
        print(f"alpha {al:.2f} step {k}: accuracy (CC) {100 * cc(y, x).mean():.1f}%   (SNR gain {snr.mean():+.2f} dB)", flush=True)
