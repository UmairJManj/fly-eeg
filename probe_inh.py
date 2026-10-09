"""Typical inhibitory synaptic input I_i = -(W_inh h)_i in a trained brain (sets the scale of --divnorm k_i)."""
import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
from nfly.brain import rnn as R
bd = load_brain(sys.argv[1], "cpu"); S = bd.a; m = bd.model
da = argparse.Namespace(artifact="eog", protocol="whole", n_train=4000, n_test=500, extra_train=0, aug=4, artifact_split="disjoint", snr_lo=-7.0, snr_hi=2.0)
x, y, _ = fd.load_data(da, np.random.default_rng(0)); yt = torch.as_tensor(y["te"][:3] / y["te"][:3].std(1, keepdims=True), dtype=torch.float32)
with torch.no_grad():
    H = bd.states(yt, torch.arange(m.n))                                  # (B, T, N) all neurons
    W = m.weights(); w_inh = W.w * (m.sign < 0).float()
    h = H[:, ::32].reshape(-1, m.n)
    I = -R._SparseRecurrent.apply(h, w_inh, m.pre, m.post, m.edge_chunk); E = R._SparseRecurrent.apply(h, W.w * (m.sign > 0).float(), m.pre, m.post, m.edge_chunk)
    q = lambda v: [round(float(np.percentile(v.numpy(), p)), 4) for p in (50, 90, 99)]
    print("h      p50/p90/p99", q(h.abs())); print("I_inh  p50/p90/p99", q(I.clamp(min=0))); print("E_exc  p50/p90/p99", q(E))
