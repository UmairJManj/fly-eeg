"""ERP preservation check for a retuned fly brain (ground truth known).
A P300-like evoked response (N1 at 100 ms, P3 at 350 ms after a stimulus placed at 0.6 s) is added to every clean
EEGdenoiseNet test epoch AND to its noisy version. The brain cleans both noisy versions (with and without the ERP);
the ERP it lets through = mean over epochs of clean(noisy + erp) - clean(noisy). Reported:
  retention  = <recovered, erp> / <erp, erp>   (1.0 = amplitude kept)
  wave_cc    = correlation of recovered and true ERP waveform
  p3_peak    = recovered / true P3 peak amplitude, latency shift (ms)
Run from nfly:  python ../fly-eeg-nesi/erp_preserve.py --brain ../best_brain/eog_AC_span256_full --out DIR
"""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fly_eeg_denoise as fd
from fly_brain_apply import load_brain
from fly_eeg_brain import predict

FS, T = 256, 512


def erp_template(amp):
    t = np.arange(T) / FS - 0.6                       # stimulus at 0.6 s
    n1 = -0.5 * np.exp(-((t - 0.10) ** 2) / (2 * 0.025 ** 2))
    p3 = 1.0 * np.exp(-((t - 0.35) ** 2) / (2 * 0.08 ** 2))
    return (amp * (n1 + p3)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--brain", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--amps", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="P3 peak in units of the clean-epoch std")
    p.add_argument("--batch", type=int, default=50); p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default=None, help="override weights with a training checkpoint's state (ckpt.pt / ckpt_mid.pt)")
    p.add_argument("--n-test", type=int, default=0, help="use only the first N test epochs (quick check)")
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    bd = load_brain(a.brain, a.device); S = bd.a
    if a.ckpt:
        st = torch.load(a.ckpt, map_location=a.device, weights_only=False)["state"]; bd.load_state_dict(st, strict=False); bd.eval()
        print(f"weights from checkpoint {a.ckpt}", flush=True)
    da = argparse.Namespace(artifact=S.artifact, protocol="whole", n_train=S.n_train, n_test=S.n_test, extra_train=0, aug=S.aug,
                            artifact_split=S.artifact_split, snr_lo=-7.0, snr_hi=2.0)
    x, y, _ = fd.load_data(da, np.random.default_rng(0)); xc, yn = x["te"], y["te"]
    if a.n_test: xc, yn = xc[:a.n_test], yn[:a.n_test]
    base = predict(bd, yn, a.batch); res = {}
    for amp in a.amps:
        e = erp_template(amp)[None] * xc.std(1, keepdims=True)    # ERP scaled per epoch to that epoch's clean EEG
        out = predict(bd, yn + e, a.batch)
        rec, tru = (out - base).mean(0), e.mean(0)
        ret = float(rec @ tru / (tru @ tru)); wcc = float(np.corrcoef(rec, tru)[0, 1])
        i3 = int(np.argmax(tru)); j3 = int(np.argmax(rec[i3 - 40:i3 + 40])) + i3 - 40
        res[f"amp{amp}"] = dict(retention=ret, wave_cc=wcc, p3_ratio=float(rec[j3] / tru[i3]), p3_shift_ms=(j3 - i3) * 1000 / FS)
        # same metrics on the CLEANED vs CLEAN average (what an ERP study would see after averaging)
        m = fd.metrics(out, xc + e, yn + e, None)
        res[f"amp{amp}"].update(cc=float(m["cc"].mean()), snr_gain=float(m["snr_gain"].mean()))
        np.save(a.out / f"erp_amp{amp}.npy", np.stack([tru, rec]))
        print(f"amp {amp}: retention {ret:.3f}  wave CC {wcc:.3f}  P3 ratio {res[f'amp{amp}']['p3_ratio']:.3f} shift {res[f'amp{amp}']['p3_shift_ms']:+.1f} ms"
              f"  | cleaning with ERP: CC {res[f'amp{amp}']['cc']:.3f} SNR {res[f'amp{amp}']['snr_gain']:+.2f} dB", flush=True)
    json.dump(res, open(a.out / "erp_preserve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
