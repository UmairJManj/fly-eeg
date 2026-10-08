"""Round N trend watcher: one snapshot of every round-N arm (latest step + per-pass val curve), compared at equal
pass against the matched jo48 reference (eog_J_jo48_disjoint + the N seed replicates, best-so-far, averaged).
Lever arms that trail the reference by more than MARGIN dB at two consecutive passes (from pass MIN_PASS on) get a
STOP file: fly_eeg_brain.py then ends cleanly after the current pass and still writes the test summary.
Controls (seeds, base, shuffled) and the long-schedule arm are never stopped."""
import json, re, sys, time
from pathlib import Path
import numpy as np

BRAIN = Path("/nesi/project/aut04653/Manj/Fly/brain")
MARGIN, MIN_PASS = 1.5, 5
PROTECTED = ("_s1", "_s2", "_base", "_shuffled", "_long", "emg_", "both_")
REFS = ["eog_J_jo48_disjoint", "eog_N_jo48_s1", "eog_N_jo48_s2"]


def curve(tag):
    h = BRAIN / tag / "history.json"
    try:
        return [r["val_gain"] for r in json.load(open(h))]
    except Exception:
        return []


def ref_curve(exclude=None):
    cs = [np.maximum.accumulate(curve(t)) for t in REFS if curve(t) and t != exclude]
    L = max(len(c) for c in cs)
    return [float(np.mean([c[i] for c in cs if len(c) > i])) for i in range(L)], len(cs)


def last_step(tag):
    logs = sorted((BRAIN / tag).glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs: return None, None
    txt = logs[-1].read_text(errors="ignore")
    m = re.findall(r"pass (\d+) step (\d+)/(\d+) loss ([\d.]+) .* ([\d.]+)s/step", txt)
    age = time.time() - logs[-1].stat().st_mtime
    return (m[-1] if m else None), age


def main():
    ref, nref = ref_curve()
    tags = [l.split()[0] for l in open(BRAIN / "units.txt") if re.match(r"^(eog|emg|both|ecg|motion)_[NX]_", l)]
    print(f"--- {time.strftime('%H:%M')} round N (ref = jo48 best-so-far mean of {nref} run(s): {' '.join(f'{v:.1f}' for v in ref)})")
    for t in tags:
        d = BRAIN / t
        if (d / "summary.json").exists():
            s = json.load(open(d / "summary.json")); print(f"  {t:22s} DONE test CC {100 * s['brain']['cc']:.1f}%  SNR {s['brain']['snr_gain']:+.2f} dB  RRMSE {s['brain']['rrmse']:.3f}"); continue
        if not d.exists():
            print(f"  {t:22s} queued"); continue
        c = curve(t); st, age = last_step(t)
        pos = f"p{st[0]} {st[1]}/{st[2]} loss {float(st[3]):.3f} {float(st[4]):.1f}s/st" if st else "starting"
        stale = f" (log idle {age / 60:.0f} min)" if age and age > 900 else ""
        verdict = ""
        if c:
            b = np.maximum.accumulate(c); k = len(b)
            rt, _ = ref_curve(exclude=t)
            if k <= len(rt) and t.startswith("eog_"): verdict = f" | vs ref @p{k}: {b[-1] - rt[k - 1]:+.1f}"
            behind = [i for i in range(len(b)) if i < len(rt) and i + 1 >= MIN_PASS and b[i] < rt[i] - MARGIN]
            if (t.startswith("eog_") and not any(p in t for p in PROTECTED) and len(behind) >= 2
                    and behind[-1] == len(b) - 1 and behind[-2] == len(b) - 2 and not (d / "STOP").exists()):
                (d / "STOP").write_text(f"trails jo48 ref by >{MARGIN} dB at passes {behind[-2] + 1},{behind[-1] + 1}")
                verdict += "  -> STOP written"
            elif (d / "STOP").exists(): verdict += "  (stopping)"
        print(f"  {t:22s} {pos}{stale} | val {' '.join(f'{v:.1f}' for v in c) or '-'}{verdict}")


if __name__ == "__main__":
    main()
