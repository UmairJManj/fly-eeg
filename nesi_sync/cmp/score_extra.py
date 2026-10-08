"""Score extra channel-wise prediction files (cmp_data/<prefix>_S*_...npy) with run_compare.scores and merge into
compare.csv (replacing old rows of those methods), then rewrite the summary.   python score_extra.py PREFIX:NAME ..."""
import sys, os, numpy as np, pandas as pd
import run_compare as rc
pairs = [a.split(":") for a in sys.argv[1:]]; df = pd.read_csv(f"{rc.RES}/compare.csv"); rows = []
for s in sorted(df.subject.unique()):
    z = np.load(f"{rc.DATA}/S{s}.npz"); base, names, run, fs = z["base"].astype(np.float64), list(z["ch_names"]), z["run"], float(z["sfreq"])
    for nch, subset in rc.SUBSETS.items():
        idx = list(range(len(names))) if subset is None else [names.index(n) for n in subset if n in names]; nm = [names[i] for i in idx]
        for key in [k for k in z.files if k.startswith("noisy_")]:
            _, art, snr = key.split("_"); noisy = z[key].astype(np.float64)[idx]; b = base[idx]
            for prefix, mname in pairs:
                fp = f"{rc.DATA}/{prefix}_S{s}_{art}_{snr}_{len(idx)}.npy"
                if os.path.exists(fp):
                    rows.append(dict(subject=s, artifact=art, snr=int(snr), nch=len(idx), method=mname, seconds=0, note="", **rc.scores(np.load(fp).astype(np.float64), b, noisy, nm, run, fs)))
new = pd.DataFrame(rows); print(len(new), "rows scored")
df = pd.concat([df[~df.method.isin([m for _, m in pairs])], new]); df.to_csv(f"{rc.RES}/compare.csv", index=False)
ok = df[df.note.isna() | (df.note == "")]
g = ok.groupby(["artifact", "nch", "method"])[["snr_gain", "snr_gain_contam", "cc", "alpha_ratio_err"]].mean().round(3)
open(f"{rc.RES}/summary_v4_with_ai.md", "w").write("# Track C: multichannel comparison incl. channel-wise AI denoisers\n\n" + g.to_markdown() + "\n"); print(g)
