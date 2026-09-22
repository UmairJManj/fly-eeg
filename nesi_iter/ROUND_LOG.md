# Fly EEG cleaner: iteration log (monitor loop)

Goal: push the fixed fly-brain reservoir + trained readout as close to a perfect EEG cleaner as it
will go, on the SPAR-EEG centered protocol (eog / emg / both). Score = region dSNR (mean over 26
levels of per-level medians); also RRMSE and CC. Stop rule: two consecutive rounds with < 0.1 dB
improvement on every artifact = ceiling reached (report it as the ceiling, not "perfect").

Baseline (round 0, jobs 9233586/87/97, readout v1 mlp 5-shift, 6k steps):
| artifact | RRMSE | CC | region dSNR | SPAR-EEG |
| eog  | 0.421 | 0.899 | 11.70 | 8.28 |
| emg  | 0.440 | 0.888 | 10.50 | 9.05 |
| both | 0.470 | 0.874 |  9.88 | 8.00 |

Levers, in order: (1) readout capacity/training (MLP under-trained), (2) denser time shifts /
learned temporal filters (TCN), (3) more training records (2400 -> 3250 of 3400) + augmentation
(needs brain re-run), (4) one readout over all artifacts, (5) brain tuning (--train neuron).

## Round 1 (readout only, cached states): ridge5/9/17, mlp9 (1024 hidden, 30k steps, 3 seeds), tcn (3 seeds)
smoke=9234758
round1 jobs: smoke 9234758 -> eog 9234759, emg 9234760, both 9234761 (submitted 15:20)
smoke 9234758 FAILED (launcher path bug, fixed) -> resubmitted smoke 9234793, chained eog/emg/both re-queued 15:26
smoke 9234793 PASSED (tcn best on smoke); eog 9234794 moved to debug QOS 15:31

### Round 1 result, eog (job 9234794): ridge5 10.64 | ridge9 10.84 | ridge17 10.94 | mlp9 x3 ens 12.00 | TCN x3 ens 13.66 (RRMSE 0.309, CC 0.943)
TCN single seeds 13.26-13.38; ensemble +0.3. TCN train RMSE 0.06 vs val 0.174 -> data-limited. emg/both r1 pending.

## Round 2a (readout only, existing caches): TCN sweep tcn+s20000 / +w512 / +d6 (RF +-126) / +p0.3, 3 seeds each  [submitted 15:47]
## Round 2b (brain re-run, --aug 4: 4 SNR + artifact-pairing draws per train record, same test records): ridge5 + tcn+s20000  [smoke 9235735 first]
### Round 1 result, emg (job 9234795): ridge5 +8.88 | ridge9 +9.15 | ridge17 +9.25 | mlp9 +10.67 | tcn +11.30 | best:  |   (TCN RRMSE 0.386)
### Round 1 result, both (job 9234796): ridge5 +8.24 | ridge9 +8.45 | ridge17 +8.55 | mlp9 +10.01 | tcn +11.00 |  (TCN RRMSE 0.396)
aug smoke 9235735 PASSED (tcn 10.09 on 192-epoch smoke); aug4-eog 9235736 -> debug QOS 16:19
### Round 2a result, eog (job 9235732, 3-seed ensembles): tcn+s20000 13.73 | +w512 13.79 | +d6 14.06 | +p0.3 13.99  (r1 tcn 13.66)
Decision: receptive field (d6 = 6 dilation blocks, +-126 samples) and dropout 0.3 both help, width does not, 20k vs 12k steps +0.07.
Round 3 plan: combine tcn+d6+p0.3+s20000 (and d7) on the aug4 caches from round 2b, 5 seeds.
### Round 2a result, emg (job 9235733): tcn+s20000 +11.33 | tcn+w512+s20000 +11.34 | tcn+d6+s20000 +11.91 | tcn+p0.3+s20000 +11.48 | best:  |  (r1 tcn 11.30)
### Round 2a result, both (job 9235734): tcn+s20000 +11.06 | tcn+w512+s20000 +11.09 | tcn+d6+s20000 +11.56 | tcn+p0.3+s20000 +11.19 | best:  |  (r1 tcn 11.00)
Round 2a summary (best = tcn+d6+s20000 everywhere): eog 14.06 | emg 11.91 | both 11.56   (r1: 13.66 / 11.30 / 11.00; r0: 11.70 / 10.50 / 9.88)
## Round 3 (readout): tcn+d6+p0.3+s20000 and tcn+d7+p0.3+s20000, 5 seeds; on base caches (fly-rd3-*) AND on aug4 caches (fly-rd3aug-*, chained on 2b)  [resubmitted 16:25 after RD_EXTRA launcher fix]
### Round 2b result, eog aug4 (job 9235736, 21 min total, 14 min brain): ridge5 10.75 | tcn+s20000 x3 15.41 (RRMSE 0.259, CC 0.957)   vs same readout no-aug 13.73 -> +1.7 dB from 4x data
Decision: data is the main lever. Round 4 = aug 12 (same 2400 records + same test set), readout tcn+s20000 (comparison) and tcn+d6+p0.3+s30000; train cache kept on CPU when > 20 GB.
## Round 4 (brain re-run aug 12): fly-aug12-eog/emg/both; readout cpu-cache smoke 9239764  [submitted 16:42]
### Round 2b result, emg aug4 (job 9235737): ridge5 +8.89 | tcn+s20000 +12.12 |  (no-aug tcn+s20000 11.33)
### Round 2b result, both aug4 (job 9235738): ridge5 +8.30 | tcn+s20000 +11.80 |  (no-aug tcn+s20000 11.06)
### Round 3 result, eog aug4 (job 9238511, 5 seeds): tcn+d7+p0.3+s20000 +16.80 (RRMSE 0.215, CC 0.970) | best:  (RRMSE tcn+d7+p0.3+s20000, CC ) |  (aug4 tcn+s20000 15.41)
Decision (r3 eog aug4): d6->d7 still +0.75 dB (RF +-254 samples), single-seed 16.47 vs 5-seed ensemble 16.80. Receptive field not exhausted.
## Round 5 (readout on aug12 caches, chained on r4): tcn+d7+p0.3+s30000 and tcn+d8+p0.3+s30000 (RF = whole epoch), 5 seeds  [submitted 17:10]
### Round 3 result, emg aug4 (job 9238512, 5 seeds): tcn+d7+p0.3+s20000 +15.78 (RRMSE 0.238, CC 0.963) | best:  (RRMSE tcn+d7+p0.3+s20000, CC ) |  (aug4 tcn+s20000 12.12)
### Round 3 result, both aug4 (job 9238513, 5 seeds): tcn+d6+p0.3+s20000 +12.90 (RRMSE 0.320, CC 0.939) | tcn+d7+p0.3+s20000 +15.72 (RRMSE 0.243, CC 0.962) |  (aug4 tcn+s20000 11.80)
### Round 4 result, eog aug12 (job 9239765): tcn+s20000 +15.94 (RRMSE 0.244, CC 0.961) | tcn+d6+p0.3+s30000 +16.58 (RRMSE 0.230, CC 0.965) |  (aug4: tcn+s20000 15.41, tcn+d6+p0.3 16.05)
Decision (r4 eog aug12): 4x->12x draws only +0.5 dB (15.41->15.94 base, 16.05->16.58 d6) -> augmentation of the same records is saturating; next data lever = more distinct clean records.
## Round 6 (brain re-run, aug 12 + --extra-train -1: all 850 unused records added to train, SAME test set): readout tcn+d7+p0.3+s30000 x3  [smoke 9244402, then fly-r6all-eog/emg/both, submitted 17:52]
### Round 3 result, eog BASE cache (job 9238508, 5 seeds): tcn+d6+p0.3+s20000 +14.35 | tcn+d7+p0.3+s20000 +14.78 |  (r2a d6 14.06; aug4 same readouts 16.05 / 16.80)
xtr smoke 9244402 PASSED (train train 176 epochs with XTR=40 AUG=2)
### Round 4 result, emg aug12 (job 9239766): tcn+s20000 +12.52 (RRMSE 0.333, CC 0.934) | tcn+d6+p0.3+s30000 +13.95 (RRMSE 0.281, CC 0.952) |  (aug4: tcn+s20000 12.12; aug4 d7 15.78)
### Round 3 result, emg BASE cache (job 9238509, 5 seeds): tcn+d6+p0.3+s20000 +12.06 | tcn+d7+p0.3+s20000 +13.16 |  (r2a d6 11.91; aug4 d7 15.78)
### Round 4 result, both aug12 (job 9239767): tcn+s20000 +12.19 (RRMSE 0.341, CC 0.932) | tcn+d6+p0.3+s30000 +13.75 (RRMSE 0.293, CC 0.948) |  (aug4: tcn+s20000 11.80; aug4 d7 15.72)
### Round 5 result, eog aug12 (job 9241477, 5 seeds): tcn+d7+p0.3+s30000 +18.42 (RRMSE 0.190, CC 0.975) | tcn+d8+p0.3+s30000 +19.48 (RRMSE 0.178, CC 0.978) |  (aug12 d6 16.58; aug4 d7 16.80)
Decision (r5 eog aug12): d7->d8 +1.06 dB (18.42 -> 19.48; single-seed 18.98, 5-seed ens 19.48). Whole-epoch context is the strongest readout lever.
## Round 7 (readout on r6 all-records aug12 caches, chained on r6): tcn+d8+p0.3+s40000 and tcn+d9+p0.3+s40000 (full context), 5 seeds  [submitted 18:53]
### Round 6 result, eog aug12+all records (job 9244403, 3 seeds): tcn+d7+p0.3+s30000 +18.50 (RRMSE 0.188, CC 0.976) (aug12 2400-rec d7 x5 18.42)
Decision (r6 eog): +850 distinct records = +0.08 dB -> data saturated on both axes (draws and records); remaining lever = readout context (d8/d9 in r7) and ensembling.
### Round 6 result, emg aug12+all records (job 9244404, 3 seeds): tcn+d7+p0.3+s30000 +17.32 (RRMSE 0.203, CC 0.973) (aug12 2400-rec d6 13.95; aug4 d7 15.78)
### Round 6 result, both aug12+all records (job 9244405, 3 seeds): tcn+d7+p0.3+s30000 +17.00 (RRMSE 0.213, CC 0.971) (aug12 2400-rec d6 13.75; aug4 d7 15.72)
### Round 3 result, both BASE cache (job 9238510, 5 seeds): tcn+d6+p0.3+s20000 +11.66 | tcn+d7+p0.3+s20000 +12.65 |  (r2a d6 11.56; aug4 d7 15.72)
### Round 5 result, emg aug12 (job 9241478, 5 seeds): tcn+d7+p0.3+s30000 +17.64 (RRMSE 0.212, CC 0.969) | tcn+d8+p0.3+s30000 +18.85 (RRMSE 0.196, CC 0.971) |  (aug12 d6 13.95; r6 all-rec d7 x3 17.32)
### Round 5 result, both aug12 (job 9241479, 5 seeds): tcn+d7+p0.3+s30000 +17.34 (RRMSE 0.219, CC 0.967) | tcn+d8+p0.3+s30000 +18.52 (RRMSE 0.203, CC 0.969) |  (aug12 d6 13.75; r6 all-rec d7 x3 17.00)
### Round 7 result, eog all-rec aug12 (job 9248534, 5 seeds): tcn+d8+p0.3+s40000 +19.89 (RRMSE 0.164, CC 0.982) | tcn+d9+p0.3+s40000 +20.26 (RRMSE 0.159, CC 0.982) |  (r5 2400-rec d8 19.48)
Decision (r7 eog): d8 40k all-rec 19.89, d9 20.26 (+0.37); per-round gains now ~0.4 dB -> consolidation round.
## Round 8 (final push, all-rec aug12 caches): tcn+d9+p0.3+s60000 and tcn+d9+w384+p0.3+s60000, 6 seeds each + ens_all (12-model ensemble)  [submitted 20:34]
### Round 7 result, emg all-rec aug12 (job 9248535, 5 seeds): tcn+d8+p0.3+s40000 +19.39 (RRMSE 0.174, CC 0.978) | tcn+d9+p0.3+s40000 +19.77 (RRMSE 0.170, CC 0.979) |  (r5 2400-rec d8 18.85)
### Round 7 result, both all-rec aug12 (job 9248536, 5 seeds): tcn+d8+p0.3+s40000 +19.05 (RRMSE 0.182, CC 0.976) | tcn+d9+p0.3+s40000 +19.33 (RRMSE 0.178, CC 0.977) |  (r5 2400-rec d8 18.52)
CAVEAT (from fig1): final per-level gains reach ~30 dB at -20 dB input. In the centered protocol the artifact sits only in the middle third, so a whole-epoch readout can partly reconstruct that third from the clean outer thirds. Legitimate under SPAR-EEG's protocol, but must be reported.
## Honesty check W1 (EEGdenoiseNet WHOLE protocol, artifact over the full epoch, SNR U(-7,2), aug 4, 4000/500): ridge5, tcn (d5), tcn+d9+p0.3+s40000, 3 seeds -> does wide context still help when there is no clean context?  [fly-whole4-eog/emg/both submitted 20:50]
W1 9255337 FAILED at launch (set -e + failing $(...) in STATES line when XTR=0; fixed with || true) -> resubmitted 21:00
W1 eog 9255405: brain phase OK (whole protocol, ridge fly +10.57 dB vs FIR +7.98), readout crashed (no per-level SNR in whole protocol -> score fallback to snr_gain added); readout-only resubmitted 9256280 21:36
### W1 result, eog WHOLE protocol aug4 (job 9256280, 3 seeds; SNR gain dB, whole epoch): ridge5 +11.15 (RRMSE 0.387, CC 0.918) | tcn+s20000 +13.05 (RRMSE 0.319, CC 0.942) | tcn+d9+p0.3+s40000 +15.04 (RRMSE 0.258, CC 0.961) | ens_all +14.40 (RRMSE 0.275, CC 0.956) |  FIR +7.98, ridge fly +10.57; laptop v1 mlp +11.33
### W1 result, artifact=emg WHOLE protocol aug4 (job 9255406, 3 seeds; SNR gain): ridge5 +7.40 (RRMSE 0.581, CC 0.816) | tcn+s20000 +8.84 (RRMSE 0.499, CC 0.859) | tcn+d9+p0.3+s40000 +10.05 (RRMSE 0.438, CC 0.891) | ens_all +9.68 (RRMSE 0.455, CC 0.883) |  FIR SNR gain +7.31 dB
### W1 result, artifact=both WHOLE protocol aug4 (job 9255407, 3 seeds; SNR gain): ridge5 +7.36 (RRMSE 0.592, CC 0.808) | tcn+s20000 +8.89 (RRMSE 0.504, CC 0.855) | tcn+d9+p0.3+s40000 +10.13 (RRMSE 0.442, CC 0.887) | ens_all +9.76 (RRMSE 0.458, CC 0.879) |  FIR SNR gain +7.31 dB
### Round 8 result, emg all-rec aug12 (job 9254952, 6 seeds each): tcn+d9+p0.3+s60000 +20.34 (RRMSE 0.165, CC 0.979) | tcn+d9+w384+p0.3+s60000 +20.98 (RRMSE 0.157, CC 0.980) | ens_all +20.72 (RRMSE 0.159, CC 0.980) |  (r7 d9 x5 19.77)
Decision (r8 emg + W1): centered score still rising (+1.2 dB/round) but W1 shows that under the whole protocol full context adds only ~1.2 dB over d5 TCN for EMG/both; the centered headroom is clean-context interpolation. Stop chasing the centered score after r8; final round W2 = strongest recipe on the WHOLE protocol.
## Round W2 (WHOLE protocol, aug 12, 4000/500): tcn+d9+p0.3+s40000 and tcn+d9+w384+p0.3+s60000, 3 seeds + ens_all  [fly-whole12-eog/emg/both submitted 23:37]
### Round 8 result, both all-rec aug12 (job 9254953, 6 seeds each): tcn+d9+p0.3+s60000 +19.84 (RRMSE 0.171, CC 0.978) | tcn+d9+w384+p0.3+s60000 +20.78 (RRMSE 0.161, CC 0.979) | ens_all +20.33 (RRMSE 0.164, CC 0.979) |  (r7 d9 x5 19.33)
### Round 8 result, eog all-rec aug12 (job 9254951, 6 seeds each): tcn+d9+p0.3+s60000 +20.60 (RRMSE 0.153, CC 0.984) | tcn+d9+w384+p0.3+s60000 +21.60 (RRMSE 0.143, CC 0.985) | ens_all +21.14 (RRMSE 0.146, CC 0.985) |  (r7 d9 x5 20.26)
Round 8 summary (best tcn+d9+w384+p0.3+s60000 x6): eog +21.60 | emg 20.98 | both 20.78 (r7: 20.26 / 19.77 / 19.33) -> CENTERED ITERATION CLOSED; awaiting W2 (whole protocol) for the honest final.
### W2 result, eog WHOLE aug12 (job 9259256, 3 seeds; SNR gain): tcn+d9+p0.3+s40000 +15.90 (RRMSE 0.234, CC 0.968) | tcn+d9+w384+p0.3+s60000 +17.13 (RRMSE 0.208, CC 0.973) | ens_all +16.71 (RRMSE 0.216, CC 0.972) |  (W1 aug4 d9 +15.04)
### W2 result, emg WHOLE aug12 (job 9259257, 3 seeds; SNR gain): tcn+d9+p0.3+s40000 +10.42 (RRMSE 0.419, CC 0.899) | tcn+d9+w384+p0.3+s60000 +11.43 (RRMSE 0.379, CC 0.915) | ens_all +11.05 (RRMSE 0.393, CC 0.910) |  FIR SNR gain +7.24 dB (W1 aug4 d9 +10.05)
### W2 result, both WHOLE aug12 (job 9259258, 3 seeds; SNR gain): tcn+d9+p0.3+s40000 +10.57 (RRMSE 0.417, CC 0.900) | tcn+d9+w384+p0.3+s60000 +11.54 (RRMSE 0.378, CC 0.917) | ens_all +11.18 (RRMSE 0.391, CC 0.912) |  FIR SNR gain +7.39 dB (W1 aug4 d9 +10.13)

## CLOSED 2026-09-23 03:10 NZST
Stop reason: centered-protocol score still rising ~1 dB/round at r8, but W1/W2 showed that headroom is clean-context interpolation
(artifact only in the middle third). Iteration closed after r8 (centered) + W2 (whole protocol, the honest benchmark).
Final table: iter/FINAL_TABLE.md; plots: iter/plots/fig1_per_level, fig2_progression, fig3_levers (PDF+PNG).
Centered (SPAR-EEG score): EOG 21.60 / EMG 20.98 / both 20.78 dB (SPAR-EEG 8.28 / 9.05 / 8.00; round 0 was 11.70 / 10.50 / 9.88).
Whole protocol (SNR gain): EOG +17.13 / EMG +11.43 / both +11.54 dB (FIR +7.9 / +7.2 / +7.4; v1 fly ridge +10.4 / +6.1 / +6.2).
What mattered, in order: learned temporal readout (TCN) > augmentation (x4) > receptive field (d7-d9) > width/steps/seeds; extra distinct records ~0.
Not done: shuffled-wiring control with the TCN readout (needed before any connectome-specific claim), brain tuning (--train neuron), a single readout over all artifacts.
Note: results_*_centered.npz and results_*.npz in fly-eeg/ were overwritten by the latest brain runs (aug12 all-rec centered; aug12 whole); round-0 numbers live in states_*_centered/readouts.npz and this log.
