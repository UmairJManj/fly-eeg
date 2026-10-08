# TARGET (user, 2026-09-24 17:40): the fly brain with a FIXED wire must beat the published deep denoisers, not the filter.
Leak-free bar (EEGDiR 2024, authors' code, 3 seeds): EOG 17.4 / EMG 13.1 / mixed 12.7 dB (complex CNN 16.2 / 11.4 / 11.5). Brain today: 7.5 / 4.7 / 4.5. Levers in order:
training budget (aug12, long, fast crops) -> output bandwidth (lag 8, fast readout neurons) -> 3-brain average -> curriculum.
Every round reports against 16.2 / 11.4 / 11.5. "Point 2" (reservoir + TCN) is only the comparison table: the brain adds nothing there.

# FINDINGS SO FAR (updated 2026-09-24 08:10) -- read this first

**Track A: the brain itself cleans (fixed random +-1 wire over 1,421 descending/motor neurons; leak, bias, synapse gains, JO input gains trained by BPTT).** EOG whole-epoch protocol, test set, SNR gain dB (references: untrained brain + fixed wire -3.2; untrained brain + fitted ridge +6.0; FIR filter +8.1; TCN decoder +17.1-17.6):
- shuffled wiring, methodology lr 1e-3: **+7.66** (CC 0.83)
- shuffled wiring, methodology D (lr 3e-3, bias lr 3e-4, warm-up, cosine): **+8.57** (CC 0.86) -> beats the linear filter, paired +0.48 dB [+0.32, +0.66]
- REAL fly wiring, methodology D: **+7.51 test** (CC 0.82, plain) and +7.31 (truncated-BPTT variant), both done; E_aug4 (4x data) +5.7 @ pass 5 and climbing faster
- muscle (EMG), real wiring, D: **+4.70** final (FIR +7.2); mixed (EOG+EMG), real wiring, D: **+4.54** final (FIR +7.3)
- Retired methodologies: lr 1e-2 (diverges), bias lr 3e-3 (unstable), all-neuron wire (slower), old lr 1e-3 real wiring (+1.1, stuck)
- Open: real wiring lags shuffled by ~1-2 dB at equal passes under the same methodology; round E (aug x4, wider leak range, gain weight-decay) queued/running.

**Track B correction: the reservoir adds nothing to the decoder.** TCN alone on the raw signal: EOG +17.56 / EMG +11.58 / mixed +11.69 dB vs brain+TCN +16.99 / +11.43 / +11.63. The +17 dB headline is a TCN result. Only Track A is fly-specific.

**Track B: single-channel SOTA table (EEGdenoiseNet whole protocol, SNR U(-7,2), aug x12 training, 3 seeds; test SNR gain dB, eye / muscle / mixed):**
| method | EOG | EMG | EOG+EMG |
|---|---|---|---|
| noisy input | 0 | 0 | 0 |
| FIR 65-tap + ridge (linear) | 8.1 | 7.5 | 7.3 |
| **fly brain, fixed wire (real wiring)** | **7.5** | **4.7** | **4.5** |
| fly brain, fixed wire (shuffled wiring) | 8.6 | - | - |
| FCNN (EEGdenoiseNet-style) | 9.6 | 9.1 | 9.5 |
| LSTM (EEGdenoiseNet-style) | 11.8 | 8.1 | 8.2 |
| simple CNN (EEGdenoiseNet-style) | 14.6 | **11.9** | **14.6** |
| complex CNN (EEGdenoiseNet-style) | 15.4 | 11.0 | 11.3 |
| transformer (EEGDnet-style) | 14.5 | 8.9 | 9.4 |
| TCN d9 w384 (ours, no brain) | **17.6** | 11.6 | 11.7 |
| fly reservoir + TCN | 17.0 | 11.4 | 11.6 |
Reading: the deep learned decoders are far ahead of the brain-only model; among them our TCN is best on blinks, the simple CNN on muscle/mixed. The brain-only model's value is scientific (a fixed connectome learning the task), not SOTA performance. (jobs 9295965-67; baselines saved in tcn_models/baselines_<art>)

**INDEPENDENT DATASET #2 (Klados SS2016 EEG/EOG, 13,620 test windows), SNR gain dB:**
| method | trained on | SS2016 test |
|---|---|---|
| FIR fitted on SS2016 | SS2016 | 9.5 |
| TCN d9 (3 seeds) | EEGdenoiseNet (zero-shot) | 9.0-9.2 (home score 17.6) |
| **retuned fly brain (eog_D_biaslow)** | EEGdenoiseNet (zero-shot) | **7.50 (home score 7.51: no transfer loss)** |
| complex CNN | SS2016 (in-domain) | 14.8 |
| simple CNN | SS2016 (in-domain) | 11.7 |
| EEGDiR | SS2016 (in-domain) | 16.1 |
| complex CNN (leak-free, 3 seeds) | EEGdenoiseNet (zero-shot) | 7.7-7.8 (home 16.2) |
| simple CNN (leak-free, 3 seeds) | EEGdenoiseNet (zero-shot) | 7.3-7.6 (home 14.6) |
| EEGDiR (leak-free, 3 seeds) | EEGdenoiseNet (zero-shot) | 7.3-7.4 (home 17.4) |
| retuned fly brain (E_aug4 / D / D_tbptt128) | EEGdenoiseNet (zero-shot) | 7.77 / 7.50 / 7.52 (home 7.98 / 7.51 / 7.31) |
| retuned fly brain trained on SS2016 | SS2016 | queued (eog_SS2016_fast) |
Reading (2026-09-25, jobs 9299952 + 9306947): every deep model loses 8-10 dB on the unseen recording set; the retuned brain loses nothing and ZERO-SHOT it MATCHES OR BEATS EEGDiR / both CNNs (7.5-7.8 vs 7.3-7.8); only our TCN (9.0-9.4) and the in-domain models stay ahead. Transfer robustness is the fly-specific claim. (cmp/results/ss2016_zeroshot.json + ss2016_zeroshot_sota.json)

**Track B, LEAK-FREE (disjoint artifact pools; the numbers to report), test SNR gain dB eye / muscle / mixed, 3 seeds:**
| method | EOG | EMG | EOG+EMG |
|---|---|---|---|
| FIR (linear) | ~8.1 | ~7.5 | ~7.3 |
| fly brain, fixed wire (real wiring; shared-split runs so far) | 7.5-7.8 | 4.7 | 4.5 |
| simple CNN (EEGdenoiseNet-style, 60k) | 14.6 | 8.4 | 8.2 |
| complex CNN (EEGdenoiseNet-style, 60k) | 16.2 | 11.4 | 11.5 |
| TCN d9 w384 60k (ours, no brain) | **17.6** | **11.5** | 11.1 |
| **EEGDiR (authors' code, 2024, 20k steps)** | **17.4** | **13.1** | **12.7** |
| FCNN (leak-free, 20k) | 9.9 | 9.2 | 9.1 |
| LSTM (leak-free, 20k) | 11.8 | 8.3 | 8.3 |
| transformer, EEGDnet-style (leak-free, 20k) | 14.3 | 8.6 | 8.3 |
Shared-split numbers that were inflated by template memorisation: simple CNN EMG 12.7 -> 8.4, mixed 15.9 -> 8.2. TCN and complex CNN unchanged.

**Track C: multichannel benchmark (EEGBCI 64 ch, real EOG/EMG injected, 4 subjects x 3 SNR), pooled SNR gain dB @ 8 / 64 channels; alpha-effect error in brackets:**
| method | EOG | EMG | EOG+EMG |
|---|---|---|---|
| ICA + ICLabel | 14.5 / 14.6 [<=1%] | 4.4 / 14.4 [<=1%] | 4.3 / 13.6 [<=1%] |
| ASR | 8.2 / 1.0 [<=7%] | 13.7 / 1.7 | 9.9 / 3.0 |
| GEDAI | 6.9 / 5.9 [50-80%] | 12.0 / 6.8 | 9.9 / 5.8 |
| TCN single-channel (no brain) | 8.1 / 4.2 [10%] | 12.8 / 6.3 [11%] | 10.7 / 5.9 [6%] |
| fly + TCN (subset of conditions) | 7.7 / 5.9 [4%] | 12.2 / 7.0 [11%] | 10.8 / 7.0 [9%] |
Story: with few channels (<=16) the single-channel deep cleaner wins on muscle and mixed artifacts; with many channels ICA wins everywhere and on blinks always; ASR is the best 8-16 ch muscle cleaner; GEDAI suppresses genuine strong alpha. The EEGdenoiseNet-trained TCN drops from +17.6 to ~8 dB on blinks on this other dataset (domain shift) -- a real limitation to report. Files: cmp/results/summary_v3_all_methods.md, cmp/results/plots/figC1-2, tableC.md.

---
# Fly-brain EEG cleaner: standing campaign (user direction 2026-09-23 17:50)

Goal: a state-of-the-art EEG cleaner "for all kinds of scenarios" built from the fly brain, run unattended.
Track A (the user's core ask): the BRAIN ITSELF cleans (fixed output wire, brain parameters trained). Iterate the methodology until it
  beats the filter (+8.1 dB EOG whole), then the decoder (+17.1), on real wiring; shuffled control alongside.
Track B (the SOTA candidate today): fly reservoir + TCN decoder. Bound it (no-brain control), then compare against the
  published single-channel deep denoisers (EEGDnet / DenoiseFormer / 1D-ResCNN / SPAR-EEG) on identical splits.
Track C (scenarios): multi-channel datasets with ground truth (8/16/32/64 ch); unsupervised spatial tools ICA+ICLabel, ASR,
  GEDAI (pygedai) and multichannel deep ART; fly run channel-wise. Two scores: dB gain AND preservation of a known effect.
Track D: one cleaner for all artifact types (eog+emg+both mixed training), real recordings judged by downstream tasks.

Mechanics: arms = lines in brain/units.txt; lanes = fly-eeg/nesi_fly_lanes.sl (claim/resume/idle); keeper =
  fly-eeg/lane_keeper.sh in a Monitor loop (reports + refills 1 debug + 3 normal lanes). Checkpoints on nobackup via symlinks.
Status log: see the keeper output / brain/*/history.json; results table per arm in brain/<tag>/plots/table.md.

Findings so far (EOG, whole protocol, dB gain): untrained brain + fixed wire -3.1 | + ridge (no input skip) +6.1 | FIR +8.1 |
  TCN decoder +17.1 | shuffled brain trained (lr 1e-3) +6.8 @ pass 17 and rising | real wiring same methodology +1.1 (stuck) |
  lr 1e-2 diverges; lr 3e-3 unstable early (bias drift) -> round D = bias lr 3e-4 + warm-up + cosine.
retired round C (bias lr 3e-3 unstable): tbptt128 -4.9 dB @3, outall/tbptt64/lr3e-3 not started; lanes go to D family
Track C prep 2026-09-23 18:10: cmp/.venv (py3.12: mne, pygedai, pandas, mat73) ready; cmp/art = ART repo + checkpoints (ART, ICUNet_attn) downloaded to cmp/art/checkpoints/model/. Next: pick a multichannel ground-truth dataset (ART's ICA-synthetic pairs or GEDAI's simulation), write cmp/run_compare.py.
Track C job 9277839 (make_dataset + run_compare: none/ICA+ICLabel/ASR/GEDAI at 8/16/32/64 ch, EEGBCI S1-4)
18:31 Round D pass 1 on REAL wiring: biaslow +2.89 dB (CC 0.40), biaslow_tbptt128 +2.61 (CC 0.36), stable -> bias-lr was the instability; old methodology real wiring +1.1 @ pass 8.
18:51 Track B/C plumbing: fly_eeg_readout_v2.py --save-model; fly-eeg/fly_apply.py (brain+TCN on continuous channels, Hann overlap-add); nesi_fly_tcn_save.sl jobs 9277962 (eog) 9277963 (emg) 9277964 (both) on L4 -> models in nobackup/Fly/tcn_models/<art>/. Then: fly_apply on cmp_data/S*.npz (GPU) and re-run run_compare.py to add the 'fly' row.
18:53 retired eog_D_biaslow_wireall (+1.1 @3, CC 0.37 < DN wire); DN wire (1421 neurons) is the readout of choice
18:58 retired eog_whole_random (old methodology lr1e-3 real wiring, +1.2 dB @ pass 9; superseded by D) + takeover lanes
19:15 Track C v1 result (ICA+ICLabel only, others failed): EOG +13.2/13.6/10.2/14.1 dB @8/16/32/64ch; EMG +4.8/10.0/8.4/11.0; both +4.3/6.9/8.8/12.5; alpha-ratio err <=1.3%. Fixed asrpy numpy-2 bugs + GEDAI lead-field; rerun job 9278643
19:53 Track C: asrpy diverges (1e14) on 32ch -> meegkit ASR; primary metric now POOLED SNR gain (per-channel mean punished any change on the many near-clean channels: GEDAI -9.4 per-ch vs +5.0 pooled); rerun job 9278921
20:38 DONE eog_whole_random_shuffled (lr 1e-3, 30 passes): TEST +7.66 dB CC 0.825 RRMSE 0.568 (untrained fixed -3.2; r0 ridge +6.0; FIR +8.1; TCN +17.1)
20:52 Track C v2 (pooled gain, 4 subj x 3 SNR): ICA+ICLabel eog 14.5/15.4/13.0/~15 @8/16/32/64ch, emg 4.4/9.0/10.4/14.4, both 4.3/5.7/10.4/13.6; ASR emg 13.7/13.9/10.9/2.1, eog 8.2/7.6/6.7/11.6, both 9.9/7.9/3.6/3.0; GEDAI 5-12 dB but alpha-ratio err 0.5-0.8 (settings/leadfield check needed). Table: cmp/results/summary_v2_spatial_only.md
20:53 GEDAI check: on CLEAN base it removes 16% of power and halves the EC/EO alpha ratio (26.6 -> 14.8): lead-field prior suppresses strong occipital alpha -> its alpha-ratio error is genuine, quote as a limitation (denoising_strength only accepts auto*/numeric). fly-apply job 9279544 chained after TCN-save.
21:20 D plain pass 6: CC 0.66 but gain +2.8 (amplitude wobble), synapse gains up to 57x -> queued eog_E_wd (wd 1e-3 on log-gains)
21:50 tcn-save eog DONE: test +16.99 dB CC 0.973 (reproduces +17.13); 3 models in nobackup/Fly/tcn_models/eog. emg/both on seed 2; fly-apply 9279544 chained.
22:27 tcn-save ALL DONE and reproduced: eog +16.99 / emg +11.43 / both +11.63 dB; 9 models in nobackup/Fly/tcn_models/{eog,emg,both}. fly-apply 9279544 now queued (GPU).
23:20 D shuffled pass 15 = +8.09 dB (matches FIR); D real wiring +5.9 @ pass 9 (both arms)
23:21 moved to L4: fly-apply 9283201, nobrain 9283202
01:22 fly-apply too slow (~20 h full scope) -> resubmitted 9284708: subjects 1-2, 0 dB only, batch 256
02:20 DONE eog_D_biaslow_shuffled: TEST +8.57 dB CC 0.856 RRMSE 0.514 (beats FIR +8.09 / CC 0.838; r0 ridge +6.0; old methodology shuffled +7.66). Brain-only cleaner now > linear filter.
02:22 *** NO-BRAIN CONTROL (job 9283202): TCN d9 w384 60k on RAW input, EOG whole aug12, 3 seeds -> +17.56 dB RRMSE 0.202 CC 0.973 vs brain+TCN +16.99/+17.13. THE RESERVOIR ADDS NOTHING under a strong decoder. Track B headline must be reported as 'TCN single-channel denoiser'; fly-specific claim = Track A only. (log: logs/fly-nobrain-9283202.log; states cache nobackup/Fly/states_eog_whole_aug12_nobrain)
02:22 submitted no-brain TCN (with --save-model) for eog/emg/both -> tcn_models/nobrain_<art>; fly_apply.py now handles K==1 (no brain) models
04:53 nobrain eog saved: +17.56 dB, 3 models in tcn_models/nobrain_eog
04:53 tcn_apply.sl job 9286294 chained: no-brain TCN on all 4 subjects x 9 conditions -> 'tcn_nobrain' row in Track C
06:08 NO-BRAIN CONTROL COMPLETE (all artifacts, 3 seeds, whole aug12): TCN alone eog +17.56 / emg +11.58 / both +11.69 vs brain+TCN +16.99 / +11.43 / +11.63 -> reservoir adds ~0 (slightly negative) everywhere. Models: tcn_models/nobrain_{eog,emg,both}. Track B headline = TCN single-channel denoiser.
07:23 DONE both_D_biaslow (real wiring): TEST +4.54 dB CC 0.618 (FIR +7.39, r0 ridge +6.22) -> brain-only below filter on mixed artifacts; eye D arms +7.3 @ pass 23; emg D +4.5 @ 21
07:24 TRACK C COMPLETE (cmp/results/summary_v3_all_methods.md, plots cmp/results/plots/figC1/figC2): single-channel TCN (with or without brain, equal) beats ICA at 8 ch for EMG (12.8 vs 4.4) and mixed (10.7 vs 4.3) but loses on EOG (8.1 vs 14.5); ICA wins at >=32 ch everywhere (64ch: 14.6/14.4/13.6); ASR best at 8-16 ch EMG (13.7-13.9); GEDAI 5-12 dB but wrecks alpha (50-80% err); TCN alters alpha 6-11%, ICA/ASR <=1%. EOG single-channel gain here (~8 dB) << EEGdenoiseNet (+17.6) -> big domain shift. fly_brain_tcn rows = S1 all + S2 eog @0 dB only.
09:10 DONE emg_D_biaslow (real wiring): TEST +4.70 dB CC 0.623 (FIR ~+7.2) -> brain-only below filter for muscle, like mixed
10:30 retired eog_E_alpha_wide (+3.1 @ pass 6 vs D +4.2): wider leak range hurts; next E_wd
10:56 DONE eog_D_biaslow_tbptt128 (REAL wiring): TEST +7.31 dB CC 0.808 RRMSE 0.590 (FIR +8.09; r0 +6.06); first real-wiring eye result
11:09 DONE eog_D_biaslow (REAL wiring, plain D): TEST +7.51 dB CC 0.816 RRMSE 0.577 (FIR +8.09) -- best real-wiring result
11:21 retired eog_E_wd (wd 1e-3: +3.5 @5, gains pinned); F_aug4_wd lowered to wd 1e-4
12:27 L4 lanes now skip --aug arms (nesi_fly_lanes_v2.sl); lane target 4; recycled L4 lane that had emg_F_aug4 (2.1 h/pass)
12:28 queued eog_D_seed1/2; submitted A100 lane 9294086 + H100 lane 9294087 for the aug arms
13:30 SOTA comparison: baselines jobs 9295965-67 (fcnn/scnn/ccnn/rnn/xfmr on raw, 3 seeds x 20k, eog/emg/both); brain-only on SPAR-EEG centered protocol queued (eog/emg/both_D_centered, region dSNR added to fly_eeg_brain.py)
13:32 user: do NOT compare with SPAR-EEG; centered arms blocked. SOTA comparators = fcnn/scnn/ccnn/rnn/xfmr (EEGdenoiseNet/EEGDnet-style), TCN, ICA+ICLabel/ASR/GEDAI
14:44 cnn60k: scnn eog 14.3 / emg 12.7 / both 15.9; ccnn 16.0 / 11.4 / 11.7 (TCN 17.6 / 11.6 / 11.7). CAVEAT: whole protocol shares artifact templates across splits -> memorisation possible (big dense heads). Added --artifact-split disjoint; leak-free rerun of scnn/ccnn/TCN submitted (nesi_fly_leakfree.sl)
14:51 SPEED-UP: --train-len 256 (1 s random crops), batch 64 on A100/H100 (32 on L4); retired eog_F_long60 (33 h on L4); round G aug12 fast arms queued; smoke_fast at top; tried a pro_6000 lane 9297413
14:58 big GPUs all busy: resumed eog_E_aug4 (pass 11, +7.7) on a forced L4 lane 9297556 (ALLOW_AUG=1); seed1 lane cancelled (resumes later)
15:58 LEAK CONFIRMED: disjoint artifact split -> scnn eog 14.6 / emg 8.4 / both 8.2 (shared: 14.3 / 12.7 / 15.9); ccnn 16.2 / 11.4 / 11.5 (unchanged). Memorising models were flattered on emg/both. TCN leak-free pending. Round H (brain-only, disjoint, fast) queued; ALL final numbers to be reported leak-free.
16:44 user asked age of comparators: single-channel deep rows are 2021-2023 re-implementations; adding 2024-2026 (EEGDiR, LRR-UNet, BandRouteNet) + ART multichannel
16:45 2024-2026 comparators: EEGDiR (2024, code: github woldier/EEGDiR, cloned to cmp/sota/EEGDiR) to port; LRR-UNet (2025, CNS Neurosci Ther, no public code) and BandRouteNet (2026 arXiv 2604.24428, no code found) -> re-implement from paper or mark as 'reported only'; note arXiv 2606.08594 (2026) argues EEGdenoiseNet is saturated + metric-utility gap -> cite
16:46 EEGDiR (authors' code, 2024) ported into readout v2 as 'eegdir'; leak-free SOTA jobs (eegdir 60k + fcnn/rnn/xfmr 20k) submitted for eog/emg/both
17:22 HEADLINE (user): 'We retuned the brain of a fly so that if we give EEG data as input we get cleaned EEG as output.'
17:27 USER FRAMING: compare the retuned fly brain vs recent DL EEG cleaners (not our TCN), validate on independent datasets. fly_brain_apply.py written; brain-apply job 9299930 (eog/emg/both D brains on EEGBCI S1-2 @0 dB) -> 'retuned_fly_brain' row in Track C. Next independent set: Klados SS2016 semi-simulated EEG/EOG (via EEGDiR HF dataset woldier/eeg_denoise_dataset)
17:29 SS2016 (Klados, via EEGDiR HF; x=clean, y=contaminated, 53540 train / 13620 test 512-sample windows): cmp/ss2016_export.py, cmp/zeroshot_eval.py, job 9299952 (zero-shot saved models + retuned brains, then in-domain ccnn/scnn/eegdir); brain arm eog_SS2016_fast queued (fly_eeg_brain --data-dir)
18:02 readout v2 now saves baseline models too (was TCN-only -> leakfree/cnn60k/sota dirs lack CNN/EEGDiR models); sotasave jobs chained (ccnn/scnn/eegdir leak-free, saved) for zero-shot on SS2016
19:02 brain-apply too slow for one job (~2 h/condition on L4) -> split: 9300920 (S1 emg/both), 9300921 (S2 all three); S1 eog already saved
19:44 leak-free single-channel table COMPLETE (EEGDiR 17.4/13.1/12.7; ccnn 16.2/11.4/11.5; xfmr 14.3/8.6/8.3; scnn 14.6/8.4/8.2; LSTM 11.8/8.3/8.3; FCNN 9.9/9.2/9.1; FIR ~8.1/7.5/7.3; brain 7.9*/5.7*/6.7* running)
10:33 9300636 (SS2016 zero-shot of saved ccnn/scnn/eegdir) TIMED OUT at 2 h after the brain rows; resubmitted as 9306947 (L4, 8 h, sota_eog + nobrain_eog) -> cmp/results/ss2016_zeroshot_sota.json
10:33 seed replicates real-wiring eye (shared split, D methodology): seed0 +7.51 / seed1 +4.07 (early-stopped pass 17) / seed2 +7.24 -> mean 6.3 +- 1.9 dB; H leak-free fast arms eog +4.68 / emg +4.30 (both running, val 3.9 @ pass 8); F_aug4_wd +4.68; E_aug4 (full-length, 9 h) +7.98 -> train-len 256 crops likely cost ~3 dB, not the disjoint split; needs a full-length disjoint arm
11:40 RESUMED (user: 'use this brain for eeg cleaning in a perfect way, continue'). Keeper Monitor re-armed (NORMAL_TARGET 5); L4 lanes now take fast-crop aug arms (only full-length aug arms are skipped); centered arms commented as BLOCKED. both_H_aug4_disjoint DONE +4.31 (leak-free fast trio: 4.68 / 4.30 / 4.31).
11:40 NEW LEVER 'sensory periphery': --jo-delays D in fly_eeg_brain.py (JO neuron i sees the input delayed by i mod (D+1) samples; brain + wire unchanged; loader patched). Smoke 9307276. Round J = jo-delays 16 / 48 x lag 24 / 48, leak-free fast. Also 4-brain average eval job 9307301 (cmp/brain_ensemble_eval.py -> cmp/results/brain_ensemble_eog.json). SS2016 sota zero-shot rerun 9306947.
11:45 SS2016 ZERO-SHOT COMPLETE: ccnn 7.7-7.8, scnn 7.3-7.6, EEGDiR 7.3-7.4, TCN 8.9-9.4, retuned brain 7.50-7.77 -> brain >= EEGDiR/CNNs out of domain, no transfer loss. Round J (jo-delays 16/48 x lag 24/48; emg/both jo16) queued.
11:55 4-brain average (job 9307301): D 7.51 / E_aug4 7.96 / seed2 7.24 / tbptt128 7.31 -> avg2 7.86, avg3 7.73, avg4 7.66 < best single 7.96. Brains make the SAME errors (systematic bandwidth limit, not variance) -> 'brain average' lever RETIRED. Lane 9297413 hit 12 h wall (arm was done), keeper replaced it with 9307657; ensemble json dump bug fixed.
12:40 two L4 lanes (9307263/64) idled 1 h: submitted before the lane-rule patch (old script skipped all aug arms) -> cancelled, keeper resubmitted with the new rule. eog_I_lag8_disjoint val +6.56 @ pass 6 (fast H arm was +4.4 @ pass 12): SHORT output lag helps a lot.
14:10 SCRUTINY (user asked 'are we on the right path'): brain sits at FIR level (~8) on every methodology; averaging adds 0; shuffled = real; lag8 > lag24 -> bottleneck is precision-through-time / causality / fixed wire, not budget. Diagnostics job submitted (cmp/brain_diag.py: fixed wire vs bidirectional vs ridge readout on the TRAINED brain); lag 0/4 arms queued at top; lag48 arms demoted.
14:11 CLUSTER CONGESTED (a100 149 pending, l4 12, h100 15, pro_6000 27): 3 of 5 lanes requeued to PD; debug slot held by lane 9292605 (lag8 arm at pass 10/12, resumes from ckpt). Diag job resubmitted normal L4 as 9310381. Nothing to do but wait; keeper armed.
15:26 DIAG RESULT (eog_E_aug4): fixed wire 7.98 | reversed 7.40 | bidirectional avg 8.01 | ridge readout on the TRAINED brain 8.50. Neither the fixed wire nor causality is the wall: the brain STATE carries only ~filter-level information about clean EEG (untrained+ridge 6.0 -> trained+ridge 8.5, TCN on raw 17.6). Bottleneck = internal representation / trainability. Round K (h-max 100, slow leaks, rho 1.2, input gain 0.3; all lag 8, leak-free fast) queued at top; J delay-line arms next.
15:39 SPEED-UP (user: 'how can we do it fast'): L4 = 9.7 s/step vs A100 2.9 -> L4 lanes capped at 2; keeper now spreads lanes over a100/h100/pro_6000/l4, NORMAL_TARGET 8. Round S SCREENING arms at top: 2000 epochs x 5 passes (~1 h on A100) for every K/J lever + control + wire-all + fast leaks; winners get full runs.
15:41 DIAG 2nd brain (eog_D_biaslow): fixed 7.51 | reversed 6.96 | bidir 7.50 | ridge readout 8.08 -> same pattern; confirmed: wire and causality are not the wall.
15:57 DONE eog_I_lag8_disjoint: TEST +7.44 (same fast leak-free methodology at lag 24 = +4.68) -> lag 8 = +2.8 dB, biggest single lever so far. DIAG emg_D_biaslow: fixed 4.70 | bidir 5.11 | ridge 6.21 -> muscle loses ~1.5 dB to the fixed wire, but still far below 13. Screening arms moved to top of units.txt (lanes were picking round I first).
16:56 SCREEN eog_S_ctrl = +4.58 dB (baseline for round S, ~1 h on A100)
2026-09-28 CATCH-UP (lanes idled since 09-27 14:38; queue empty). Finished since 09-25 17:00 (TEST dB, leak-free disjoint unless noted; FIR 7.99-8.28):
  round S screens (5 passes): ctrl 4.58 | alpha_fast 5.98 | jo16 5.69 | wireall 5.39 | jo48 5.19 | rho12 5.06 | hmax100 4.93 | lag0 4.40 | gain03 4.39 | alpha_slow 3.78
  round K full: rho12 7.71 | hmax100 7.16 | alpha_slow 6.77 | gain03 6.67 (lag8 ref 7.44) -> no K lever beats the ref
  round J full: *** eog jo48 9.69 dB CC 0.89 *** (FIRST leak-free brain > FIR 8.28, +1.4 dB) | jo16 7.39 | lag4 7.20 | lag0 6.45; emg jo16 6.15 (H 4.30) | both jo16 6.33 (H 4.31)
  round I: aug12_long (20 passes) 9.11 | fastdn wire 7.76; round G (shared): eog 7.05 / emg 6.50 / both 4.90; H_full (full-length disjoint) 6.20 -> crops were NOT the 3 dB cost
  eog_SS2016_fast (in-domain SS2016) 7.22 vs FIR 10.17
  Levers ranked: JO delay-line periphery (jo48) > more aug/passes > lag 8 > rest.
2026-09-28 ROUND L queued (top of units.txt): eog jo48+aug12 long, jo96, jo48+lag8, jo48+rho1.2, emg jo48, both jo48, eog jo48 shuffled control. Lanes 9350351 (debug) + 9350352-58.
2026-09-28 ROUND M = tweak the brain ITSELF (user). Rationale = DIAG 09-25: ridge on the TRAINED brain only 8.5 dB (TCN on raw 17.6) -> the neurons' internal representation is the wall, not wire/causality. Added BrainPlus to fly_eeg_brain.py (wiring + Dale signs untouched): --syn per-neuron synaptic filter (2nd-order / resonant neurons), --adapt spike-frequency adaptation (slow self-trace subtracted; high-pass), --slope per-neuron transfer gain. Smoke 9350452 (L4, jo48 + all three, batch 64); round M 5-pass screens on jo48 (ctrl/syn/adapt/slope/all) auto-enable when the smoke passes.
2026-09-28 REVIEW (user: "don't be hasty, review first"). Findings from all 44 finished arms:
 1. CONFOUND: lanes pick batch by GPU (L4 32, else 64) with a fixed pass count -> L4 arms got 2x the optimizer steps. The "winners"
    (jo48 9.69, lag8 7.44, rho12 7.71) all ran at 6000 steps; the arms they beat mostly ran at 3000. At ~3000 steps jo48's own val
    curve (6.6 @ pass 6) is no better than jo16 (7.1). jo48 "breakthrough" is NOT established; steps are a hidden lever.
 2. No overfitting anywhere (train RMSE = val RMSE); curves still rising until the cosine schedule ends -> underfitting / budget-limited.
    Most steps = best: aug12_long 15000 steps 9.11.
 3. Equal-step (3000) comparisons that ARE clean: fastdn wire 7.76, jo16 7.39, lag4 7.20, lag0 6.45 vs base lag24 4.68 -> readout
    from fast neurons, input delay line and short lag are real levers.
 4. 625-step screens (round S) did not predict full-run ranking -> retire short screens; judge at >= 6000 steps.
 5. Seed spread partly confounded too (seed1 4.07 had 1071 steps at batch 64). Real seed noise unknown -> replicate.
 6. Correction: full-length vs 1 s crops = 6.20 vs 4.68 (crops cost ~1.5 dB); shared vs disjoint split costs ~1.8 dB (E_aug4 7.98 vs H_full 6.20).
 7. Shuffled >= real wiring still stands (8.57 vs 7.51, D methodology). The fly-specific claim needs a matched shuffled control on the best methodology.
 8. DIAG: ridge on the trained brain only 8.5 -> representation is the wall; neuron-intrinsic tweaks (syn/adapt/slope) are the right next
    family, but must be tested at equal budget against replicated controls.
DECISION: rounds L and M superseded by ROUND N (every arm pins --batch 32, 6000 steps): jo48 seeds 1/2, matched base, jo16, jo48 shuffled,
 jo48 + syn / adapt / slope, jo48 + fastdn, emg/both jo48, jo48 aug12 long (30000 steps). units.txt backup: nobackup/Fly/units.txt.bak_20260928.
2026-09-28 12:10 trend watcher: fly-eeg/trend_watch.py (2-min Monitor) compares every round-N arm per pass to the jo48 ref (best-so-far mean of J_jo48 + N seeds); lever arms trailing by >1.5 dB at 2 consecutive passes from pass 5 get brain/<tag>/STOP -> fly_eeg_brain.py ends cleanly after the pass and still writes the test summary. Controls (seeds/base/shuffled/long/emg/both) never stopped.
12:20 backfill: +3 h lanes 9351440/42/43 (a100/h100/a100); L4 lanes resubmitted (rule: arms with pinned --batch run on L4 too)
14:37 ROUND N interim (equal budget, batch 32): jo48 s1 val 6.4/7.1/5.3/7.2/7.6/8.3 @p6 (orig jo48 6.6 @p6) -> jo48 replicating; s2 3.6/5.5/5.8 @p3 (slow start); base (no delay line) 3.2/3.4 @p2 -> delay line worth ~3 dB early.
16:13 DONE eog_N_jo48_s1 (seed 1, batch 32, 6000 steps): TEST +9.44 dB CC 0.880 RRMSE 0.473 (orig jo48 seed 0: 9.69) -> jo48 REPLICATES; > FIR.
16:47 DONE eog_N_jo48_s2 (seed 2): TEST +9.55 dB CC 0.884. jo48 3 seeds: 9.69 / 9.44 / 9.55 -> mean 9.56 +- 0.13 dB (FIR 8.28): REPLICATED, +1.3 dB over FIR.
18:14 adapt (jo48+adaptation) 8.6/8.7 @p4/p5 = +1.6 dB over jo48 ref; syn trailing (-0.8). Queued eog_N_jo48_adapt_s1 (seed replicate) ahead of slope.
19:04 syn -> 9.1 @p10 (= jo48); adapt 9.1 @p7-8 (faster, likely same ~9.2 ceiling). Reordered queue: fastdn + jo48_long (budget/readout ceiling tests) ahead of adapt_s1 / slope.
19:09 DONE eog_N_base (no delay line, equal budget): TEST +7.58 dB CC 0.815 (< FIR 8.28). jo48 mean 9.56 -> delay line worth +1.98 dB, and is what lifts the brain above FIR.
19:31 DONE eog_N_jo48_syn: TEST +9.60 dB CC 0.884 (jo48 3-seed 9.56+-0.13) -> syn = no gain (within seed noise).
19:48 SHUFFLED jo48 val 6.4 / 8.5 / 9.3 @p1-3 -- already at the real-wiring jo48 FINAL (9.2 val) by pass 3 (+2.6 vs real at p3). Random rewiring of the same neurons/synapses trains faster and at least as high -> 'fly wiring' is NOT the source of the gain.
20:05 DONE eog_N_jo48_adapt: TEST +9.92 dB CC 0.892 RRMSE 0.448 -- best brain-only result so far (jo48 3-seed 9.56+-0.13, i.e. +0.36 dB = ~2.8 sd; syn 9.60). adapt_s1 running to confirm.
22:51 DONE eog_N_jo48_adapt_s1: TEST +9.55 dB CC 0.883 -> adapt 2 seeds 9.92/9.55 = 9.74+-0.26 vs jo48 9.56+-0.13: NOT a reliable gain (seed 0 was lucky). Adaptation = faster learning, same ceiling.
23:18 DONE eog_N_jo48_shuffled: TEST +10.78 dB CC 0.909 RRMSE 0.410 vs real wiring jo48 9.56+-0.13 -> shuffled wiring +1.2 dB BETTER (~9 sd). The fly connectome is not what makes the cleaner; the gain is from delay-line input + retunable connectome-scale Dale network. Framing decision needed.
23:22 DONE eog_N_jo48_slope: TEST +9.96 dB CC 0.893 (single seed; best real-wiring single run, cf adapt s0 9.92 which did not replicate). jo16 6.4 @p6 (-1.6).
23:24 queued eog_N_jo48_slope_s1 + eog_N_jo48_shuffled_s1 (seed replicates of the two best eog results)
00:07 STOP eog_N_jo48_long (matched-step val 7.8@4800 steps vs jo48 9.2; more aug data does not break ceiling). Queued both_N_jo48_lrlo (lr/lr-edge 1e-3; mixed run has gradnorm ~150, val 3.0 @p4).
00:18 DONE (stopped) eog_N_jo48_long: TEST +8.08 dB CC 0.873 after 4800 steps (< FIR 8.28, < jo48 9.56). aug12 does not help at this budget. emg_N_jo48 val 6.8 @p5.
00:48 both_N_jo48_lrlo p1 5.8 vs lr3e-3 1.0 -> lr was the mixed-noise problem. emg stuck 6.5-6.8 (<FIR 7.25): queued emg_N_jo48_lrlo.
00:56 DONE eog_N_jo48_fastdn (REAL wiring, readout = fastest half of DN/motor): TEST +10.13 dB CC 0.897 -- best real-wiring result, first real-wiring brain > 10 dB (jo48 9.56+-0.13; shuffled 10.78). Next: fastdn seed replicate + fastdn+slope combo.
01:00 STOP both_N_jo48 (lr 3e-3, 5.3 @p7) -- superseded by both_N_jo48_lrlo (5.8 @p1); frees lane for fastdn_s1
2026-09-29 11:00 ROUND N OVERNIGHT RESULTS (test dB, eog FIR 8.28 / emg 7.25 / both 7.16):
  eog real jo48 9.44/9.55/9.69 = 9.56+-0.13 | base 7.58 | jo16 7.69 | long(stopped) 8.08
  eog +syn 9.60 | +adapt 9.92/9.55 = 9.74 | +slope 9.96/9.47 = 9.72 -> neuron tweaks: NO reliable gain
  eog +fastdn 10.13 (s1 running, weak so far) | SHUFFLED 10.78/10.43 = 10.61+-0.25 -> shuffled > real by ~1.0 dB, CONFIRMED on 2 seeds
  emg jo48 7.31 (=FIR, +0.06) | emg lrlo 7.18 ; both jo48 6.43 | both lrlo 7.08 (< FIR 7.16)
  Keeper was down 01:40-11:00 (classifier outage) -> only 1 lane running at 11:00.
12:50 INDEPENDENT TEST v2 (user: 'check it independently'): job 9374071 cmp/ss2016_v2.sl -> SS2016 zero-shot, EEGdenoiseNet-trained SOTA (EEGDiR/ccnn/scnn) + in-domain SS2016 SOTA + round-N brains (jo48 x3, fastdn, shuffled x2); metrics CC%/RRMSE_t/RRMSE_s/SNR (zeroshot_eval.py now adds RRMSE_s). Next: EEGBCI channel-wise vs SOTA, then more datasets.
12:52 dropped eog_N_fastdn_slope (user: forget combos). Debug slot -> SS2016 v2 after fastdn_s1.
12:57 SS2016 v2 resubmitted on pro_6000 as 9374115 (idle lane 9373404 cancelled to free the GPU)
13:00 no training arms left -> keeper set to NO_SUBMIT; pending idle lanes cancelled: 9350353 9350357 9351440 9351443 9373402 9373403 9374125 
14:00 SS2016 v2 DONE (zero-shot, trained on EEGdenoiseNet only): brains CC 77.1-78.7% SNR +9.2-9.7 dB vs SOTA zero-shot CC 61-67% SNR +7.3-7.8 (EEGDiR 62%/+7.4); brains win all 4 metrics. In-domain SS2016 EEGDiR 93.8%/+15.6 (upper bound). results cmp/results/ss2016_v2.json
14:05 MULTI-DATASET PLAN (user direction: fly brain vs published SOTA on many clean-ground-truth datasets; CC% + RRMSE_t/s + SNR):
  D1 EEGdenoiseNet EOG/EMG/mixed (done, leak-free)   D2 SS2016 Klados EEG/EOG (done: zero-shot brains 77-79% CC > SOTA 61-67%)
  D3 PhysioNet motion-artifact EEG (Sweeney, 23 pairs, real motion + clean reference)  -> downloading to nobackup/Fly/ext_data/motion
  D4 EEGdenoiseNet clean EEG + MIT-BIH ECG artifact (GCTNet/AnEEG protocol)          -> downloading ext_data/mitdb
  D5 EEGBCI multichannel semi-synthetic (ours)  -> brain + SOTA channel-wise
  Protocol per dataset: (a) zero-shot of EEGdenoiseNet-trained models where artifact type matches; (b) in-domain: brain (jo48 methodology) and SOTA (EEGDiR/ccnn/scnn) trained on identical splits.
14:10 DONE eog_N_jo48_fastdn_s1: fastdn_s1 TEST CC 87.1% SNR +9.08 dB RRMSE 0.492 -> fastdn 2 seeds; all lanes released.
16:20 ROUND X: export 9375770 -> bench 9375771 (motion zero-shot + in-domain SOTA on motion & ECG) + 2 lanes for ecg_X_jo48 / motion_X_jo48
19:40 export 9375770 FAILED (RECORDS includes fNIRS files) -> fixed (eeg_* only), resubmitted chain export 9376974 / bench 9376975 / 2 lanes
20:05 export 9376974 FAILED (EEG_tr channel constant; triggers are in .trigger WFDB annotations, code 6 start / 3 end) -> fixed, resubmitted export 9378220 / bench 9378221 / 2 lanes
20:32 bench split (PART=zs/motion/ecg) on h100 + 3 h lanes; slow a100/pro_6000 copies cancelled
21:05 D3 MOTION: in-domain SOTA ccnn 84.6%/+1.72, eegdir 81.7%/+1.04, scnn 81.1%/+0.37 (noisy 78.3%). ZERO-SHOT: all SOTA HURT (CC 56-70%, SNR -2.1..-3.8 dB); fly brains (eog-trained) CC 77.7-79.0% SNR -0.35..-0.69, shuffled 81.4%/+0.57 -> brains degrade far less. brain in-domain (motion_X_jo48) pending.
21:35 D4 ECG in-domain SOTA: EEGDiR 94.1%/+13.42, ccnn 93.2%/+12.64, scnn 90.6%/+11.69 (noisy 60.2%). ecg_X_jo48 brain val ~5 dB @p4 (running).
23:30 DONE ecg_X_jo48: TEST +7.38 dB CC 81.8% RRMSE 0.578 (EEGDiR in-domain 94.1%/+13.42; ccnn 93.2/+12.64; scnn 90.6/+11.69)
13:38 DONE motion_X_jo48: motion_X_jo48 TEST CC 78.5% SNR -0.41 dB RRMSE 0.767 (in-domain SOTA ccnn 84.6%/+1.72, eegdir 81.7%/+1.04; noisy 78.3%)
14:29 ROUND Y prepared (residual / bidir / filterbank8 / spec+cc loss / out-affine / trainable readout / all), gated on smoke_y 9393230. Review: SOTA sees full window (brain 94 ms lookahead), residual learning, trained head, time+freq loss, multi-band input, 20-60k steps.
15:52 smoke_y PASSED (all Y options, 6.2 GB); round Y enabled; debug lane 9396432
17:12 eog_Y_res val 8.5/9.3 @p1-2 (jo48 5.5/6.6) -> remaining Y levers now stacked on --residual
21:08 DONE eog_Y_res (residual): TEST +10.46 dB CC 90.4% RRMSE 0.424 vs jo48 9.56/88% -> +0.9 dB, +2.4 CC pts (single seed)
22:26 STOP eog_Y_res_fb8 (2.4/3.3 vs res 8.5/9.3): filterbank front-end hurts -> dropped; eog_Y_all replaced by eog_Y_all_nofb (res+bidir+loss+ro)
02:05 DONE eog_Y_res_bidir: TEST +10.90 dB CC 91.1% RRMSE 0.405 (best real-wiring so far; res alone 10.46/90.4; jo48 9.56/88)
04:00 DONE eog_Y_res_loss: TEST +10.43 dB CC 90.2% (= residual alone) -> spec+cc loss no SNR/CC gain
04:10 ROUND Z queued after Y: eog resbidir seed1 + long (24 passes), emg, both (lr 1e-3)
05:45 STOP eog_Y_res_ro (neutral), drop eog_Y_all_nofb -> lane goes to round Z
06:01 ROUND Y FINAL (eye, test): res 10.46/90.4 | res+bidir 10.90/91.1 (BEST) | res+loss 10.43/90.2 | res+ro 10.30/90.1 (stopped p5) | res+fb8 3.51/67.0. Winner methodology: residual + bidirectional.
08:30 L4 lanes too slow for bidir (21 s/step vs 6 on A100): cancelled L4 lane 9417910 (long), kept s1 on L4 9393239; order now emg(debug) -> both -> long
13:16 LINE NOISE added (user): synthetic 50 Hz +/-0.5 Hz + 100 Hz harmonic, AM; brain unit line_Z_resbidir queued; SOTA bench cmp/line_bench.sl submitted
13:46 smoke_diff PASSED (unroll 2 + wide SNR); eog_D_wide + eog_D_unroll2 enabled

## 2026-10-01 round W: rewire the brain (user: "who said its fixed ... fix it")
- `--new-edges 200000` (half JO->readout neurons, half random central; zero-init, free sign, learned in units of median |w0| = 6e-3),
  `--free-signs` (every synapse may flip E/I), `--envelope` (1/4 of JO neurons get the >30 Hz burst envelope).
- First smoke diverged (CC -0.12) because new synapses learned in absolute units; fixed by the median-|w0| unit. Re-smokes OK.
- Skipped: temporal output filter, wider readout (trainable readout was neutral -> readout is not the bottleneck).
- Arms emg_W_rewire_env / emg_W_rewire / emg_W_env on top of res+bidir (muscle ref 81.3%, EEGDiR 92.9%).
- eog_Z_resbidir_s1 STOPPED early to free the debug lane; full-budget seed repeat requeued as eog_Z_resbidir_s1b after W.
