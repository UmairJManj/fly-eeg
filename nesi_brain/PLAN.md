# TARGET (user, 2026-09-24 17:40): the fly brain with a FIXED wire must beat the published deep denoisers, not the filter.
Leak-free bar (complex CNN, 3 seeds): EOG 16.2 / EMG 11.4 / mixed 11.5 dB. Brain today: 7.5 / 4.7 / 4.5. Levers in order:
training budget (aug12, long, fast crops) -> output bandwidth (lag 8, fast readout neurons) -> 3-brain average -> curriculum.
Every round reports against 16.2 / 11.4 / 11.5. "Point 2" (reservoir + TCN) is only the comparison table: the brain adds nothing there.

# FINDINGS SO FAR (updated 2026-09-24 08:10) -- read this first

**Track A: the brain itself cleans (fixed random +-1 wire over 1,421 descending/motor neurons; leak, bias, synapse gains, JO input gains trained by BPTT).** EOG whole-epoch protocol, test set, SNR gain dB (references: untrained brain + fixed wire -3.2; untrained brain + fitted ridge +6.0; FIR filter +8.1; TCN decoder +17.1-17.6):
- shuffled wiring, recipe lr 1e-3: **+7.66** (CC 0.83)
- shuffled wiring, recipe D (lr 3e-3, bias lr 3e-4, warm-up, cosine): **+8.57** (CC 0.86) -> beats the linear filter, paired +0.48 dB [+0.32, +0.66]
- REAL fly wiring, recipe D: **+7.51 test** (CC 0.82, plain) and +7.31 (truncated-BPTT variant), both done; E_aug4 (4x data) +5.7 @ pass 5 and climbing faster
- muscle (EMG), real wiring, D: **+4.70** final (FIR +7.2); mixed (EOG+EMG), real wiring, D: **+4.54** final (FIR +7.3)
- Retired recipes: lr 1e-2 (diverges), bias lr 3e-3 (unstable), all-neuron wire (slower), old lr 1e-3 real wiring (+1.1, stuck)
- Open: real wiring lags shuffled by ~1-2 dB at equal passes under the same recipe; round E (aug x4, wider leak range, gain weight-decay) queued/running.

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
| EEGDiR | SS2016 (in-domain) | running |
| retuned fly brain trained on SS2016 | SS2016 | queued (eog_SS2016_fast) |
Reading: the deep model loses ~8.5 dB when moved to another recording set; the retuned brain loses nothing. In absolute terms the in-domain deep models still lead. (job 9299952; cmp/results/ss2016_zeroshot.json)

**Track B, LEAK-FREE (disjoint artifact pools; the numbers to report), test SNR gain dB eye / muscle / mixed, 3 seeds:**
| method | EOG | EMG | EOG+EMG |
|---|---|---|---|
| FIR (linear) | ~8.1 | ~7.5 | ~7.3 |
| fly brain, fixed wire (real wiring; shared-split runs so far) | 7.5-7.8 | 4.7 | 4.5 |
| simple CNN (EEGdenoiseNet-style, 60k) | 14.6 | 8.4 | 8.2 |
| complex CNN (EEGdenoiseNet-style, 60k) | 16.2 | 11.4 | 11.5 |
| TCN d9 w384 60k (ours, no brain) | **17.6** | **11.5** | 11.1 |
| EEGDiR (authors' code, 2024), FCNN, LSTM, transformer | running (jobs 9299607-9) | | |
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
Track A (the user's core ask): the BRAIN ITSELF cleans (fixed output wire, brain parameters trained). Iterate the recipe until it
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
  TCN decoder +17.1 | shuffled brain trained (lr 1e-3) +6.8 @ pass 17 and rising | real wiring same recipe +1.1 (stuck) |
  lr 1e-2 diverges; lr 3e-3 unstable early (bias drift) -> round D = bias lr 3e-4 + warm-up + cosine.
retired round C (bias lr 3e-3 unstable): tbptt128 -4.9 dB @3, outall/tbptt64/lr3e-3 not started; lanes go to D family
Track C prep 2026-09-23 18:10: cmp/.venv (py3.12: mne, pygedai, pandas, mat73) ready; cmp/art = ART repo + checkpoints (ART, ICUNet_attn) downloaded to cmp/art/checkpoints/model/. Next: pick a multichannel ground-truth dataset (ART's ICA-synthetic pairs or GEDAI's simulation), write cmp/run_compare.py.
Track C job 9277839 (make_dataset + run_compare: none/ICA+ICLabel/ASR/GEDAI at 8/16/32/64 ch, EEGBCI S1-4)
18:31 Round D pass 1 on REAL wiring: biaslow +2.89 dB (CC 0.40), biaslow_tbptt128 +2.61 (CC 0.36), stable -> bias-lr was the instability; old recipe real wiring +1.1 @ pass 8.
18:51 Track B/C plumbing: fly_eeg_readout_v2.py --save-model; fly-eeg/fly_apply.py (brain+TCN on continuous channels, Hann overlap-add); nesi_fly_tcn_save.sl jobs 9277962 (eog) 9277963 (emg) 9277964 (both) on L4 -> models in nobackup/Fly/tcn_models/<art>/. Then: fly_apply on cmp_data/S*.npz (GPU) and re-run run_compare.py to add the 'fly' row.
18:53 retired eog_D_biaslow_wireall (+1.1 @3, CC 0.37 < DN wire); DN wire (1421 neurons) is the readout of choice
18:58 retired eog_whole_random (old recipe lr1e-3 real wiring, +1.2 dB @ pass 9; superseded by D) + takeover lanes
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
02:20 DONE eog_D_biaslow_shuffled: TEST +8.57 dB CC 0.856 RRMSE 0.514 (beats FIR +8.09 / CC 0.838; r0 ridge +6.0; old recipe shuffled +7.66). Brain-only cleaner now > linear filter.
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
