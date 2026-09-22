# Fly EEG cleaner: iteration campaign, final table

Score = SPAR-EEG region dSNR (mean over 26 input-SNR levels of per-level medians), dB. RRMSE / CC are whole-epoch means.

| round | data | best readout | EOG dSNR / RRMSE / CC | EMG dSNR / RRMSE / CC | EOG+EMG dSNR / RRMSE / CC |
|---|---|---|---|---|---|
| r0 | 2400 rec x1 | mlp 5-shift 6k (v1) | 11.70 / 0.421 / 0.899 | 10.50 / 0.440 / 0.888 | 9.88 / 0.470 / 0.874 |
| r1 | 2400 rec x1 | tcn | 13.66 / 0.309 / 0.943 | 11.30 / 0.386 / 0.912 | 11.00 / 0.396 / 0.909 |
| r2a | 2400 rec x1 | tcn+d6+s20000 | 14.06 / 0.291 / 0.949 | 11.91 / 0.359 / 0.923 | 11.56 / 0.372 / 0.919 |
| r2b | 2400 rec x4 | tcn+s20000 | 15.41 / 0.259 / 0.957 | 12.12 / 0.351 / 0.927 | 11.80 / 0.358 / 0.925 |
| r3 | 2400 rec x4 | tcn+d7+p0.3+s20000 | 16.80 / 0.215 / 0.970 | 15.78 / 0.238 / 0.963 | 15.72 / 0.243 / 0.962 |
| r4 | 2400 rec x12 | tcn+d6+p0.3+s30000 | 16.58 / 0.230 / 0.965 | 13.95 / 0.281 / 0.952 | 13.75 / 0.293 / 0.948 |
| r5 | 2400 rec x12 | tcn+d8+p0.3+s30000 | 19.48 / 0.178 / 0.978 | 18.85 / 0.196 / 0.971 | 18.52 / 0.203 / 0.969 |
| r6 | all rec x12 | tcn+d7+p0.3+s30000 | 18.50 / 0.188 / 0.976 | 17.32 / 0.203 / 0.973 | 17.00 / 0.213 / 0.971 |
| r7 | all rec x12 | tcn+d9+p0.3+s40000 | 20.26 / 0.159 / 0.982 | 19.77 / 0.170 / 0.979 | 19.33 / 0.178 / 0.977 |
| r8 | all rec x12 | tcn+d9+w384+p0.3+s60000 | 21.60 / 0.143 / 0.985 | 20.98 / 0.157 / 0.980 | 20.78 / 0.161 / 0.979 |

## Final best per artifact

| artifact | round | data | readout | region dSNR | RRMSE | CC | SPAR-EEG | FIR (65 taps) |
|---|---|---|---|---|---|---|---|---|
| EOG | r8 | all rec x12 | tcn+d9+w384+p0.3+s60000 | 21.60 | 0.143 | 0.985 | 8.28 | 8.31 |
| EMG | r8 | all rec x12 | tcn+d9+w384+p0.3+s60000 | 20.98 | 0.157 | 0.980 | 9.05 | 7.37 |
| EOG+EMG | r8 | all rec x12 | tcn+d9+w384+p0.3+s60000 | 20.78 | 0.161 | 0.979 | 8.00 | 7.20 |

## Honesty check: EEGdenoiseNet whole-epoch protocol (artifact everywhere, no clean context)

| artifact | FIR (65 taps) | fly ridge (v1) | ridge5, x4 | tcn d5, x4 | tcn d9, x4 | tcn d9, x12 | tcn d9 w384 60k, x12 (final) |
|---|---|---|---|---|---|---|---|
| EOG | +7.87 dB | +10.36 dB | +11.15 dB / 0.387 / 0.918 | +13.05 dB / 0.319 / 0.942 | +15.04 dB / 0.258 / 0.961 | +15.90 dB / 0.234 / 0.968 | +17.13 dB / 0.208 / 0.973 |
| EMG | +7.24 dB | +6.10 dB | +7.40 dB / 0.581 / 0.816 | +8.84 dB / 0.499 / 0.859 | +10.05 dB / 0.438 / 0.891 | +10.42 dB / 0.419 / 0.899 | +11.43 dB / 0.379 / 0.915 |
| EOG+EMG | +7.39 dB | +6.22 dB | +7.36 dB / 0.592 / 0.808 | +8.89 dB / 0.504 / 0.855 | +10.13 dB / 0.442 / 0.887 | +10.57 dB / 0.417 / 0.900 | +11.54 dB / 0.378 / 0.917 |

Cells: SNR gain / RRMSE / CC (whole epoch). The centered-protocol scores above are larger partly because the artifact occupies only the middle third there, so a whole-epoch readout also uses the clean outer thirds.
