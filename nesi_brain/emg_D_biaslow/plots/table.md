# EMG (whole), brain-trained, readout=random, shuffle=False

| method | SNR gain (dB) | RRMSE | CC |
|---|---|---|---|
| FIR + ridge (no brain) | +7.47 | 0.584 | 0.815 |
| untrained brain fixed readout | -2.89 | 1.952 | -0.062 |
| TRAINED brain fixed readout | +4.70 | 0.790 | 0.623 |

best pass 30, 2,937,265 brain parameters, 0.99 h
