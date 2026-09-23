# EOG (whole), brain-trained, readout=random, shuffle=True

| method | SNR gain (dB) | RRMSE | CC |
|---|---|---|---|
| FIR + ridge (no brain) | +8.09 | 0.543 | 0.838 |
| untrained brain fixed readout | -3.23 | 2.014 | -0.019 |
| untrained brain ridge readout | +5.98 | 0.687 | 0.727 |
| TRAINED brain fixed readout | +7.66 | 0.568 | 0.825 |

best pass 28, 2,937,265 brain parameters, 1.71 h
