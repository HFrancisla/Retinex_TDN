# Step 03 training dynamics

Eval rows parsed: `340`

A run is flagged when eval total loss decreases while the R consistency proxy drops by more than 10 dB.

## Potential loss/R-consistency mismatch

- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.05er_0.5smv1_20260720-121319`: total Δ=-0.0541, R proxy Δ=-16.94 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.15er_0.5smv1_20260720-135547`: total Δ=-0.0431, R proxy Δ=-36.13 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255`: total Δ=-0.0476, R proxy Δ=-31.07 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824`: total Δ=-0.0484, R proxy Δ=-28.89 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.7smv1_20260720-035020`: total Δ=-0.0486, R proxy Δ=-27.22 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.8smv1_20260720-053058`: total Δ=-0.0501, R proxy Δ=-25.62 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_1.0smv1_20260720-071136`: total Δ=-0.0506, R proxy Δ=-22.49 dB
- `LOLv2_1.0rh_0.3rl_0.003crh_0.003crl_0.1er_0.5smv1_20260720-153620`: total Δ=-0.0478, R proxy Δ=-31.16 dB
- `LOLv2_1.0rh_0.3rl_0.005crh_0.005crl_0.1er_0.5smv1_20260720-171701`: total Δ=-0.0478, R proxy Δ=-31.01 dB

Figure: `/home/ipr4090/2024_hzf/Retinex_TDN/_analysis/RetinexPixelTrans/paired/steps/results/figures/training_total_proxy.png`
