# Step 03 training dynamics

Eval rows parsed: `600`

A run is flagged when eval total loss decreases while the R consistency proxy drops by more than 10 dB.

## Potential loss/R-consistency mismatch

- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.05er_0.5smv1_20260720-121319`: total Δ=-0.0541, R proxy Δ=-16.94 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.15er_0.5smv1_20260720-135547`: total Δ=-0.0431, R proxy Δ=-36.13 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016`: total Δ=-0.0308, R proxy Δ=-28.19 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv2_20260712-213703`: total Δ=-0.0364, R proxy Δ=-25.52 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255`: total Δ=-0.0476, R proxy Δ=-31.07 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv2_20260720-085219`: total Δ=-0.0336, R proxy Δ=-24.17 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv3_20260720-103249`: total Δ=-0.0339, R proxy Δ=-24.68 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824`: total Δ=-0.0484, R proxy Δ=-28.89 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.7smv1_20260720-035020`: total Δ=-0.0486, R proxy Δ=-27.22 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.8smv1_20260720-053058`: total Δ=-0.0501, R proxy Δ=-25.62 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_1.0smv1_20260720-071136`: total Δ=-0.0506, R proxy Δ=-22.49 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.1smv2_20260717-052115`: total Δ=-0.0252, R proxy Δ=-33.05 dB
- `LOLv2_1.0rh_0.3rl_0.003crh_0.003crl_0.1er_0.5smv1_20260720-153620`: total Δ=-0.0478, R proxy Δ=-31.16 dB
- `LOLv2_1.0rh_0.3rl_0.005crh_0.005crl_0.1er_0.5smv1_20260720-171701`: total Δ=-0.0478, R proxy Δ=-31.01 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.1er_0.1smv2_20260717-015927`: total Δ=-0.0310, R proxy Δ=-28.47 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv2_20260717-070107`: total Δ=-0.0264, R proxy Δ=-33.02 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv3_20260717-084100`: total Δ=-0.0268, R proxy Δ=-32.71 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.1er_0.1smv2_20260717-034129`: total Δ=-0.0326, R proxy Δ=-30.25 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.1smv3_20260717-123101`: total Δ=-0.0292, R proxy Δ=-34.09 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.2smv3_20260717-174512`: total Δ=-0.0305, R proxy Δ=-34.03 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114`: total Δ=-0.0273, R proxy Δ=-34.39 dB
- `LOLv2_1.0rh_0.3rl_0.1crh_0.1crl_0.2er_0.1smv3_20260717-141642`: total Δ=-0.0360, R proxy Δ=-34.01 dB

Figure: `/home/ipr4090/2024_hzf/Retinex_TDN/_analysis/RetinexPixelTrans/paired/steps/results/figures/training_total_proxy.png`
