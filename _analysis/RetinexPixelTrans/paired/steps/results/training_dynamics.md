# Step 03 training dynamics

Eval rows parsed: `281`

A run is flagged when eval total loss decreases while the R consistency proxy drops by more than 10 dB.

## Potential loss/R-consistency mismatch

- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv1_20260713-042929`: total Δ=-0.0337, R proxy Δ=-30.38 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016`: total Δ=-0.0308, R proxy Δ=-28.19 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv1_20260713-061113`: total Δ=-0.0320, R proxy Δ=-30.81 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv2_20260712-213703`: total Δ=-0.0364, R proxy Δ=-25.52 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255`: total Δ=-0.0476, R proxy Δ=-31.07 dB
- `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.1smv2_20260717-052115`: total Δ=-0.0252, R proxy Δ=-33.05 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.1er_0.1smv2_20260717-015927`: total Δ=-0.0310, R proxy Δ=-28.47 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv2_20260717-070107`: total Δ=-0.0264, R proxy Δ=-33.02 dB
- `LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv3_20260717-084100`: total Δ=-0.0268, R proxy Δ=-32.71 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.1er_0.1smv2_20260717-034129`: total Δ=-0.0326, R proxy Δ=-30.25 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.1smv3_20260717-123101`: total Δ=-0.0292, R proxy Δ=-34.09 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.2smv3_20260717-174512`: total Δ=-0.0305, R proxy Δ=-34.03 dB
- `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114`: total Δ=-0.0273, R proxy Δ=-34.39 dB
- `LOLv2_1.0rh_0.3rl_0.1crh_0.1crl_0.2er_0.1smv3_20260717-141642`: total Δ=-0.0360, R proxy Δ=-34.01 dB

Figure: `/home/ipr4090/2024_hzf/Retinex_TDN/_analysis/RetinexPixelTrans/paired/steps/results/figures/training_total_proxy.png`
