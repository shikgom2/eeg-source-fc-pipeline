"""
ASD vs TD 그룹 평균 68x68 FC 행렬 히트맵
- wPLI, theta/alpha 밴드 기준
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FC_TENSOR_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor")
ROI_NAMES_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\roi_timeseries\roi_names.json")
OUT_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor\fc_heatmap_asd_vs_td.png")

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
PLOT_BANDS = ["theta", "alpha"]
METHOD = "wpli"

roi_names = json.loads(ROI_NAMES_PATH.read_text(encoding="utf-8"))
index_table = pd.read_csv(FC_TENSOR_DIR / "index_table.csv")
tensor = np.load(FC_TENSOR_DIR / f"FC_tensor_{METHOD}.npy")  # (n_obs, n_band, 68, 68)

asd_mask = (index_table["label"] == "ASD").values
td_mask = (index_table["label"] == "TD").values
print(f"ASD: {asd_mask.sum()}명, TD: {td_mask.sum()}명")

fig, axes = plt.subplots(len(PLOT_BANDS), 3, figsize=(18, 6 * len(PLOT_BANDS)))
if len(PLOT_BANDS) == 1:
    axes = axes[None, :]

for row, band in enumerate(PLOT_BANDS):
    b = BANDS.index(band)
    asd_mat = np.nanmean(tensor[asd_mask, b], axis=0)
    td_mat = np.nanmean(tensor[td_mask, b], axis=0)
    diff_mat = asd_mat - td_mat

    vmax = max(np.nanmax(asd_mat), np.nanmax(td_mat))

    for col, (mat, title, cmap, vlim) in enumerate([
        (asd_mat, f"ASD mean ({band}, {METHOD})", "viridis", (0, vmax)),
        (td_mat, f"TD mean ({band}, {METHOD})", "viridis", (0, vmax)),
        (diff_mat, f"ASD - TD diff ({band})", "RdBu_r", (-np.nanmax(np.abs(diff_mat)), np.nanmax(np.abs(diff_mat)))),
    ]):
        ax = axes[row, col]
        im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(str(OUT_PATH), dpi=150)
print(f"저장 완료: {OUT_PATH}")
