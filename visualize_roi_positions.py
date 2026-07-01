"""
DK 68 ROI 위치 시각화 (뇌 3개 단면에 투영)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mne

SUBJECTS_DIR = r"D:\Github\eeg-source-fc-pipeline\outputs\freesurfer_subjects"
SUBJECT = "nihpd_4.5-8.5"
OUT_PATH = r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor\roi_positions.png"

# DK labels 로드
labels = mne.read_labels_from_annot(
    SUBJECT, parc="aparc", subjects_dir=SUBJECTS_DIR, verbose="error"
)
labels = [l for l in labels if "unknown" not in l.name]

# 각 label의 centroid 좌표 (MRI RAS 좌표계, mm 단위)
src = mne.read_source_spaces(
    r"D:\Github\eeg-source-fc-pipeline\outputs\head_model\nihpd_4.5-8.5-src.fif",
    verbose="error"
)

positions = []
names = []
hemis = []
for label in labels:
    hemi_idx = 0 if label.hemi == "lh" else 1
    verts = src[hemi_idx]["rr"][label.vertices] * 1000  # m → mm
    centroid = verts.mean(axis=0)
    positions.append(centroid)
    short = label.name.replace("-lh", "").replace("-rh", "")
    names.append(short)
    hemis.append(label.hemi)

pos = np.array(positions)  # (68, 3) — x=LR, y=AP, z=IS

lh_mask = np.array(hemis) == "lh"
rh_mask = ~lh_mask

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

views = [
    ("Axial (top)", pos[:, 0], pos[:, 1], "X (L-R, mm)", "Y (P-A, mm)"),
    ("Sagittal (side)", pos[:, 1], pos[:, 2], "Y (P-A, mm)", "Z (I-S, mm)"),
    ("Coronal (front)", pos[:, 0], pos[:, 2], "X (L-R, mm)", "Z (I-S, mm)"),
]

for ax, (title, xdata, ydata, xlabel, ylabel) in zip(axes, views):
    ax.scatter(xdata[lh_mask], ydata[lh_mask], c="#4fc3f7", s=60, zorder=3, label="LH")
    ax.scatter(xdata[rh_mask], ydata[rh_mask], c="#ef9a9a", s=60, zorder=3, label="RH")

    for i in range(len(names)):
        ax.annotate(
            names[i], (xdata[i], ydata[i]),
            fontsize=4.5, color="white", alpha=0.85,
            xytext=(3, 3), textcoords="offset points"
        )

    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color="#aaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white", framealpha=0.5)

fig.suptitle("Desikan-Killiany 68 ROI Positions (NIHPD 4.5-8.5 template)",
             color="white", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"저장 완료: {OUT_PATH}")
