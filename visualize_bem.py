"""
BEM 모델 시각화 (뇌/두개골/두피 3-layer)
"""
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SUBJECTS_DIR = r"D:\Github\eeg-source-fc-pipeline\outputs\freesurfer_subjects"
SUBJECT = "nihpd_4.5-8.5"
OUT_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\head_model\bem_visualization.png")

# 서버에서 freesurfer_subjects 폴더가 필요하므로 BEM fif에서 직접 표면 정보 로드
bem_path = r"D:\Github\eeg-source-fc-pipeline\outputs\head_model\nihpd_4.5-8.5-bem-sol.fif"
bem = mne.read_bem_solution(bem_path)

fig = mne.viz.plot_bem(
    subject=SUBJECT,
    subjects_dir=SUBJECTS_DIR,
    brain_surfaces="white",
    orientation="coronal",
    slices=[80, 100, 120, 140],
    show=False,
)
fig.savefig(str(OUT_PATH), dpi=150)
print(f"저장 완료: {OUT_PATH}")
