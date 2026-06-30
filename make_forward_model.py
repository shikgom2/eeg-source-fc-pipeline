"""
Forward model 생성
- NIHPD 4.5-8.5 BEM + Source space + GSN-64 montage
- 출력: nihpd_4.5-8.5-fwd.fif
"""
import mne
from pathlib import Path

HEAD_MODEL_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\head_model")
SUBJECT = "nihpd_4.5-8.5"

bem_path = HEAD_MODEL_DIR / f"{SUBJECT}-bem-sol.fif"
src_path = HEAD_MODEL_DIR / f"{SUBJECT}-src.fif"
fwd_path = HEAD_MODEL_DIR / f"{SUBJECT}-fwd.fif"

print("=== BEM, Source space 로드 ===")
bem = mne.read_bem_solution(str(bem_path))
src = mne.read_source_spaces(str(src_path))

print("\n=== GSN-64 montage 로드 ===")
montage = mne.channels.make_standard_montage("GSN-HydroCel-64_1.0")

# 64채널 - VREF = 63채널 info 생성
ch_names = [ch for ch in montage.ch_names if ch != "VREF"]
info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
info.set_montage(montage, on_missing="warn")

print(f"채널 수: {len(ch_names)}")

print("\n=== Forward model 생성 ===")
fwd = mne.make_forward_solution(
    info=info,
    trans="fsaverage",  # 표준 변환 사용
    src=src,
    bem=bem,
    eeg=True,
    meg=False,
    verbose=True,
)

mne.write_forward_solution(str(fwd_path), fwd, overwrite=True)
print(f"\nForward model 저장: {fwd_path}")
print(f"  Sources: {fwd['nsource']}")
print(f"  Channels: {fwd['nchan']}")
