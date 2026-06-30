"""
Inverse operator 생성 (eLORETA)
- 피험자별 equalized epoch + 공통 forward model
- 출력: {subject}_inv.fif
"""
import argparse
from pathlib import Path

import mne
import numpy as np

PREPROCESSED_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\preprocessed")
HEAD_MODEL_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\head_model")
OUTPUT_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\inverse")
FWD_PATH = HEAD_MODEL_DIR / "nihpd_4.5-8.5-fwd.fif"


def process_subject(epo_path: Path, fwd: mne.Forward, output_dir: Path) -> dict:
    subject_key = epo_path.name.replace("_equalized-epo.fif", "")
    out_path = output_dir / f"{subject_key}_inv.fif"

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose="error")
    epochs.set_eeg_reference("average", projection=True, verbose="error")

    # 노이즈 공분산: epoch 전체 구간에서 추정 (resting이라 baseline 없음)
    noise_cov = mne.compute_covariance(epochs, method="shrunk", verbose="error")

    info = epochs.info
    inv = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=0.2, depth=0.8, verbose="error"
    )
    mne.minimum_norm.write_inverse_operator(str(out_path), inv, overwrite=True, verbose="error")

    return {"subject_key": subject_key, "status": "ok", "out_path": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fwd = mne.read_forward_solution(str(FWD_PATH), verbose="error")

    epo_files = sorted(PREPROCESSED_DIR.rglob("*_equalized-epo.fif"))
    print(f"대상 피험자: {len(epo_files)}명")

    results = []
    for i, epo_path in enumerate(epo_files, 1):
        subject_key = epo_path.name.replace("_equalized-epo.fif", "")
        out_path = OUTPUT_DIR / f"{subject_key}_inv.fif"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(epo_files)}] {subject_key}: 건너뜀 (기존)")
            continue
        try:
            r = process_subject(epo_path, fwd, OUTPUT_DIR)
            results.append(r)
            print(f"  [{i}/{len(epo_files)}] {subject_key}: 완료")
        except Exception as e:
            results.append({"subject_key": subject_key, "status": "error", "error": str(e)})
            print(f"  [{i}/{len(epo_files)}] {subject_key}: 오류 - {e}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n완료: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
