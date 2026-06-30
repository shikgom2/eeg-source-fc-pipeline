"""
eLORETA source 추출 + DK(Desikan-Killiany) 68 ROI 평균
- 입력: {subject}_equalized-epo.fif + {subject}_inv.fif
- 출력: {subject}_roi_timeseries.npy (68 ROI x n_epochs x n_times), roi_names.json
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

PREPROCESSED_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\preprocessed")
INVERSE_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\inverse")
SUBJECTS_DIR = r"D:\Github\eeg-source-fc-pipeline\outputs\freesurfer_subjects"
SUBJECT = "nihpd_4.5-8.5"
OUTPUT_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\roi_timeseries")

METHOD = "eLORETA"
LAMBDA2 = 1.0 / 9.0  # SNR=3 가정


def get_dk_labels():
    labels = mne.read_labels_from_annot(
        SUBJECT, parc="aparc", subjects_dir=SUBJECTS_DIR, verbose="error"
    )
    # unknown / corpuscallosum 등 제외
    labels = [l for l in labels if "unknown" not in l.name]
    return labels


def process_subject(subject_key: str, labels, output_dir: Path) -> dict:
    epo_path = PREPROCESSED_DIR.rglob(f"{subject_key}_equalized-epo.fif")
    epo_path = next(epo_path, None)
    inv_path = INVERSE_DIR / f"{subject_key}_inv.fif"

    if epo_path is None or not inv_path.exists():
        return {"subject_key": subject_key, "status": "missing"}

    out_path = output_dir / f"{subject_key}_roi_timeseries.npy"

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose="error")
    epochs.set_eeg_reference("average", projection=True, verbose="error")
    inv = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose="error")

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=LAMBDA2, method=METHOD, verbose="error"
    )

    # ROI별 평균 시계열 (mean_flip: 극성 상쇄 방지)
    roi_data = mne.extract_label_time_course(
        stcs, labels, inv["src"], mode="mean_flip", verbose="error"
    )
    roi_array = np.array(roi_data)  # (n_epochs, n_roi, n_times)

    np.save(str(out_path), roi_array)
    return {"subject_key": subject_key, "status": "ok", "shape": roi_array.shape}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = get_dk_labels()
    roi_names = [l.name for l in labels]
    (OUTPUT_DIR / "roi_names.json").write_text(
        json.dumps(roi_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"DK ROI 개수: {len(roi_names)}")

    inv_files = sorted(INVERSE_DIR.glob("*_inv.fif"))
    subject_keys = [f.name.replace("_inv.fif", "") for f in inv_files]
    print(f"대상 피험자: {len(subject_keys)}명\n")

    results = []
    for i, key in enumerate(subject_keys, 1):
        out_path = OUTPUT_DIR / f"{key}_roi_timeseries.npy"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(subject_keys)}] {key}: 건너뜀")
            continue
        try:
            r = process_subject(key, labels, OUTPUT_DIR)
            results.append(r)
            print(f"  [{i}/{len(subject_keys)}] {key}: {r['status']} {r.get('shape', '')}")
        except Exception as e:
            results.append({"subject_key": key, "status": "error", "error": str(e)})
            print(f"  [{i}/{len(subject_keys)}] {key}: 오류 - {e}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n완료: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
