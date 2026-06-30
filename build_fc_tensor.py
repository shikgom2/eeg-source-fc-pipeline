"""
FC_tensor[subject, time, band, roi_i, roi_j] 생성
- wPLI, AEC-c, imaginary coherence 각각 별도 텐서로 저장
- index_table.csv: 텐서의 0번 축(observation) 순서와 매칭되는 subject_id/timepoint/label
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FC_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc")
QC_TABLE_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\qc_table.csv")
OUTPUT_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor")

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METHODS = ["wpli", "aec", "imcoh"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qc = pd.read_csv(QC_TABLE_PATH)
    included = qc[qc["included_in_analysis"]].sort_values("subject_key").reset_index(drop=True)

    n_obs = len(included)
    n_roi = 68
    tensors = {m: np.full((n_obs, len(BANDS), n_roi, n_roi), np.nan) for m in METHODS}

    missing = []
    for i, row in included.iterrows():
        key = row["subject_key"]
        for b, band in enumerate(BANDS):
            for m in METHODS:
                fp = FC_DIR / f"{key}_fc_{m}_{band}.npy"
                if not fp.exists():
                    missing.append(str(fp))
                    continue
                tensors[m][i, b] = np.load(fp)

    for m in METHODS:
        out_path = OUTPUT_DIR / f"FC_tensor_{m}.npy"
        np.save(out_path, tensors[m])
        print(f"{out_path}: shape={tensors[m].shape}")

    index_table = included[
        ["subject_key", "subject_id", "timepoint", "condition", "label", "diagnosis"]
    ].reset_index(drop=True)
    index_table.to_csv(OUTPUT_DIR / "index_table.csv", index=False, encoding="utf-8-sig")

    meta = {"bands": BANDS, "methods": METHODS, "n_roi": n_roi, "n_obs": n_obs}
    (OUTPUT_DIR / "tensor_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nindex_table.csv: {len(index_table)}행")
    if missing:
        print(f"누락 파일: {len(missing)}개 (NaN으로 채움)")
        for m in missing[:5]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
