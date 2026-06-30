"""
qc_table[subject, time] 생성
- clean_epochs: 균등화 전 원본 clean epoch 수
- n_equalized_epochs: 균등화 후 epoch 수 (고정값, 보통 40)
- restValidity: EEG_restValidity (1=good, 2=ok)
- max_impedance: 3개 impedance 측정값 중 최댓값
- movement_flag: EEG_restingNotes에 움직임 관련 키워드 존재 여부
- diagnosis/label: ASD=1, TD=0 (참고용 추가 컬럼)
"""
import json
import re
from pathlib import Path

import pandas as pd

PREPROCESSED_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\preprocessed")
OUTPUT_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\qc_table.csv")

MOVEMENT_KEYWORDS = [
    "moving", "movement", "talking", "touch", "기침", "움직임", "말함", "talk",
    "흔듦", "떪", "흔들", "만짐", "긁음", "당김",
]


def has_movement(note) -> bool:
    if not isinstance(note, str) or not note.strip():
        return False
    note_lower = note.lower()
    return any(kw.lower() in note_lower for kw in MOVEMENT_KEYWORDS)


def main():
    eeg = pd.read_excel(
        r"D:\Datasets\rest\Resting(Pre) EEG\NEST EEG ERP notes.xlsx", sheet_name="Sheet0"
    )
    diag = pd.read_excel(
        r"D:\Datasets\rest\Resting(Pre) EEG\NESTdata_fromCCPL_260604_Demo only.xlsx"
    )
    eeg["ID"] = eeg["ID"].str.strip()
    diag["ID"] = diag["ID"].str.strip()

    diag["birthdate"] = pd.to_datetime(diag["birthdate"], format="mixed", errors="coerce")
    eeg["EEG_visitDate"] = pd.to_datetime(eeg["EEG_visitDate"], format="mixed", errors="coerce")

    imp_cols = ["EEG_restImpdncValue1", "EEG_restImpdncValue2", "EEG_restImpdncValue3"]
    for c in imp_cols:
        eeg[c] = pd.to_numeric(eeg[c], errors="coerce")
    eeg["max_impedance"] = eeg[imp_cols].max(axis=1)
    eeg["movement_flag"] = eeg["EEG_restingNotes"].apply(has_movement)

    rows = []
    summary_files = sorted(PREPROCESSED_DIR.glob("*/preprocessing_summary.json"))
    for sp in summary_files:
        data = json.loads(sp.read_text(encoding="utf-8"))
        subject_key = data.get("subject")
        if subject_key is None:
            continue
        m = re.match(r"(NT\d+)_T(\d)_(\w+)", subject_key)
        if not m:
            continue
        subject_id, tp, condition = m.group(1), int(m.group(2)), m.group(3)

        eeg_row = eeg[(eeg["ID"] == subject_id) & (eeg["time"] == tp)]
        diag_row = diag[(diag["ID"] == subject_id) & (diag["time"] == tp)]

        equalized_path = sp.parent / f"{subject_key}_equalized-epo.fif"

        age_months = None
        if len(eeg_row) and len(diag_row):
            visit_date = eeg_row["EEG_visitDate"].values[0]
            birth_date = diag_row["birthdate"].values[0]
            if pd.notna(visit_date) and pd.notna(birth_date):
                days = (pd.Timestamp(visit_date) - pd.Timestamp(birth_date)).days
                if 0 < days < 365.25 * 18:  # 비정상 날짜(음수 등) 걸러내기
                    age_months = round(days / 30.4375, 1)

        rows.append({
            "subject_key": subject_key,
            "subject_id": subject_id,
            "timepoint": tp,
            "condition": condition,
            "clean_epochs": data.get("n_clean_epochs"),
            "n_equalized_epochs": 40 if equalized_path.exists() else None,
            "included_in_analysis": equalized_path.exists(),
            "age_months": age_months,
            "restValidity": eeg_row["EEG_restValidity"].values[0] if len(eeg_row) else None,
            "max_impedance": eeg_row["max_impedance"].values[0] if len(eeg_row) else None,
            "movement_flag": eeg_row["movement_flag"].values[0] if len(eeg_row) else None,
            "n_bad_channels": len(data.get("bad_channels", [])),
            "n_ica_excluded": len(data.get("ica_excluded_components", [])),
            "diagnosis": int(diag_row["Diagnosis"].values[0]) if len(diag_row) else None,
            "label": (
                "ASD" if len(diag_row) and diag_row["Diagnosis"].values[0] == 1
                else ("TD" if len(diag_row) else None)
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"총 행: {len(df)}")
    print(f"분석 포함(included_in_analysis=True): {df['included_in_analysis'].sum()}")
    print(f"movement_flag=True: {df['movement_flag'].sum()}")


if __name__ == "__main__":
    main()
