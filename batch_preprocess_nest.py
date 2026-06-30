"""
NEST 코호트 resting EEG 배치 전처리 스크립트
- 입력: EGI MFF (D:/Datasets/rest/Resting(Pre) EEG/)
- QC: 엑셀 EEG_restValidity 1~2 포함, 3 제외
- Eyes-open/closed 분리:
    T1: single "resting" 블록
    T2: resting_pre (eyes-open), resting_post (eyes-closed) 별도 처리
- Epoch 균등화: 배치 완료 후 전체 최솟값에 맞춤
- 출력: outputs/preprocessed/{subject}_{T}_{condition}/
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from source_fc_utils import (
    GSN64_FRONTAL_PROXY_CHANNELS,
    PROJECT_ROOT,
    equalize_epoch_counts,
    find_mff_condition,
    load_raw_from_mff,
    make_fixed_length_epochs,
    preprocess_raw,
    save_clean_epochs,
    save_preprocessed_raw,
    save_preprocessing_summary,
)

MFF_ROOT = Path(r"D:\Datasets\rest\Resting(Pre) EEG")
EXCEL_PATH = Path(r"C:\Users\youngseok\Downloads\NEST EEG ERP notes.xlsx")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessed"

L_FREQ = 1.0
H_FREQ = 45.0
NOTCH_FREQ = 60.0
RESAMPLE_SFREQ = 250.0
MAX_VALIDITY = 2
EPOCH_DURATION = 2.0    # 초
AMP_THRESHOLD = 150e-6  # 150 µV

# T1: resting 단일 / T2: pre + post
CONDITION_MAP = {
    1: ["resting"],
    2: ["pre", "post"],
}


def load_qc_table(excel_path: Path, max_validity: int) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="Sheet0")
    df = df[df["EEG_restValidity"].notna()].copy()
    df = df[df["EEG_restValidity"] <= max_validity].copy()
    df["subject"] = df["ID"].astype(str).str.strip()
    df["timepoint"] = df["time"].astype(int)
    return df[["subject", "timepoint", "EEG_restValidity"]].reset_index(drop=True)


def run_subject_condition(
    subject: str,
    timepoint: int,
    condition: str,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    tp_str = f"T{timepoint}"
    subject_key = f"{subject}_{tp_str}_{condition}"
    out_dir = output_root / subject_key

    # 이미 처리됐으면 건너뜀
    epo_path = out_dir / f"{subject_key}_clean_epochs-epo.fif"
    if epo_path.exists() and not args.overwrite:
        return {"subject_key": subject_key, "status": "skipped"}

    try:
        mff_path = find_mff_condition(MFF_ROOT, subject, tp_str, condition)
        raw = load_raw_from_mff(mff_path)
        raw, summary = preprocess_raw(
            raw,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freqs=(args.notch_freq,),
            resample_sfreq=args.resample_sfreq,
            detect_bad_channels_flag=True,
            interpolate_bads_flag=True,
            run_ica=not args.skip_ica,
            frontal_proxy_channels=GSN64_FRONTAL_PROXY_CHANNELS,
        )

        # Raw 저장 (source 분석용)
        save_preprocessed_raw(raw, out_dir, subject_key)

        # Epoch 분할 + artifact 제거
        epochs = make_fixed_length_epochs(
            raw,
            epoch_duration=args.epoch_duration,
            amplitude_threshold=args.amp_threshold,
        )
        n_clean = len(epochs)

        # Epoch 저장 (FC 분석용)
        save_clean_epochs(epochs, out_dir, subject_key)

        save_preprocessing_summary(
            out_dir,
            subject_key,
            summary,
            extra={
                "mff_path": str(mff_path),
                "timepoint": tp_str,
                "condition": condition,
                "n_clean_epochs": n_clean,
                "epoch_duration_s": args.epoch_duration,
                "amp_threshold_uv": args.amp_threshold * 1e6,
            },
        )
        return {
            "subject_key": subject_key,
            "status": "ok",
            "n_clean_epochs": n_clean,
            "bad_channels": list(summary.bad_channels),
            "ica_excluded": list(summary.ica_excluded_components),
        }

    except FileNotFoundError as e:
        return {"subject_key": subject_key, "status": "no_mff", "error": str(e)}
    except Exception as e:
        return {
            "subject_key": subject_key,
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEST resting EEG 배치 전처리")
    p.add_argument("--timepoints", nargs="*", type=int, default=[1, 2])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None,
                   help="특정 condition만 (resting/pre/post). 미지정 시 시점별 자동 설정")
    p.add_argument("--max-validity", type=int, default=MAX_VALIDITY)
    p.add_argument("--l-freq", type=float, default=L_FREQ)
    p.add_argument("--h-freq", type=float, default=H_FREQ)
    p.add_argument("--notch-freq", type=float, default=NOTCH_FREQ)
    p.add_argument("--resample-sfreq", type=float, default=RESAMPLE_SFREQ)
    p.add_argument("--epoch-duration", type=float, default=EPOCH_DURATION)
    p.add_argument("--amp-threshold", type=float, default=AMP_THRESHOLD,
                   help="Epoch artifact threshold in Volts (기본: 150e-6)")
    p.add_argument("--skip-ica", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-equalize", action="store_true",
                   help="배치 완료 후 epoch 균등화 건너뜀")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qc = load_qc_table(EXCEL_PATH, args.max_validity)
    qc = qc[qc["timepoint"].isin(args.timepoints)]
    if args.subjects:
        qc = qc[qc["subject"].isin(args.subjects)]

    # 처리할 (subject, timepoint, condition) 목록 생성
    jobs = []
    for row in qc.itertuples():
        conditions = args.conditions or CONDITION_MAP.get(row.timepoint, ["resting"])
        for cond in conditions:
            jobs.append((row.subject, row.timepoint, cond))

    total = len(jobs)
    print(f"처리 대상: {total}건 (피험자 {len(qc)}명 × conditions)")
    print(f"출력 경로: {args.output_dir}\n")

    results = []
    for i, (subject, timepoint, cond) in enumerate(jobs, 1):
        label = f"{subject} T{timepoint} [{cond}]"
        print(f"[{i:03d}/{total}] {label} ...", end=" ", flush=True)
        result = run_subject_condition(subject, timepoint, cond, args.output_dir, args)
        results.append(result)
        print(result["status"])

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    no_mff = [r for r in results if r["status"] == "no_mff"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n=== 전처리 완료 ===")
    print(f"  성공: {len(ok)}건")
    print(f"  건너뜀: {len(skipped)}건")
    print(f"  MFF 없음: {len(no_mff)}건")
    print(f"  오류: {len(errors)}건")

    if ok:
        counts = [r["n_clean_epochs"] for r in ok]
        print(f"  Clean epoch 수: min={min(counts)}, max={max(counts)}, median={sorted(counts)[len(counts)//2]}")

    if no_mff:
        print("\n  [MFF 없음]")
        for r in no_mff:
            print(f"    {r['subject_key']}")

    if errors:
        print("\n  [오류]")
        for r in errors:
            print(f"    {r['subject_key']}: {r['error'][:100]}")

    # Epoch 균등화 (성공한 것들 대상)
    if not args.no_equalize and len(ok) > 1:
        print("\n=== Epoch 균등화 ===")
        import mne
        epochs_dict = {}
        for r in ok:
            key = r["subject_key"]
            epo_path = args.output_dir / key / f"{key}_clean_epochs-epo.fif"
            if epo_path.exists():
                try:
                    epochs_dict[key] = mne.read_epochs(str(epo_path), preload=True, verbose=False)
                except Exception as e:
                    print(f"  로드 실패: {key} — {e}")

        if epochs_dict:
            eq_epochs, min_count, all_counts = equalize_epoch_counts(epochs_dict)
            print(f"  균등화 기준: {min_count} epochs (최솟값)")
            for key, epochs in eq_epochs.items():
                save_clean_epochs(epochs, args.output_dir / key, f"{key}_equalized")
            # 균등화 결과 요약
            for r in ok:
                r["n_equalized_epochs"] = min_count
                r["n_original_epochs"] = all_counts.get(r["subject_key"], None)

    # 결과 저장 (timepoint별 별도 파일 + 전체 통합 파일)
    tp_label = "_".join(str(t) for t in args.timepoints)
    report_path = args.output_dir / f"batch_preprocessing_report_T{tp_label}.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n보고서 저장: {report_path}")


if __name__ == "__main__":
    main()
