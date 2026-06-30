"""
Epoch 균등화 스크립트
- 전처리 완료된 *_clean_epochs-epo.fif 파일들을 읽어서
- 최소 epoch 수 기준으로 모든 피험자를 동일하게 맞춤
- 결과: *_equalized-epo.fif 저장

사용법:
    python equalize_epochs.py
    python equalize_epochs.py --min-epochs 20 --output-dir D:/Github/outputs/preprocessed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mne
import numpy as np

OUTPUT_ROOT = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\preprocessed")


def load_epoch_counts(output_dir: Path) -> dict[str, int]:
    """각 피험자의 clean epoch 수를 JSON 요약에서 읽음 (빠름)."""
    counts = {}
    for summary_path in sorted(output_dir.glob("*/preprocessing_summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        key = data.get("subject")
        n = data.get("n_clean_epochs")
        if key and n is not None:
            counts[key] = n
    return counts


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir

    # 1. epoch 수 파악
    counts = load_epoch_counts(output_dir)
    if not counts:
        print("ERROR: preprocessing_summary.json 파일을 찾을 수 없습니다.")
        print(f"       경로 확인: {output_dir}")
        return

    total = len(counts)
    print(f"전처리 완료 피험자: {total}명\n")

    # 2. 분포 출력
    vals = sorted(counts.values())
    print("=== Clean Epoch 분포 ===")
    print(f"  min={min(vals)}, 중앙값={vals[len(vals)//2]}, max={max(vals)}")
    print(f"  {args.min_epochs}개 미만 → 제외 대상: "
          f"{sum(1 for v in vals if v < args.min_epochs)}명")
    print()

    # 3. 최소 기준 미달 피험자 제외
    excluded = {k: v for k, v in counts.items() if v < args.min_epochs}
    included = {k: v for k, v in counts.items() if v >= args.min_epochs}

    if excluded:
        print(f"=== 제외 ({len(excluded)}명, epoch < {args.min_epochs}) ===")
        for k, v in sorted(excluded.items()):
            print(f"  {k}: {v}개")
        print()

    target_n = min(included.values())
    print(f"=== 균등화 기준: {target_n} epochs ({target_n * 2}초) ===")
    print(f"대상: {len(included)}명\n")

    # 4. 균등화 실행
    rng = np.random.default_rng(97)
    done = 0
    skipped = 0
    failed = []

    for subject_key, n_epochs in sorted(included.items()):
        epo_path = output_dir / subject_key / f"{subject_key}_clean_epochs-epo.fif"
        out_path = output_dir / subject_key / f"{subject_key}_equalized-epo.fif"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if not epo_path.exists():
            print(f"  [없음] {subject_key}")
            failed.append(subject_key)
            continue

        try:
            epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
            if len(epochs) > target_n:
                idx = np.sort(rng.choice(len(epochs), size=target_n, replace=False))
                epochs = epochs[idx]
            epochs.save(str(out_path), overwrite=True, verbose=False)
            done += 1
            print(f"  [{done:03d}] {subject_key}: {n_epochs} → {len(epochs)} epochs")
        except Exception as e:
            print(f"  [오류] {subject_key}: {e}")
            failed.append(subject_key)

    # 5. 결과 요약
    print(f"\n=== 완료 ===")
    print(f"  균등화 완료: {done}명")
    print(f"  건너뜀(기존): {skipped}명")
    print(f"  제외(epoch 부족): {len(excluded)}명")
    print(f"  실패: {len(failed)}명")
    print(f"\n  최종 분석 가능: {done + skipped}명")
    print(f"  피험자당 epoch: {target_n}개 = {target_n * 2}초")
    print(f"\n출력 파일: {{subject}}_equalized-epo.fif")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Epoch 균등화")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT,
                   help="전처리 결과 폴더 (기본: D:/Github/outputs/preprocessed)")
    p.add_argument("--min-epochs", type=int, default=20,
                   help="이 수 미만이면 분석에서 제외 (기본: 20)")
    p.add_argument("--overwrite", action="store_true",
                   help="이미 균등화된 파일 덮어쓰기")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
