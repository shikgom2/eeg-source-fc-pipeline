"""
FC (Functional Connectivity) 계산
- 입력: {subject}_roi_timeseries.npy (40 epochs, 68 ROI, 501 timepoints)
- 방법: wPLI(1순위), AEC-c(2순위, leakage-orthogonalized), imaginary coherence(보조)
- 밴드: delta(1-4)/theta(4-8)/alpha(8-13)/beta(13-30)/gamma(30-45) Hz
  * delta: 2초 epoch으로는 5-cycle 기준(5초 이상) 미달 → 신뢰도 낮음, exploratory로만 사용
  * theta/alpha: 1차 가설 대역 (ERP theta, resting alpha aperiodic 선행 보고)
- 출력: {subject}_fc_{method}_{band}.npy (68 x 68 행렬)
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from mne_connectivity import spectral_connectivity_epochs

ROI_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\roi_timeseries")
OUTPUT_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc")
SFREQ = 250.0

BANDS = {
    "delta": (1, 4),   # exploratory (epoch 길이 한계로 신뢰도 낮음)
    "theta": (4, 8),   # 1차 가설
    "alpha": (8, 13),  # 1차 가설
    "beta": (13, 30),
    "gamma": (30, 45),
}

METHODS = ["wpli", "imcoh"]  # mne-connectivity로 한 번에 계산 가능한 것들


def bandpass(data: np.ndarray, low: float, high: float, sfreq: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def compute_phase_metrics(data: np.ndarray, low: float, high: float) -> dict:
    """wPLI + imaginary coherence를 한 번에 계산. data: (n_epochs, n_roi, n_times)"""
    con = spectral_connectivity_epochs(
        data, method=METHODS, mode="multitaper", sfreq=SFREQ,
        fmin=low, fmax=high, faverage=True, verbose=False,
    )
    out = {}
    for name, c in zip(METHODS, con):
        mat = c.get_data(output="dense")[:, :, 0]
        out[name] = mat + mat.T  # 하삼각만 채워지므로 대칭화
    return out


def compute_aec_c(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """진폭 envelope correlation with pairwise leakage correction (orthogonalization).
    Colclough et al. 2015 방식, ROI 쌍마다 벡터화."""
    filtered = bandpass(data, low, high, SFREQ)  # (n_epochs, n_roi, n_times)
    n_epochs, n_roi, n_times = filtered.shape
    analytic = hilbert(filtered, axis=-1)

    mat_sum = np.zeros((n_roi, n_roi))
    for ep in range(n_epochs):
        z = analytic[ep]  # (n_roi, n_times) complex
        amp = np.abs(z)
        amp_c = amp - amp.mean(axis=1, keepdims=True)
        amp_std = amp.std(axis=1)

        for i in range(n_roi):
            zi_unit = z[i] / np.abs(z[i])
            z_orth = np.imag(z * np.conj(zi_unit)[None, :])  # (n_roi, n_times), real
            amp_orth = np.abs(z_orth)
            amp_orth_c = amp_orth - amp_orth.mean(axis=1, keepdims=True)
            amp_orth_std = amp_orth.std(axis=1)

            cov = (amp_c[i][None, :] * amp_orth_c).mean(axis=1)
            denom = amp_std[i] * amp_orth_std
            corr = np.where(denom > 1e-30, cov / denom, 0.0)
            mat_sum[i, :] += corr

    mat = mat_sum / n_epochs
    mat = (mat + mat.T) / 2  # pairwise 직교화로 약간 비대칭이라 평균내서 대칭화
    np.fill_diagonal(mat, 0)
    return mat


def process_subject(npy_path: Path, output_dir: Path) -> dict:
    subject_key = npy_path.name.replace("_roi_timeseries.npy", "")
    data = np.load(npy_path)  # (n_epochs, n_roi, n_times)

    bands_done = []
    for band_name, (low, high) in BANDS.items():
        phase = compute_phase_metrics(data, low, high)
        np.save(output_dir / f"{subject_key}_fc_wpli_{band_name}.npy", phase["wpli"])
        np.save(output_dir / f"{subject_key}_fc_imcoh_{band_name}.npy", np.abs(phase["imcoh"]))

        aec_mat = compute_aec_c(data, low, high)
        np.save(output_dir / f"{subject_key}_fc_aec_{band_name}.npy", aec_mat)

        bands_done.append(band_name)

    return {"subject_key": subject_key, "status": "ok", "bands": bands_done}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(ROI_DIR.glob("*_roi_timeseries.npy"))
    print(f"대상 피험자: {len(npy_files)}명\n")

    results = []
    for i, npy_path in enumerate(npy_files, 1):
        subject_key = npy_path.name.replace("_roi_timeseries.npy", "")
        check_path = OUTPUT_DIR / f"{subject_key}_fc_wpli_delta.npy"
        if check_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(npy_files)}] {subject_key}: 건너뜀")
            continue
        try:
            r = process_subject(npy_path, OUTPUT_DIR)
            results.append(r)
            print(f"  [{i}/{len(npy_files)}] {subject_key}: 완료")
        except Exception as e:
            results.append({"subject_key": subject_key, "status": "error", "error": str(e)})
            print(f"  [{i}/{len(npy_files)}] {subject_key}: 오류 - {e}")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n완료: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
