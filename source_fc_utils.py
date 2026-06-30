from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
MNE_RUNTIME_HOME = PROJECT_ROOT / "outputs" / ".mne_runtime"
MNE_RUNTIME_HOME.mkdir(parents=True, exist_ok=True)
os.environ["USERPROFILE"] = str(PROJECT_ROOT)
os.environ["HOME"] = str(PROJECT_ROOT)
os.environ["MNE_HOME"] = str(MNE_RUNTIME_HOME)
os.environ["_MNE_FAKE_HOME_DIR"] = str(MNE_RUNTIME_HOME)

import mne
import numpy as np
import pandas as pd
from mne import read_bem_solution, read_source_spaces
from mne.minimum_norm import (
    apply_inverse_raw,
    make_inverse_operator,
    read_inverse_operator,
    write_inverse_operator,
)
from mne.preprocessing import ICA
from mne_connectivity import envelope_correlation
from scipy.signal import butter, coherence, filtfilt, hilbert, welch


CANONICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

DEFAULT_FRONTAL_PROXY_CHANNELS: Tuple[str, ...] = ("Fp1", "Fp2", "AF7", "AF8", "AF3", "AF4")

# EGI GSN-64 채널 중 전두엽에 해당하는 채널 (EOG proxy용)
GSN64_FRONTAL_PROXY_CHANNELS: Tuple[str, ...] = ("E1", "E8", "E14", "E19", "E21", "E25")
DEFAULT_ROI_CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config_default.json"


@dataclass(frozen=True)
class ROI:
    name: str
    group: str
    label_names: Tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingSummary:
    sfreq: float
    n_channels: int
    bad_channels: Tuple[str, ...]
    frontal_proxy_channels: Tuple[str, ...]
    ica_excluded_components: Tuple[int, ...]


def _default_roi_specs() -> Tuple[ROI, ...]:
    return (
        ROI("mPFC_L", "DMN", ("medialorbitofrontal-lh",)),
        ROI("mPFC_R", "DMN", ("medialorbitofrontal-rh",)),
        ROI("PCC_L", "DMN", ("posteriorcingulate-lh", "isthmuscingulate-lh")),
        ROI("PCC_R", "DMN", ("posteriorcingulate-rh", "isthmuscingulate-rh")),
        ROI("Precuneus_L", "DMN", ("precuneus-lh",)),
        ROI("Precuneus_R", "DMN", ("precuneus-rh",)),
        ROI("Angular_L", "DMN", ("inferiorparietal-lh",)),
        ROI("Angular_R", "DMN", ("inferiorparietal-rh",)),
        ROI("STS_L", "Social Brain", ("bankssts-lh",)),
        ROI("STS_R", "Social Brain", ("bankssts-rh",)),
        ROI("TPJ_L", "Social Brain", ("supramarginal-lh", "inferiorparietal-lh")),
        ROI("TPJ_R", "Social Brain", ("supramarginal-rh", "inferiorparietal-rh")),
        ROI("IFG_L", "Social Brain", ("parsopercularis-lh", "parstriangularis-lh", "parsorbitalis-lh")),
        ROI("IFG_R", "Social Brain", ("parsopercularis-rh", "parstriangularis-rh", "parsorbitalis-rh")),
        ROI("dlPFC_L", "Frontal", ("rostralmiddlefrontal-lh", "caudalmiddlefrontal-lh")),
        ROI("dlPFC_R", "Frontal", ("rostralmiddlefrontal-rh", "caudalmiddlefrontal-rh")),
        ROI("ACC_L", "Frontal", ("rostralanteriorcingulate-lh", "caudalanteriorcingulate-lh")),
        ROI("ACC_R", "Frontal", ("rostralanteriorcingulate-rh", "caudalanteriorcingulate-rh")),
        ROI("OFC_L", "Frontal", ("lateralorbitofrontal-lh", "medialorbitofrontal-lh")),
        ROI("OFC_R", "Frontal", ("lateralorbitofrontal-rh", "medialorbitofrontal-rh")),
        ROI("V1_L", "Sensory", ("pericalcarine-lh", "cuneus-lh")),
        ROI("V1_R", "Sensory", ("pericalcarine-rh", "cuneus-rh")),
        ROI("S1_L", "Sensory", ("postcentral-lh",)),
        ROI("S1_R", "Sensory", ("postcentral-rh",)),
        ROI("A1_L", "Sensory", ("transversetemporal-lh", "superiortemporal-lh")),
        ROI("A1_R", "Sensory", ("transversetemporal-rh", "superiortemporal-rh")),
        ROI("PosteriorParietal_L", "Sensory", ("superiorparietal-lh", "inferiorparietal-lh")),
        ROI("PosteriorParietal_R", "Sensory", ("superiorparietal-rh", "inferiorparietal-rh")),
    )


ROI_SPECS: Tuple[ROI, ...] = _default_roi_specs()


def find_mff(data_dir: Path, subject: str, timepoint: str = "T1") -> Path:
    """Top-level MFF 폴더 경로. data_dir/{timepoint}/{subject}.mff"""
    mff_path = data_dir / timepoint / f"{subject}.mff"
    if not mff_path.exists():
        raise FileNotFoundError(f"MFF not found: {mff_path}")
    return mff_path


def find_mff_condition(
    data_dir: Path, subject: str, timepoint: str, condition: str
) -> Path:
    """
    T1 : 단일 resting 블록 → top-level MFF
    T2 resting_pre  → top-level signal1.bin이 있는 MFF (NT003.mff 등)
    T2 resting_post → EGI 소프트웨어가 sub-MFF를 다른 container에 저장하는 버그 존재
                      전체 T2 폴더를 cross-search해서 {subject}_T? *_resting_post*.mff 검색
    """
    top = find_mff(data_dir, subject, timepoint)

    if condition == "resting":
        return top

    if condition == "pre":
        if (top / "signal1.bin").exists():
            return top
        raise FileNotFoundError(f"resting_pre (signal1.bin) not found in {top}")

    if condition == "post":
        tp_dir = data_dir / timepoint
        # 먼저 자기 container 안 확인
        local = sorted(top.glob(f"*resting_post*.mff"))
        if local:
            return local[0]
        # EGI 버그: 다른 container에 저장됐을 수 있음 → 전체 검색
        # 패턴: {subject}_{timepoint}_*_resting_post*.mff (하위 폴더 포함)
        pattern = f"{subject}_{timepoint}*resting_post*.mff"
        cross = sorted(tp_dir.rglob(pattern))
        if cross:
            return cross[0]
        raise FileNotFoundError(
            f"resting_post not found for {subject} {timepoint}. "
            f"(Searched {top} and all containers under {tp_dir})"
        )

    raise ValueError(f"Unknown condition: {condition}. Use 'pre', 'post', or 'resting'")


def _read_raw_egi_with_date_fix(mff_path: Path) -> mne.io.BaseRaw:
    """EGI 소프트웨어 버그로 연도가 잘못 기록된 MFF를 읽을 때 사용.
    info.xml의 <recordTime>을 임시로 유효한 날짜로 교체 후 원본 복원."""
    import re

    info_xml = mff_path / "info.xml"
    if not info_xml.exists():
        raise FileNotFoundError(f"info.xml not found in {mff_path}")

    original = info_xml.read_bytes()
    try:
        content = original.decode("utf-8")
        # 연도가 1970 미만이면 2000으로 교체 (날짜 메타데이터만, 신호 데이터 무관)
        fixed = re.sub(
            r"(<recordTime>)(\d{4})(-\d{2}-\d{2}T)",
            lambda m: f"{m.group(1)}2000{m.group(3)}" if int(m.group(2)) < 1970 else m.group(0),
            content,
        )
        info_xml.write_text(fixed, encoding="utf-8")
        raw = mne.io.read_raw_egi(str(mff_path), preload=True, verbose="error")
    finally:
        info_xml.write_bytes(original)  # 반드시 원본 복원
    return raw


def load_raw_from_mff(
    mff_path: Path,
    preprocess: bool = False,
) -> mne.io.BaseRaw:
    """EGI MFF 파일을 읽고 GSN-64 몬타주를 설정한다."""
    try:
        raw = mne.io.read_raw_egi(str(mff_path), preload=True, verbose="error")
    except ValueError as e:
        if "meas_date" in str(e):
            raw = _read_raw_egi_with_date_fix(mff_path)
        else:
            raise
    # 잘못된 recording date 제거 (EGI 소프트웨어 버그로 연도가 잘못 기록될 수 있음)
    # meas_date는 신호 분석에 불필요하므로 None으로 설정해 downstream 오류 방지
    raw.set_meas_date(None)
    # VREF 채널 제거 (기준 채널, EEG 신호 아님)
    if "VREF" in raw.ch_names:
        raw.drop_channels(["VREF"])
    # GSN-64 몬타주 적용
    montage = mne.channels.make_standard_montage("GSN-HydroCel-64_1.0")
    raw.set_montage(montage, on_missing="warn", verbose="error")
    return raw


def find_eeglab_set(data_dir: Path, subject: str) -> Path:
    eeg_dir = data_dir / subject / "eeg"
    matches = sorted(eeg_dir.glob(f"{subject}_task-Rest*_eeg.set"))
    if not matches:
        raise FileNotFoundError(f"Could not find EEGLAB .set file under {eeg_dir}")
    return matches[0]


def infer_subject_from_set_path(set_path: Path) -> str:
    stem = set_path.stem
    parts = stem.split("_")
    if parts:
        return parts[0]
    return "sub-001"


def load_dataset_metadata(data_dir: Path, subject: str) -> Dict[str, object]:
    json_path = data_dir / subject / "eeg" / f"{subject}_task-Rest_eeg.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_raw_rest(
    data_dir: Path,
    subject: str,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    resample_sfreq: float | None = 250.0,
    preprocess: bool = True,
) -> mne.io.BaseRaw:
    return load_raw_from_set_path(
        find_eeglab_set(data_dir, subject),
        l_freq=l_freq,
        h_freq=h_freq,
        resample_sfreq=resample_sfreq,
        preprocess=preprocess,
    )


def load_raw_from_set_path(
    set_path: Path,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    resample_sfreq: float | None = 250.0,
    preprocess: bool = True,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    if not preprocess:
        return raw
    processed_raw, _ = preprocess_raw(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        resample_sfreq=resample_sfreq,
    )
    return processed_raw


def load_raw_from_fif(fif_path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_fif(fif_path, preload=True, verbose="error")


def ensure_fsaverage(subjects_dir: Path) -> None:
    fsaverage_dir = subjects_dir / "fsaverage"
    if not fsaverage_dir.exists():
        raise FileNotFoundError(
            "fsaverage template not found. Run:\n"
            "python -c \"import mne; mne.datasets.fetch_fsaverage(subjects_dir=r'YOUR_SUBJECTS_DIR')\""
        )


def load_roi_specs(roi_config_path: Path | None = None) -> Tuple[ROI, ...]:
    config_path = roi_config_path or DEFAULT_ROI_CONFIG_PATH
    if not config_path.exists():
        return ROI_SPECS

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    roi_items = payload["rois"] if isinstance(payload, dict) else payload
    roi_specs = []
    for item in roi_items:
        roi_specs.append(
            ROI(
                name=str(item["name"]),
                group=str(item["group"]),
                label_names=tuple(item["label_names"]),
            )
        )
    if not roi_specs:
        raise ValueError(f"No ROI definitions found in {config_path}")
    return tuple(roi_specs)


def _combine_label_set(labels: Sequence[mne.Label], label_names: Sequence[str]) -> mne.Label:
    label_map = {label.name: label for label in labels}
    selected = [label_map[name] for name in label_names if name in label_map]
    if not selected:
        missing = ", ".join(label_names)
        raise ValueError(f"Could not find any matching labels for: {missing}")

    combined = selected[0].copy()
    for label in selected[1:]:
        combined += label
    return combined


def load_roi_labels(
    subjects_dir: Path,
    roi_config_path: Path | None = None,
) -> Tuple[List[mne.Label], List[str], List[str]]:
    ensure_fsaverage(subjects_dir)
    roi_specs = load_roi_specs(roi_config_path)
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc",
        subjects_dir=subjects_dir,
        verbose="error",
    )
    combined_labels = []
    roi_names = []
    roi_groups = []
    for roi in roi_specs:
        combined_labels.append(_combine_label_set(labels, roi.label_names))
        roi_names.append(roi.name)
        roi_groups.append(roi.group)
    return combined_labels, roi_names, roi_groups


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    z_threshold: float = 3.5,
) -> List[str]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[idx] for idx in eeg_picks]
    data = raw.get_data(picks=eeg_picks)
    if data.shape[0] < 4:
        return []

    channel_std = np.std(data, axis=1)
    diff_std = np.std(np.diff(data, axis=1), axis=1)
    median_trace = np.median(data, axis=0)
    correlations = np.array(
        [
            np.corrcoef(channel, median_trace)[0, 1] if np.std(channel) > 0 and np.std(median_trace) > 0 else 0.0
            for channel in data
        ]
    )
    correlations = np.nan_to_num(correlations, nan=0.0)

    bad_mask = (
        (np.abs(_robust_zscore(np.log10(channel_std + 1e-12))) > z_threshold)
        | (np.abs(_robust_zscore(np.log10(diff_std + 1e-12))) > z_threshold)
        | (_robust_zscore(correlations) < -z_threshold)
    )
    return sorted([name for name, is_bad in zip(eeg_names, bad_mask) if is_bad])


def _select_frontal_proxy_channels(
    raw: mne.io.BaseRaw,
    frontal_proxy_channels: Sequence[str] | None = None,
) -> List[str]:
    requested = tuple(frontal_proxy_channels or DEFAULT_FRONTAL_PROXY_CHANNELS)
    return [ch for ch in requested if ch in raw.ch_names]


def _classify_ica_components(
    raw: mne.io.BaseRaw,
    ica: ICA,
    frontal_proxy_channels: Sequence[str],
    # ICLabel 기준: eye + muscle artifact 성분 제거
    icalabel_threshold: float = 0.8,
    # ICLabel 없을 때 fallback: 전두엽 상관관계 기준
    proxy_corr_threshold: float = 0.35,
    max_components: int = 6,
) -> List[int]:
    """
    ICLabel(MARA 대체)로 artifact ICA 성분 탐지.
    mne-icalabel 미설치 시 전두엽 proxy 상관관계 방식으로 자동 fallback.

    제거 대상: 'eye blink', 'muscle artifact' (확률 >= icalabel_threshold)
    """
    try:
        from mne_icalabel import label_components
        # ICLabel은 CAR 참조 상태를 요구하므로 임시 적용
        raw_car = raw.copy().set_eeg_reference("average", projection=False, verbose="error")
        labels = label_components(raw_car, ica, method="iclabel")
        del raw_car
        exclude = []
        artifact_types = {"eye blink", "muscle artifact"}
        for i, (label, proba) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
            if label in artifact_types and max(proba) >= icalabel_threshold:
                exclude.append(i)
        return exclude[:max_components]
    except ImportError:
        # fallback: 전두엽 proxy 상관관계
        return _detect_eog_like_components(raw, ica, frontal_proxy_channels,
                                            corr_threshold=proxy_corr_threshold,
                                            max_components=max_components)


def _detect_eog_like_components(
    raw: mne.io.BaseRaw,
    ica: ICA,
    frontal_proxy_channels: Sequence[str],
    corr_threshold: float = 0.35,
    max_components: int = 4,
) -> List[int]:
    if not frontal_proxy_channels:
        return []

    proxy = raw.copy().pick(frontal_proxy_channels).get_data().mean(axis=0)
    proxy = (proxy - proxy.mean()) / (proxy.std() + 1e-12)
    sources = ica.get_sources(raw).get_data()

    scores = []
    for idx, component in enumerate(sources):
        standardized = (component - component.mean()) / (component.std() + 1e-12)
        corr = np.corrcoef(standardized, proxy)[0, 1]
        scores.append(0.0 if np.isnan(corr) else abs(float(corr)))

    if not scores:
        return []

    scores_arr = np.asarray(scores)
    adaptive_threshold = max(corr_threshold, float(np.median(scores_arr) + 2.5 * np.std(scores_arr)))
    ranked = np.argsort(scores_arr)[::-1]
    selected = [int(idx) for idx in ranked if scores_arr[idx] >= adaptive_threshold]
    return selected[:max_components]


def preprocess_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    notch_freqs: Sequence[float] = (60.0,),
    resample_sfreq: float | None = 250.0,
    detect_bad_channels_flag: bool = True,
    interpolate_bads_flag: bool = True,
    run_ica: bool = True,
    frontal_proxy_channels: Sequence[str] | None = None,
    random_state: int = 97,
) -> tuple[mne.io.BaseRaw, PreprocessingSummary]:
    raw = raw.copy().load_data()
    raw.pick("eeg")
    # 몬타주가 없을 때만 standard_1020 적용 (MFF 로더가 이미 GSN-64를 설정한 경우 유지)
    if raw.get_montage() is None:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")

    if notch_freqs:
        raw.notch_filter(freqs=list(notch_freqs), verbose="error")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="error")
    if resample_sfreq is not None:
        raw.resample(resample_sfreq)

    bad_channels = detect_bad_channels(raw) if detect_bad_channels_flag else []
    if bad_channels:
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bad_channels))
    if raw.info["bads"] and interpolate_bads_flag:
        raw.interpolate_bads(reset_bads=False, verbose="error")

    proxy_channels = _select_frontal_proxy_channels(raw, frontal_proxy_channels)
    excluded_components: List[int] = []
    if run_ica and len(raw.ch_names) >= 16:
        n_components = min(len(raw.ch_names) - 1, 25)
        ica = ICA(
            n_components=n_components,
            method="infomax",
            fit_params=dict(extended=True),  # ICLabel 권장 방식
            max_iter="auto",
            random_state=random_state,
        )
        ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose="error")
        excluded_components = _classify_ica_components(raw, ica, proxy_channels)
        if excluded_components:
            ica.exclude = excluded_components
            raw = ica.apply(raw.copy(), verbose="error")

    raw.set_eeg_reference("average", projection=True, verbose="error")
    summary = PreprocessingSummary(
        sfreq=float(raw.info["sfreq"]),
        n_channels=len(raw.ch_names),
        bad_channels=tuple(raw.info["bads"]),
        frontal_proxy_channels=tuple(proxy_channels),
        ica_excluded_components=tuple(excluded_components),
    )
    return raw, summary


def save_preprocessed_raw(
    raw: mne.io.BaseRaw,
    output_dir: Path,
    subject: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject}_preprocessed_raw.fif"
    raw.save(out_path, overwrite=True)
    return out_path


def save_preprocessing_summary(
    output_dir: Path,
    subject: str,
    summary: PreprocessingSummary,
    extra: Dict[str, object] | None = None,
) -> Path:
    payload: Dict[str, object] = {
        "subject": subject,
        "sfreq": summary.sfreq,
        "n_channels": summary.n_channels,
        "bad_channels": list(summary.bad_channels),
        "frontal_proxy_channels": list(summary.frontal_proxy_channels),
        "ica_excluded_components": list(summary.ica_excluded_components),
    }
    if extra:
        payload.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "preprocessing_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def build_inverse_operator(
    raw: mne.io.BaseRaw,
    subjects_dir: Path,
    cache_dir: Path,
    spacing: str = "ico5",
    loose: float = 0.2,
    depth: float = 0.8,
) -> object:
    ensure_fsaverage(subjects_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    inv_path = cache_dir / f"fsaverage_{spacing}-inv.fif"
    if inv_path.exists():
        return read_inverse_operator(inv_path)

    fsaverage_dir = subjects_dir / "fsaverage" / "bem"
    precomputed_src_path = fsaverage_dir / "fsaverage-ico-5-src.fif"
    precomputed_bem_path = fsaverage_dir / "fsaverage-5120-5120-5120-bem-sol.fif"

    if spacing.lower() in {"ico5", "ico-5"} and precomputed_src_path.exists():
        src = read_source_spaces(precomputed_src_path, verbose="error")
    else:
        src = mne.setup_source_space(
            subject="fsaverage",
            spacing=spacing,
            add_dist=False,
            subjects_dir=subjects_dir,
            verbose="error",
        )

    if precomputed_bem_path.exists():
        bem = read_bem_solution(precomputed_bem_path, verbose="error")
    else:
        bem_model = mne.make_bem_model(
            subject="fsaverage",
            ico=4,
            conductivity=(0.3,),
            subjects_dir=subjects_dir,
            verbose="error",
        )
        bem = mne.make_bem_solution(bem_model, verbose="error")

    fwd = mne.make_forward_solution(
        info=raw.info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=5.0,
        verbose="error",
    )

    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=2.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )
    noise_cov = mne.compute_covariance(
        epochs,
        method=["shrunk", "empirical"],
        rank="info",
        verbose="error",
    )
    inverse_operator = make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=loose,
        depth=depth,
        verbose="error",
    )
    write_inverse_operator(inv_path, inverse_operator, overwrite=True)
    return inverse_operator


def compute_source_estimate(
    raw: mne.io.BaseRaw,
    inverse_operator: object,
    method: str,
    snr: float = 3.0,
):
    lambda2 = 1.0 / (snr**2)
    return apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
        verbose="error",
    )


def extract_roi_time_series(
    stc,
    inverse_operator: object,
    subjects_dir: Path,
    roi_config_path: Path | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    labels, roi_names, roi_groups = load_roi_labels(subjects_dir, roi_config_path=roi_config_path)
    src = inverse_operator["src"]
    time_courses = mne.extract_label_time_course(
        stc,
        labels,
        src,
        mode="pca_flip",
        verbose="error",
    )
    df = pd.DataFrame(time_courses.T, columns=roi_names)
    return df, roi_groups


def save_stc(stc, output_dir: Path, method: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{method.lower()}_source_estimate"
    for suffix in ("-lh.stc", "-rh.stc"):
        existing = stem.parent / f"{stem.name}{suffix}"
        if existing.exists():
            existing.unlink()
    stc.save(stem, overwrite=True)
    return stem


def save_roi_time_series(
    roi_df: pd.DataFrame,
    output_dir: Path,
    sfreq: float,
    roi_groups: Sequence[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_seconds = np.arange(len(roi_df)) / sfreq
    export_df = roi_df.copy()
    export_df.insert(0, "time_sec", time_seconds)
    export_df.to_csv(output_dir / "roi_time_series.csv", index=False)

    metadata = pd.DataFrame({"roi": roi_df.columns, "group": list(roi_groups)})
    metadata.to_csv(output_dir / "roi_metadata.csv", index=False)
    return output_dir / "roi_time_series.csv"


def load_roi_time_series_csv(path: Path) -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    if "time_sec" not in df.columns:
        raise ValueError(f"{path} must contain a time_sec column")
    time_sec = df["time_sec"].to_numpy()
    if len(time_sec) < 2:
        raise ValueError(f"{path} must contain at least two time points")
    sfreq = 1.0 / np.median(np.diff(time_sec))
    roi_df = df.drop(columns=["time_sec"])
    return roi_df, sfreq


def bandpass_filter(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(4, [l_freq / nyq, h_freq / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def compute_band_power(roi_df: pd.DataFrame, sfreq: float) -> pd.DataFrame:
    rows = []
    for roi_name in roi_df.columns:
        signal = roi_df[roi_name].to_numpy()
        freqs, psd = welch(signal, fs=sfreq, nperseg=min(2048, len(signal)))
        total_power = np.trapezoid(psd, freqs)
        for band_name, (fmin, fmax) in CANONICAL_BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            band_power = np.trapezoid(psd[mask], freqs[mask]) if np.any(mask) else 0.0
            rows.append(
                {
                    "roi": roi_name,
                    "band": band_name,
                    "abs_power": band_power,
                    "rel_power": band_power / total_power if total_power > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_fc_matrices(
    roi_df: pd.DataFrame,
    sfreq: float,
    method: str = "aec",
    bands: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, pd.DataFrame]:
    bands = bands or CANONICAL_BANDS
    data = roi_df.to_numpy().T
    results: Dict[str, pd.DataFrame] = {}
    for band_name, (fmin, fmax) in bands.items():
        filtered = bandpass_filter(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax)
        if method == "aec":
            fc = envelope_correlation(filtered[np.newaxis, :, :], orthogonalize="pairwise")
            matrix = np.squeeze(fc.get_data(output="dense"))
        elif method == "corr":
            analytic = hilbert(filtered, axis=-1)
            amplitude = np.abs(analytic)
            matrix = np.corrcoef(amplitude)
        elif method == "coh":
            n_rois, n_samples = data.shape
            matrix = np.eye(n_rois, dtype=float)
            nperseg = min(1024, n_samples)
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    freqs, coh_values = coherence(
                        data[i],
                        data[j],
                        fs=sfreq,
                        nperseg=nperseg,
                    )
                    band_mask = (freqs >= fmin) & (freqs < fmax)
                    coh_score = float(np.mean(coh_values[band_mask])) if np.any(band_mask) else 0.0
                    matrix[i, j] = coh_score
                    matrix[j, i] = coh_score
        else:
            raise ValueError("method must be one of: 'aec', 'corr', 'coh'")

        matrix = np.nan_to_num(matrix, nan=0.0)
        np.fill_diagonal(matrix, 1.0)
        results[band_name] = pd.DataFrame(matrix, index=roi_df.columns, columns=roi_df.columns)
    return results


# ── Epoch 균등화 ──────────────────────────────────────────────────────────────

def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    epoch_duration: float = 2.0,
    amplitude_threshold: float = 150e-6,
    flat_threshold: float = 1e-6,
    flat_min_duration: float = 0.5,
) -> mne.Epochs:
    """
    Continuous raw → 고정 길이 epoch 분할 후 artifact epoch 제거.

    Parameters
    ----------
    raw : 전처리 완료된 raw
    epoch_duration : epoch 길이 (초), 기본 2.0
    amplitude_threshold : peak-to-peak 기준 (V), 기본 150 µV
    flat_threshold : flat signal 기준 (V), 기본 1 µV
    flat_min_duration : flat 판정 최소 지속 시간 (초)

    Returns
    -------
    epochs : artifact-free epochs
    """
    import mne
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
        verbose=False,
    )
    reject = {"eeg": amplitude_threshold}
    flat = {"eeg": flat_threshold}
    epochs.drop_bad(reject=reject, flat=flat, verbose=False)
    return epochs


def equalize_epoch_counts(
    epochs_dict: Dict[str, mne.Epochs],
    random_state: int = 97,
) -> Dict[str, mne.Epochs]:
    """
    여러 피험자/조건의 epoch 수를 최솟값 기준으로 맞춤.
    FC 분석에서 데이터 길이 차이로 인한 편향 제거.

    Parameters
    ----------
    epochs_dict : {"subject_id": epochs, ...}
    random_state : 재현성을 위한 랜덤 시드

    Returns
    -------
    균등화된 epochs dict
    """
    counts = {k: len(v) for k, v in epochs_dict.items()}
    min_count = min(counts.values())
    rng = np.random.default_rng(random_state)

    result = {}
    for key, epochs in epochs_dict.items():
        n = len(epochs)
        if n > min_count:
            idx = rng.choice(n, size=min_count, replace=False)
            idx = np.sort(idx)
            result[key] = epochs[idx]
        else:
            result[key] = epochs
    return result, min_count, counts


def save_clean_epochs(
    epochs: mne.Epochs,
    output_dir: Path,
    subject_key: str,
) -> Path:
    """전처리된 epoch을 FIF으로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject_key}_clean_epochs-epo.fif"
    epochs.save(path, overwrite=True, verbose=False)
    return path


def save_fc_to_excel(
    matrices: Dict[str, pd.DataFrame],
    output_path: Path,
    roi_groups: Sequence[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_df = pd.DataFrame({"roi": matrices[next(iter(matrices))].index, "group": list(roi_groups)})
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        group_df.to_excel(writer, sheet_name="roi_groups", index=False)
        for band_name, matrix_df in matrices.items():
            matrix_df.to_excel(writer, sheet_name=band_name)
            long_df = (
                matrix_df.stack()
                .rename("fc")
                .reset_index()
                .rename(columns={"level_0": "roi_from", "level_1": "roi_to"})
            )
            long_df.to_excel(writer, sheet_name=f"{band_name}_long", index=False)
    return output_path


def get_output_dir(base_output_dir: Path, subject: str, method: str) -> Path:
    return base_output_dir / subject / method.lower()


def save_run_metadata(
    output_dir: Path,
    subject: str,
    inverse_method: str,
    sfreq: float,
    extra: Dict[str, object] | None = None,
) -> Path:
    payload = {"subject": subject, "inverse_method": inverse_method, "sfreq": sfreq}
    if extra:
        payload.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_metadata.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
