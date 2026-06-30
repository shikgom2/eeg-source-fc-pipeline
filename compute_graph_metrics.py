"""
graph_metrics[subject, time, band] 생성
- global efficiency, local efficiency, modularity, mean strength
- wPLI, AEC-c, imaginary coherence 각각에 대해 계산
"""
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

FC_TENSOR_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor")
OUTPUT_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\graph_metrics.csv")

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
METHODS = ["wpli", "aec", "imcoh"]


def global_efficiency_from_weight_matrix(w: np.ndarray) -> float:
    """w: 비음수 가중치 행렬 (대각 0). scipy Floyd-Warshall로 최단거리 계산."""
    n = w.shape[0]
    if n < 2:
        return np.nan
    with np.errstate(divide="ignore"):
        dist = np.where(w > 0, 1.0 / w, np.inf)
    np.fill_diagonal(dist, 0)
    sp = shortest_path(dist, method="FW", directed=False)
    mask = ~np.eye(n, dtype=bool)
    finite = np.isfinite(sp) & mask & (sp > 0)
    if not finite.any():
        return 0.0
    inv = np.zeros_like(sp)
    inv[finite] = 1.0 / sp[finite]
    return float(inv.sum() / (n * (n - 1)))


def local_efficiency_from_weight_matrix(w: np.ndarray) -> float:
    n = w.shape[0]
    effs = []
    for i in range(n):
        neighbors = np.where(w[i] > 0)[0]
        if len(neighbors) < 2:
            effs.append(0.0)
            continue
        sub = w[np.ix_(neighbors, neighbors)]
        effs.append(global_efficiency_from_weight_matrix(sub))
    return float(np.mean(effs)) if effs else np.nan


def matrix_to_graph(mat: np.ndarray, threshold: float = 0.0) -> nx.Graph:
    """대칭 비음수 연결성 행렬 -> weighted graph. distance = 1/weight."""
    n = mat.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(mat[i, j])
            if w > threshold:
                G.add_edge(i, j, weight=w, distance=1.0 / w)
    return G


def compute_metrics_for_matrix(mat: np.ndarray) -> dict:
    if np.isnan(mat).all():
        return {"global_efficiency": np.nan, "local_efficiency": np.nan,
                "modularity": np.nan, "mean_strength": np.nan}

    w = np.abs(mat).copy()
    np.fill_diagonal(w, 0)
    n = w.shape[0]

    strengths = w.sum(axis=1) / (n - 1)
    mean_strength = float(np.mean(strengths))

    if not (w > 0).any():
        return {"global_efficiency": 0.0, "local_efficiency": 0.0,
                "modularity": np.nan, "mean_strength": mean_strength}

    ge = global_efficiency_from_weight_matrix(w)
    le = local_efficiency_from_weight_matrix(w)

    try:
        G = matrix_to_graph(mat)
        partition = community_louvain.best_partition(G, weight="weight", random_state=42)
        modularity = community_louvain.modularity(partition, G, weight="weight")
    except Exception:
        modularity = np.nan

    return {"global_efficiency": ge, "local_efficiency": le,
            "modularity": modularity, "mean_strength": mean_strength}


def main():
    index_table = pd.read_csv(FC_TENSOR_DIR / "index_table.csv")
    rows = []

    for method in METHODS:
        tensor = np.load(FC_TENSOR_DIR / f"FC_tensor_{method}.npy")  # (n_obs, n_band, 68, 68)
        n_obs = tensor.shape[0]
        print(f"=== {method} ===")
        for i in range(n_obs):
            meta = index_table.iloc[i]
            for b, band in enumerate(BANDS):
                mat = tensor[i, b]
                m = compute_metrics_for_matrix(mat)
                rows.append({
                    "subject_key": meta["subject_key"],
                    "subject_id": meta["subject_id"],
                    "timepoint": meta["timepoint"],
                    "label": meta["label"],
                    "method": method,
                    "band": band,
                    **m,
                })
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n_obs}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"총 행: {len(df)} (={n_obs}명 x {len(BANDS)}밴드 x {len(METHODS)}방법)")


if __name__ == "__main__":
    main()
