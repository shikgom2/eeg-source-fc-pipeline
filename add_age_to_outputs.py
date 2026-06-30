"""
qc_table.csv의 age_months를 index_table.csv, graph_metrics.csv에 병합
"""
import pandas as pd
from pathlib import Path

QC_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\qc_table.csv")
FC_TENSOR_DIR = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\fc_tensor")
GRAPH_METRICS_PATH = Path(r"D:\Github\eeg-source-fc-pipeline\outputs\graph_metrics.csv")

qc = pd.read_csv(QC_PATH)[["subject_key", "age_months"]]

index_table = pd.read_csv(FC_TENSOR_DIR / "index_table.csv")
if "age_months" in index_table.columns:
    index_table = index_table.drop(columns=["age_months"])
index_table = index_table.merge(qc, on="subject_key", how="left")
index_table.to_csv(FC_TENSOR_DIR / "index_table.csv", index=False, encoding="utf-8-sig")
print(f"index_table.csv 업데이트: {len(index_table)}행, age_months 결측 {index_table['age_months'].isna().sum()}개")

graph_metrics = pd.read_csv(GRAPH_METRICS_PATH)
if "age_months" in graph_metrics.columns:
    graph_metrics = graph_metrics.drop(columns=["age_months"])
graph_metrics = graph_metrics.merge(qc, on="subject_key", how="left")
graph_metrics.to_csv(GRAPH_METRICS_PATH, index=False, encoding="utf-8-sig")
print(f"graph_metrics.csv 업데이트: {len(graph_metrics)}행, age_months 결측 {graph_metrics['age_months'].isna().sum()}개")
