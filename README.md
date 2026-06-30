# NEST EEG Source-Space FC Pipeline

A preprocessing → source reconstruction → FC (functional connectivity) analysis
pipeline for 64-channel resting-state EEG from the NEST cohort (ASD/TD, ages 5-8).

## Directory structure

```
eeg-source-fc-pipeline/
    source_fc_utils.py          # shared utilities (preprocessing, MFF loading, etc.)
    batch_preprocess_nest.py    # 1. preprocessing batch
    equalize_epochs.py          # 2. epoch count equalization
    make_forward_model.py       # 3. forward model (BEM + source space + montage)
    make_inverse_operator.py    # 4. inverse operator (eLORETA, per subject)
    make_source_roi_timeseries.py  # 5. source extraction + DK 68-ROI averaging
    compute_fc.py                # 6. FC computation (wPLI, AEC-c, imaginary coherence)
    build_fc_tensor.py           # 7. FC_tensor + index_table generation
    compute_graph_metrics.py     # 8. graph metrics generation
    build_qc_table.py            # 9. qc_table generation
    add_age_to_outputs.py        # 10. merge age_months into the other outputs
    visualize_bem.py             # BEM visualization
    visualize_fc_heatmap.py      # ASD/TD FC heatmap visualization
    outputs/                     # all artifacts (preprocessing through FC tensor)
```

## Pipeline order

### 1. Preprocessing

```powershell
python batch_preprocess_nest.py --timepoints 1 --overwrite
python batch_preprocess_nest.py --timepoints 2 --overwrite
```

- MFF loading, line-noise removal, 1-45Hz bandpass, automatic bad-channel detection + interpolation
- ICA (extended infomax) + ICLabel-based removal of ocular/muscle components
- 2-second fixed-length epoching
- Output: `outputs/preprocessed/{subject}_T{n}_{condition}/`

### 2. Epoch equalization

```powershell
python equalize_epochs.py --min-epochs 40 --overwrite
```

- Excludes subjects with fewer than 40 epochs (80s)
- Remaining subjects are randomly subsampled down to 40 epochs each (prevents FC bias from unequal data length)
- Output: `*_equalized-epo.fif`

### 3. Head model (built once on the server; already done)

The NIHPD 4.5-8.5-year pediatric MRI template was processed with FreeSurfer and
is stored at `outputs/freesurfer_subjects/nihpd_4.5-8.5/` (recon-all + 3-layer watershed BEM).
Only re-run this on the server if the head model needs to be rebuilt.

### 4. Forward model

```powershell
python make_forward_model.py
```

GSN-HydroCel-64 montage + BEM + source space (oct6, 8196 sources) → `outputs/head_model/nihpd_4.5-8.5-fwd.fif`

### 5. Inverse operator (per subject, 284 subjects)

```powershell
python make_inverse_operator.py --overwrite
```

Per-subject noise covariance + eLORETA inverse operator → `outputs/inverse/{subject}_inv.fif`

### 6. Source extraction + DK 68-ROI averaging

```powershell
python make_source_roi_timeseries.py --overwrite
```

Applies eLORETA → mean-flip averaging into Desikan-Killiany 68 ROIs → `outputs/roi_timeseries/{subject}_roi_timeseries.npy` (40 epochs, 68 ROIs, 501 timepoints)

### 7. FC computation

```powershell
python compute_fc.py --overwrite
```

- wPLI (primary), AEC-c (secondary, leakage-orthogonalized), imaginary coherence (auxiliary)
- Bands: delta/theta/alpha/beta/gamma (theta and alpha are the primary hypothesis bands; delta is exploratory due to epoch-length limits)
- Output: `outputs/fc/{subject}_fc_{method}_{band}.npy` (68x68)

### 8. Final integrated outputs

```powershell
python build_fc_tensor.py
python compute_graph_metrics.py
python build_qc_table.py
python add_age_to_outputs.py
```

| File | Contents |
|---|---|
| `outputs/fc_tensor/FC_tensor_{wpli,aec,imcoh}.npy` | (284, 5 bands, 68, 68) connectivity tensor |
| `outputs/fc_tensor/index_table.csv` | mapping from tensor axis 0 to subject/timepoint/label |
| `outputs/graph_metrics.csv` | global/local efficiency, modularity, mean strength |
| `outputs/qc_table.csv` | clean_epochs, restValidity, max_impedance, movement_flag, age_months |

## Known limitations

- Both T1 and T2 contain only eyes-open (resting_pre) data. Eyes-closed (resting_post) data is essentially absent (only 3 subjects), so separate eyes-open/closed analysis is not possible
- The delta band (1-4Hz) falls short of the 5-cycle criterion (5+ seconds recommended) with 2-second epochs, so it is treated as exploratory only
- The head model uses the NIHPD 4.5-8.5-year combined template rather than individual MRI (no publicly available 1-year-resolution age-specific templates were found)
