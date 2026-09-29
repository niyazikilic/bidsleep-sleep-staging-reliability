# Reliability, personalization, and temporal post-processing in wearable-derived sleep staging

Analysis code for the manuscript:

> Kılıç N. Reliability, personalization, and temporal post-processing in wearable-derived sleep staging: a secondary analysis of a multi-night dataset. (Submitted to BMJ Open.)

This repository contains the analysis code used to produce all reported results, tables, and figures. It does not contain patient data.

## Data

This is a secondary analysis of a publicly available dataset:

> Song TA, Zhang Y, Zhou Z, Dutta J. A multi-night instantaneous heart rate and accelerometry dataset with EEG sleep stage labels (version 1.0.1). PhysioNet. https://doi.org/10.13026/rees-1092

To reproduce the analysis, download the dataset from PhysioNet and place the extracted subject-night folders under `./data/` (or point `BASE_DIR` in `src/build_full_dataset.py` to your local copy).

## Environment

```
pip install -r requirements.txt
```

Developed and run with Python 3.13.15. See `requirements.txt` for exact package versions used to produce the results reported in the manuscript.

## Pipeline

Scripts are run in the following order. Each step reads the output of the previous one from `./outputs/`.

1. **`build_full_dataset.py`** — Loads raw subject-night folders (via `data_loader.py`, `feature_extraction.py`), applies the duration and completeness quality filters, removes epochs with an unknown reference label, consolidates the five AASM stages into four classes, and writes `./data/full_feature_table.parquet`.
   *(Methods: Dataset and Participants; Reference Labels; Data Quality Filtering; Feature Extraction and Preprocessing.)*

2. **`per_night_final_model_evaluation.py`** — Trains the Random Forest classifier with five-fold subject-wise (GroupKFold) cross-validation and computes per-(participant, night) Cohen's kappa and macro-F1 for the final model. Writes `./outputs/per_night_final_model.csv`, which is a shared input for steps 3 and 4.
   *(Methods: Model Architecture and Cross-Validation.)*

3. **`reliability_comparison_v2.py`** — Analysis 1: computes the naive (overlapping-reference) and corrected (leave-N-out) reliability curves. Produces Figure 2.
   *(Methods: Analysis 1.)*

4. **`variability_analysis.py`** — Analysis 2: fits the mixed-effects model (night order as a fixed effect, random participant intercept) and the classical two-night ICC(1,1). Produces the variance-decomposition values reported in Results and Figure 3 (with `resultviz_figure2_variance.py`).
   *(Methods: Analysis 2.)*

5. **`cumulative_training_experiment.py`** — Analysis 3: limits the training partition to each participant's earliest K nights (K = 1–7) and re-evaluates performance. Produces Table 2 (with `table2_extra_column.py` adding the mean-nights-actually-used column).
   *(Methods: Analysis 3.)*

6. **`personalization_dose_response_v2.py`** — Analysis 4: adds 0–4 of a target participant's own earliest nights as weighted calibration data and re-evaluates on the held-out final night, for both the raw and HMM-smoothed model. Produces Figure 4 (with `figure4_personalization_standalone.py`).
   *(Methods: Analysis 4.)*

7. **`hmm_smoothing.py`** + **`three_way_postprocessing_comparison.py`** — Analysis 5: compares raw predictions, the moving mode filter, and Hidden Markov Model / Viterbi decoding. Produces Table 3.
   *(Methods: Analysis 5.)*

8. **`per_class_performance.py`** — Computes per-class precision/recall/F1 and the confusion matrix for the final RF+HMM model. Produces Table 4.

9. **`uncertainty_analysis.py`** — Shared utility providing the subject-clustered bootstrap confidence intervals used throughout Results (participants resampled as clusters, 1000–2000 iterations).
   *(Methods: Statistical Analysis.)*

10. **`signal_example_visualization.py`** — Produces Figure 1 (representative raw signals and night-to-night hypnogram variability).

11. **`table1_extra_stats.py`** — Adds descriptive statistics to Table 1.

## What is not included

Earlier exploratory scripts that were superseded during the analysis (an initial naive-only reliability design without the leave-N-out correction; an abandoned leave-one-subject-out cross-validation scheme, later replaced by five-fold GroupKFold; earlier draft versions of the personalization and figure-generation scripts) are not included, to avoid confusion between draft and final analyses. The scripts listed above are the ones that produced the numbers reported in the manuscript.

## Notes on reproducibility

- All cross-validation is subject-wise (`GroupKFold`); no participant's nights appear in both the training and test partition of the same fold.
- Random Forest hyperparameters were fixed in advance and not tuned.
- Bootstrap and cross-validation procedures use fixed random seeds where set explicitly in each script; minor numerical differences may occur across package versions (see `requirements.txt`).
- The trained models themselves are not included in this repository; re-running the pipeline retrains them from the public dataset.

## License

MIT License (see `LICENSE`).

## Contact

Niyazi Kılıç, Department of Electrical and Electronics Engineering, Faculty of Engineering, Istanbul University-Cerrahpaşa, Istanbul, Türkiye.
Email: niyazi.kilic@iuc.edu.tr
