"""
UC YONLU KARSILASTIRMA: RF-HAM vs MODE-FILTER (kullanicinin kendi
yontemi) vs HMM/VITERBI

Kullanicinin bagimsiz calismasinda kullandigi "hareketli cogunluk oylamasi"
(mode/majority filter) yontemini, bizim RF+HMM yaklasimimizla ayni
cerceve icinde, ayni test verisinde, bootstrap CI ile karsilastirir.

Mode filter: her epoch'un etrafindaki bir pencerede (varsayilan: 5 epoch,
yani +-2) en sik gorulen tahmini o epoch'un yeni tahmini yapar. Basit
ama HMM'den farkli olarak OGRENILMIS GECIS OLASILIKLARI kullanmaz,
sadece yerel cogunluga bakar.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from hmm_smoothing import estimate_transition_matrix, estimate_start_probs, viterbi_smooth

RNG = np.random.default_rng(2026)


def mode_filter(predictions, window_size=5):
    """
    Hareketli cogunluk oylamasi (majority/mode filter).
    window_size TEK sayi olmali (merkezi pencere).
    """
    half_win = window_size // 2
    n = len(predictions)
    smoothed = np.zeros(n, dtype=int)
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        window_vals = predictions[start:end]
        smoothed[i] = scipy_stats.mode(window_vals, keepdims=False).mode
    return smoothed


def run_three_way_comparison(full_df, feature_cols, label_col="expert_label",
                                n_splits=5, n_estimators=150, random_state=42,
                                mode_filter_window=5, checkpoint_dir=None):
    """
    5-Fold CV: RF egitilir, UC farkli son-isleme stratejisi ayni test
    verisinde karsilastirilir: (1) ham, (2) mode filter, (3) HMM/Viterbi.
    """
    df = full_df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    subjects = df["subject"].unique()
    n_states = int(df[label_col].max()) + 1

    gkf = GroupKFold(n_splits=n_splits)
    fold_splits = list(gkf.split(subjects, groups=subjects))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        checkpoint_path = f"{checkpoint_dir}/fold_{fold_idx}.csv" if checkpoint_dir else None
        if checkpoint_path and os.path.exists(checkpoint_path):
            all_results.append(pd.read_csv(checkpoint_path))
            print(f"  Fold {fold_idx+1}/{n_splits} checkpoint'ten yuklendi (atlandi)")
            continue

        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]

        train_df = df[df["subject"].isin(train_subjects)]
        test_df = df[df["subject"].isin(test_subjects)]

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values.astype(int)
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10,
                                       class_weight="balanced", random_state=random_state, n_jobs=-1)
        clf.fit(X_train_s, y_train)

        train_sequences = [g.sort_values("epoch")[label_col].values.astype(int)
                            for _, g in train_df.groupby(["subject", "night"])]
        trans_matrix = estimate_transition_matrix(train_sequences, n_states=n_states)
        start_probs = estimate_start_probs(train_sequences, n_states=n_states)

        test_df_copy = test_df.copy()
        proba_all = clf.predict_proba(X_test_s)
        raw_preds = clf.predict(X_test_s)
        test_df_copy["pred_raw"] = raw_preds

        mode_preds = np.zeros(len(test_df_copy), dtype=int)
        hmm_preds = np.zeros(len(test_df_copy), dtype=int)

        for (subj, night), g in test_df_copy.groupby(["subject", "night"]):
            night_idx = g.sort_values("epoch").index
            positions = [test_df_copy.index.get_loc(i) for i in night_idx]

            night_raw_preds = raw_preds[positions]
            mode_preds[positions] = mode_filter(night_raw_preds, window_size=mode_filter_window)

            night_proba = proba_all[positions]
            hmm_preds[positions] = viterbi_smooth(night_proba, trans_matrix, start_probs)

        test_df_copy["pred_mode"] = mode_preds
        test_df_copy["pred_hmm"] = hmm_preds

        fold_rows = []
        for subj, g in test_df_copy.groupby("subject"):
            fold_rows.append({
                "fold": fold_idx, "subject": subj,
                "kappa_raw": cohen_kappa_score(g[label_col], g["pred_raw"]),
                "kappa_mode": cohen_kappa_score(g[label_col], g["pred_mode"]),
                "kappa_hmm": cohen_kappa_score(g[label_col], g["pred_hmm"]),
                "f1_macro_raw": f1_score(g[label_col], g["pred_raw"], average="macro", zero_division=0),
                "f1_macro_mode": f1_score(g[label_col], g["pred_mode"], average="macro", zero_division=0),
                "f1_macro_hmm": f1_score(g[label_col], g["pred_hmm"], average="macro", zero_division=0),
            })
        fold_result = pd.DataFrame(fold_rows)
        all_results.append(fold_result)

        if checkpoint_path:
            fold_result.to_csv(checkpoint_path, index=False)

        print(f"  Fold {fold_idx+1}/{n_splits} tamamlandi" + (" (kaydedildi)" if checkpoint_path else ""))

    return pd.concat(all_results, ignore_index=True)


def bootstrap_pairwise_ci(results_df, metric_a, metric_b, n_boot=1000, ci=0.95, rng=RNG):
    """Kumeli bootstrap ile iki metrik arasindaki farkin guven araligi (metric_a - metric_b)."""
    subjects = results_df["subject"].unique()
    boot_diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        vals_a = np.concatenate([results_df.loc[results_df["subject"]==s, metric_a].values for s in sampled])
        vals_b = np.concatenate([results_df.loc[results_df["subject"]==s, metric_b].values for s in sampled])
        boot_diffs.append(vals_a.mean() - vals_b.mean())
    boot_diffs = np.array(boot_diffs)
    point_diff = results_df[metric_a].mean() - results_df[metric_b].mean()
    alpha = 1 - ci
    return {
        "point_diff": point_diff,
        "ci_lower": np.percentile(boot_diffs, 100*alpha/2),
        "ci_upper": np.percentile(boot_diffs, 100*(1-alpha/2)),
        "excludes_zero": (np.percentile(boot_diffs, 100*alpha/2) > 0) or (np.percentile(boot_diffs, 100*(1-alpha/2)) < 0),
    }


if __name__ == "__main__":
    from cumulative_training_experiment import get_feature_cols

    full_df = pd.read_parquet("./data/full_feature_table.parquet")
    feature_cols = get_feature_cols(full_df)

    print("=== UC YONLU KARSILASTIRMA: RF-Ham vs Mode-Filter vs HMM ===\n")
    results_df = run_three_way_comparison(full_df, feature_cols,
                                            checkpoint_dir="/tmp/three_way_ckpt")

    print("\n=== Ozet (ortalama) ===")
    print(results_df[["kappa_raw", "kappa_mode", "kappa_hmm",
                        "f1_macro_raw", "f1_macro_mode", "f1_macro_hmm"]].mean())

    print("\n=== Ikili Karsilastirmalar (Kappa, bootstrap %95 GA) ===")
    for a, b, name in [("kappa_mode", "kappa_raw", "Mode-Filter vs Ham"),
                        ("kappa_hmm", "kappa_raw", "HMM vs Ham"),
                        ("kappa_hmm", "kappa_mode", "HMM vs Mode-Filter")]:
        ci = bootstrap_pairwise_ci(results_df, a, b, n_boot=200)
        sig = "ANLAMLI" if ci["excludes_zero"] else "anlamli degil"
        print(f"{name}: {ci['point_diff']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] -> {sig}")

    results_df.to_csv("./outputs/three_way_comparison_raw.csv", index=False)
    print("\nKaydedildi.")
