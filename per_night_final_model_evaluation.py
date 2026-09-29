"""
GECE-BAZLI NIHAI MODEL PERFORMANSI

Su ana kadarki tum analizlerimiz (RQ2, HMM, RQ3) katilimci basina TEK
bir Kappa uretiyordu (o katilimcinin tum test geceleri birlestirilerek).

RQ1 (guvenilirlik egrisi) ve ICC/karisik model analizi ise HER GECE
ICIN AYRI bir performans degerine ihtiyac duyar - "bu katilimcinin
1. gecesi ne kadar iyi tahmin edildi, 2. gecesi ne kadar iyi tahmin
edildi" gibi.

Bu modul, nihai onerilen model (RF + HMM/Viterbi, 5-fold subject-wise)
ile HER (katilimci, gece) kombinasyonu icin AYRI Kappa/F1 hesaplar.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from hmm_smoothing import estimate_transition_matrix, estimate_start_probs, viterbi_smooth


def per_night_final_model_evaluation(full_df, feature_cols, label_col="expert_label",
                                       n_splits=5, n_estimators=150, random_state=42,
                                       checkpoint_dir=None):
    """
    5-Fold subject-wise CV ile RF+HMM egitir, HER (subject, night) icin
    AYRI Kappa/F1 hesaplar (RQ1 ve ICC analizleri icin gerekli format).
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

        hmm_preds = np.zeros(len(test_df_copy), dtype=int)

        fold_rows = []
        for (subj, night), g in test_df_copy.groupby(["subject", "night"]):
            positions = [test_df_copy.index.get_loc(i) for i in g.sort_values("epoch").index]
            night_proba = proba_all[positions]
            night_pred_hmm = viterbi_smooth(night_proba, trans_matrix, start_probs)

            y_true = g.sort_values("epoch")[label_col].values.astype(int)
            fold_rows.append({
                "fold": fold_idx, "subject": subj, "night": night,
                "n_epochs": len(g),
                "kappa": cohen_kappa_score(y_true, night_pred_hmm),
                "f1_macro": f1_score(y_true, night_pred_hmm, average="macro", zero_division=0),
            })

        fold_result = pd.DataFrame(fold_rows)
        all_results.append(fold_result)

        if checkpoint_path:
            fold_result.to_csv(checkpoint_path, index=False)

        print(f"  Fold {fold_idx+1}/{n_splits} tamamlandi" + (" (kaydedildi)" if checkpoint_path else ""))

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    from cumulative_training_experiment import get_feature_cols

    full_df = pd.read_parquet("./data/full_feature_table.parquet")
    feature_cols = get_feature_cols(full_df)

    print("=== Gece-Bazli Nihai Model Performansi (RF+HMM) ===\n")
    per_night_df = per_night_final_model_evaluation(full_df, feature_cols,
                                                       checkpoint_dir="/tmp/per_night_final_ckpt")

    print(f"\nToplam gece-bazli gozlem: {len(per_night_df)}")
    print(per_night_df.head(10))

    per_night_df.to_csv("./outputs/per_night_final_model.csv", index=False)
    print("\nKaydedildi.")
