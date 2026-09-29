"""
SINIF-BAZLI PERFORMANS (Wake/Light/Deep/REM) - NIHAI MODEL (RF+HMM)

Hakem elestirisine yanit: Macro-F1 tek bir sayi verirken, hangi uyku
evresinin (ozellikle Deep ve REM) daha zor ayirt edildigini gostermez.
Bu modul, 5-fold RF+HMM pipeline'ini calistirir, TUM test epoch'larinin
gercek/tahmin ciftlerini biriktirir, ve sinif-bazli F1 + karisiklik
matrisi hesaplar.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

from hmm_smoothing import estimate_transition_matrix, estimate_start_probs, viterbi_smooth

STAGE_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}


def per_class_performance(full_df, feature_cols, label_col="expert_label",
                            n_splits=5, n_estimators=150, random_state=42,
                            checkpoint_dir=None):
    """
    5-Fold RF+HMM calistirir, TUM foldlardan gercek/tahmin ciftlerini
    (epoch-bazli) biriktirip dondurur - sinif-bazli analiz icin.
    """
    df = full_df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    subjects = df["subject"].unique()
    n_states = int(df[label_col].max()) + 1

    gkf = GroupKFold(n_splits=n_splits)
    fold_splits = list(gkf.split(subjects, groups=subjects))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_true, all_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        checkpoint_path = f"{checkpoint_dir}/fold_{fold_idx}.npz" if checkpoint_dir else None
        if checkpoint_path and os.path.exists(checkpoint_path):
            data = np.load(checkpoint_path)
            all_true.append(data["y_true"])
            all_pred.append(data["y_pred"])
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
        for (subj, night), g in test_df_copy.groupby(["subject", "night"]):
            positions = [test_df_copy.index.get_loc(i) for i in g.sort_values("epoch").index]
            night_proba = proba_all[positions]
            hmm_preds[positions] = viterbi_smooth(night_proba, trans_matrix, start_probs)

        y_true_fold = test_df_copy[label_col].values.astype(int)

        all_true.append(y_true_fold)
        all_pred.append(hmm_preds)

        if checkpoint_path:
            np.savez(checkpoint_path, y_true=y_true_fold, y_pred=hmm_preds)

        print(f"  Fold {fold_idx+1}/{n_splits} tamamlandi" + (" (kaydedildi)" if checkpoint_path else ""))

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    return y_true_all, y_pred_all


def build_per_class_table(y_true, y_pred):
    """Sinif-bazli F1/precision/recall tablosu."""
    report = classification_report(y_true, y_pred, target_names=[STAGE_NAMES[i] for i in sorted(STAGE_NAMES)],
                                     output_dict=True, zero_division=0)
    rows = []
    for stage_code, stage_name in STAGE_NAMES.items():
        r = report[stage_name]
        rows.append({
            "Sleep Stage": stage_name,
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1-score": round(r["f1-score"], 3),
            "Support (epochs)": int(r["support"]),
        })
    return pd.DataFrame(rows)


def build_confusion_matrix_table(y_true, y_pred):
    """4x4 karisiklik matrisi, satir/kolon etiketli, satir-normalize (%)."""
    cm = confusion_matrix(y_true, y_pred, labels=sorted(STAGE_NAMES.keys()))
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = [STAGE_NAMES[i] for i in sorted(STAGE_NAMES.keys())]
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    cm_pct_df = pd.DataFrame(cm_pct.round(1), index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    return cm_df, cm_pct_df


if __name__ == "__main__":
    from cumulative_training_experiment import get_feature_cols

    full_df = pd.read_parquet("./data/full_feature_table.parquet")
    feature_cols = get_feature_cols(full_df)

    print("=== Sinif-Bazli Performans (RF+HMM) ===\n")
    y_true, y_pred = per_class_performance(full_df, feature_cols, checkpoint_dir="/tmp/per_class_ckpt")

    table = build_per_class_table(y_true, y_pred)
    print("\n=== Sinif-Bazli Tablo ===")
    print(table.to_string(index=False))

    cm_df, cm_pct_df = build_confusion_matrix_table(y_true, y_pred)
    print("\n=== Karisiklik Matrisi (sayilar) ===")
    print(cm_df.to_string())
    print("\n=== Karisiklik Matrisi (satir-normalize %) ===")
    print(cm_pct_df.to_string())

    table.to_csv("./outputs/per_class_table.csv", index=False)
    cm_pct_df.to_csv("./outputs/confusion_matrix_pct.csv")
    print("\nKaydedildi.")
