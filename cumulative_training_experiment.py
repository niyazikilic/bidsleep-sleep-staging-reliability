"""
RQ2 (NET TANIMLI): KUMULATIF EGITIM VERISI BUYUKLUGU DENEYI

Onceki belirsizligi gidermek icin TASARIM TAM OLARAK SOYLE TANIMLANMISTIR:

1. 5-Fold Subject-Wise GroupKFold: katilimcilar 5 gruba ayrilir, her
   fold'da 1 grup test, kalan 4 grup (havuz) egitim icin kullanilir.
   Bu, sizin onerdiginiz tasarimla birebir ayni.

2. K (kumulatif egitim gecesi sayisi): EGITIM havuzundaki HER katilimci
   icin, o katilimcinin GECELERI KRONOLOJIK SIRAYLA siralanir, sadece
   ILK K gecesi egitime dahil edilir (K=1,2,3...). Bu, "K arttikca daha
   fazla veriyle egitim = daha iyi/kararli model mi?" sorusunu test eder.

3. TEST setinde HICBIR KISITLAMA yoktur - test katilimcilarinin TUM
   gecelerinde degerlendirme yapilir (boylece test seti K'ya gore
   degismez, karsilastirma adil olur).

Bu tasarim, "kisisellestirme" (personalization_analysis.py,
dose_response_analysis.py) ile KARISTIRILMAMALIDIR: o deneyler TEK BIR
katilimcinin kendi verisiyle kalibrasyonunu test eder; bu deney ise
EGITIM HAVUZUNUN BUYUKLUGUNUN genel model performansina etkisini test
eder - farkli bir soru.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(2026)


def get_feature_cols(df, window_size=11):
    """window'lu ozellik kolonlarini otomatik bulur (varsa)."""
    base_cols = [
        "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_median", "hr_iqr",
        "hr_rmssd_approx", "hr_mad_approx",
        "motion_mag_mean", "motion_mag_std", "motion_activity_mean",
        "motion_activity_std", "motion_activity_max",
    ]
    win_cols = [c for c in df.columns if c.endswith("_win_mean") or c.endswith("_win_std")]
    return base_cols + win_cols if win_cols else base_cols


def limit_training_nights(train_df, k):
    """Egitim setindeki HER katilimci icin sadece ilk K (kronolojik) geceyi tutar."""
    def _first_k_nights(g):
        nights_sorted = sorted(g["night"].unique())[:k]
        return g[g["night"].isin(nights_sorted)]
    return train_df.groupby("subject", group_keys=False).apply(_first_k_nights)


def cumulative_training_experiment(full_df, k_values=range(1, 8), n_splits=5,
                                     n_estimators=150, random_state=42, label_col="expert_label",
                                     checkpoint_dir=None):
    """
    Ana deney: K arttikca (egitim havuzunda katilimci basina gece sayisi),
    test performansi nasil degisiyor?

    checkpoint_dir verilirse, her (K, fold) TAMAMLANDIGINDA diske kaydedilir.
    Baglanti kopmasi durumunda, fonksiyon tekrar cagrildiginda zaten
    tamamlanmis (K, fold) kombinasyonlarini ATLAR.
    """
    import os

    feature_cols = get_feature_cols(full_df)
    df = full_df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    subjects = df["subject"].unique()

    gkf = GroupKFold(n_splits=n_splits)
    fold_assignments = list(gkf.split(subjects, groups=subjects))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_results = []

    for k in k_values:
        for fold_idx, (train_subj_idx, test_subj_idx) in enumerate(fold_assignments):
            checkpoint_path = f"{checkpoint_dir}/k{k}_fold{fold_idx}.csv" if checkpoint_dir else None

            if checkpoint_path and os.path.exists(checkpoint_path):
                all_results.append(pd.read_csv(checkpoint_path))
                continue

            train_subjects = subjects[train_subj_idx]
            test_subjects = subjects[test_subj_idx]

            train_df_full = df[df["subject"].isin(train_subjects)]
            test_df = df[df["subject"].isin(test_subjects)]

            train_df_limited = limit_training_nights(train_df_full, k)

            if len(train_df_limited) == 0 or len(test_df) == 0:
                continue

            X_train = train_df_limited[feature_cols].values
            y_train = train_df_limited[label_col].values.astype(int)
            X_test = test_df[feature_cols].values
            y_test = test_df[label_col].values.astype(int)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10,
                                           class_weight="balanced", random_state=random_state, n_jobs=-1)
            clf.fit(X_train_s, y_train)
            preds = clf.predict(X_test_s)

            test_df_copy = test_df.copy()
            test_df_copy["pred"] = preds

            fold_rows = []
            for subj, g in test_df_copy.groupby("subject"):
                fold_rows.append({
                    "k": k, "fold": fold_idx, "subject": subj,
                    "n_train_epochs": len(train_df_limited),
                    "kappa": cohen_kappa_score(g[label_col], g["pred"]),
                    "f1_macro": f1_score(g[label_col], g["pred"], average="macro", zero_division=0),
                })
            fold_result = pd.DataFrame(fold_rows)
            all_results.append(fold_result)

            if checkpoint_path:
                fold_result.to_csv(checkpoint_path, index=False)

        print(f"  K={k} tamamlandi ({n_splits} fold)" + (" (kaydedildi)" if checkpoint_dir else ""))

    return pd.concat(all_results, ignore_index=True)


def cluster_bootstrap_ci_by_k(results_df, metric="kappa", n_boot=1000, ci=0.95, rng=RNG):
    """Her K degeri icin, katilimci-bazli kumeli bootstrap ile ortalama ve %95 GA hesaplar."""
    summary = []
    for k, group in results_df.groupby("k"):
        subjects = group["subject"].unique()
        boot_means = []
        for _ in range(n_boot):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            vals = np.concatenate([group.loc[group["subject"] == s, metric].values for s in sampled])
            boot_means.append(vals.mean())
        summary.append({
            "k": k,
            f"{metric}_mean": group[metric].mean(),
            f"{metric}_ci_lower": np.percentile(boot_means, 2.5),
            f"{metric}_ci_upper": np.percentile(boot_means, 97.5),
        })
    return pd.DataFrame(summary).sort_values("k")


if __name__ == "__main__":
    full_df = pd.read_parquet("./data/full_feature_table.parquet")

    print("=== RQ2: KUMULATIF EGITIM VERISI BUYUKLUGU DENEYI (net tanimli) ===\n")
    results_df = cumulative_training_experiment(full_df, k_values=range(1, 4))  # sentetik veri az gece icerdigi icin kucuk aralik

    results_df.to_csv("./outputs/cumulative_training_raw.csv", index=False)

    summary_kappa = cluster_bootstrap_ci_by_k(results_df, metric="kappa", n_boot=200)
    print("\n=== Kappa (K'ya gore, %95 GA ile) ===")
    print(summary_kappa.to_string(index=False))

    summary_kappa.to_csv("./outputs/cumulative_training_summary.csv", index=False)
    print("\nSonuclar kaydedildi.")
