"""
KISISELLESTIRME DOZ-YANIT DENEYI (v2): RF-HAM vs RF+HMM PARALEL

Onceki dose_response_analysis.py'nin gelistirilmis versiyonu:
1. Net tanimli K (kalibrasyon gecesi sayisi): her katilimcinin SABIT bir
   test gecesi vardir (kronolojik olarak SON gece), K = o katilimcinin
   ILK K gecesi kalibrasyon icin kullanilir. Test gecesi K'dan bagimsizdir.
2. HER K icin, HEM RF-ham HEM RF+HMM tahminleri paralel hesaplanir -
   boylece "kisisellestirme + zamansal duzeltme birlikte ne kadar
   yardimci oluyor" sorusuna da cevap verebiliriz.
3. Checkpoint destegi: baglanti kopmalarina karsi korumali.
4. Gecis matrisi HER (K, fold) kombinasyonu icin, o kombinasyonun
   EGITIM verisinden (havuz + kalibrasyon geceleri) tahmin edilir.
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

from cumulative_training_experiment import get_feature_cols
from hmm_smoothing import estimate_transition_matrix, estimate_start_probs, viterbi_smooth

RNG = np.random.default_rng(2026)


def personalization_dose_response_v2(full_df, k_values=range(0, 5), n_splits=5,
                                       n_estimators=150, random_state=42,
                                       label_col="expert_label", checkpoint_dir=None):
    """
    K=0: sadece pooled model (kisisellestirme yok, referans).
    K=1..4: pooled + katilimcinin kendi ILK K gecesi (kalibrasyon).
    Test: HER ZAMAN katilimcinin SON gecesi (K'dan bagimsiz, sabit).

    NOT: Bu nedenle katilimci basina en az (K_max + 1) gece gereklidir;
    daha az geceli katilimcilar o K degeri icin otomatik atlanir.
    """
    feature_cols = get_feature_cols(full_df)
    df = full_df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    subjects = df["subject"].unique()
    n_states = int(df[label_col].max()) + 1

    gkf = GroupKFold(n_splits=n_splits)
    fold_assignments = list(gkf.split(subjects, groups=subjects))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    all_results = []

    for fold_idx, (train_subj_idx, test_subj_idx) in enumerate(fold_assignments):
        train_subjects = subjects[train_subj_idx]
        test_subjects = subjects[test_subj_idx]
        pool_df = df[df["subject"].isin(train_subjects)]  # havuz: bu fold'daki egitim katilimcilari

        for target_subj in test_subjects:
            subj_df = df[df["subject"] == target_subj].sort_values("night")
            subj_nights = sorted(subj_df["night"].unique())

            if len(subj_nights) < 2:
                continue  # kalibrasyon + test icin en az 2 gece gerekli

            test_night = subj_nights[-1]  # SABIT: her zaman son gece
            calib_nights_available = subj_nights[:-1]  # geri kalan gece(ler) kalibrasyon icin

            test_df = subj_df[subj_df["night"] == test_night]
            if len(test_df) == 0:
                continue

            for k in k_values:
                if k > len(calib_nights_available):
                    continue  # bu katilimci icin bu K degeri test edilemez, yetersiz gece

                checkpoint_path = f"{checkpoint_dir}/subj{target_subj}_fold{fold_idx}_k{k}.csv" if checkpoint_dir else None
                if checkpoint_path and os.path.exists(checkpoint_path):
                    all_results.append(pd.read_csv(checkpoint_path))
                    continue

                calib_nights = calib_nights_available[:k]  # ilk K gece
                calib_df = subj_df[subj_df["night"].isin(calib_nights)] if k > 0 else subj_df.iloc[0:0]

                train_df = pd.concat([pool_df, calib_df], ignore_index=True) if k > 0 else pool_df

                X_train = train_df[feature_cols].values
                y_train = train_df[label_col].values.astype(int)
                X_test = test_df[feature_cols].values
                y_test = test_df[label_col].values.astype(int)

                sample_weight = None
                if k > 0:
                    sample_weight = np.concatenate([
                        np.ones(len(pool_df)), np.full(len(calib_df), 3.0)
                    ])

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10,
                                               class_weight="balanced", random_state=random_state, n_jobs=-1)
                clf.fit(X_train_s, y_train, sample_weight=sample_weight)

                proba_test = clf.predict_proba(X_test_s)
                pred_raw = clf.predict(X_test_s)

                # HMM icin gecis matrisini AYNI egitim setinden ogren
                train_sequences = [g.sort_values("epoch")[label_col].values.astype(int)
                                    for _, g in train_df.groupby(["subject", "night"])]
                trans_matrix = estimate_transition_matrix(train_sequences, n_states=n_states)
                start_probs = estimate_start_probs(train_sequences, n_states=n_states)
                pred_hmm = viterbi_smooth(proba_test, trans_matrix, start_probs)

                row = pd.DataFrame([{
                    "fold": fold_idx, "subject": target_subj, "k": k,
                    "test_night": test_night, "n_calib_nights_used": k,
                    "kappa_raw": cohen_kappa_score(y_test, pred_raw),
                    "kappa_hmm": cohen_kappa_score(y_test, pred_hmm),
                    "f1_macro_raw": f1_score(y_test, pred_raw, average="macro", zero_division=0),
                    "f1_macro_hmm": f1_score(y_test, pred_hmm, average="macro", zero_division=0),
                }])
                all_results.append(row)

                if checkpoint_path:
                    row.to_csv(checkpoint_path, index=False)

        print(f"  Fold {fold_idx+1}/{n_splits} tamamlandi")

    return pd.concat(all_results, ignore_index=True)


def summarize_by_k(results_df, n_boot=1000, rng=RNG, restrict_to_common_subjects=False):
    """
    Her K icin, tum metrikler icin kumeli bootstrap ortalama + %95 GA.

    restrict_to_common_subjects=True ise, SADECE tum K degerlerinde ortak
    olan katilimcilar (yani en yuksek K'yi da destekleyen katilimcilar)
    kullanilir - bu, ornekleme kaymasi (selection bias) confound'unu
    ortadan kaldirir ve K'nin GERCEK etkisini izole eder.
    """
    if restrict_to_common_subjects:
        max_k = results_df["k"].max()
        common_subjects = set(results_df.loc[results_df["k"] == max_k, "subject"].unique())
        results_df = results_df[results_df["subject"].isin(common_subjects)]
        print(f"(Sabit alt-kume analizi: sadece {len(common_subjects)} ortak katilimci kullaniliyor)")

    summaries = []
    for k, group in results_df.groupby("k"):
        subjects = group["subject"].unique()
        row = {"k": k, "n_subjects": len(subjects)}
        for metric in ["kappa_raw", "kappa_hmm", "f1_macro_raw", "f1_macro_hmm"]:
            row[f"{metric}_mean"] = group[metric].mean()
            boot_means = []
            for _ in range(n_boot):
                sampled = rng.choice(subjects, size=len(subjects), replace=True)
                vals = np.concatenate([group.loc[group["subject"]==s, metric].values for s in sampled])
                boot_means.append(vals.mean())
            row[f"{metric}_ci_lower"] = np.percentile(boot_means, 2.5)
            row[f"{metric}_ci_upper"] = np.percentile(boot_means, 97.5)
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values("k")


if __name__ == "__main__":
    full_df = pd.read_parquet("./data/full_feature_table.parquet")

    print("=== Kisisellestirme Doz-Yanit v2 (RF-ham vs RF+HMM paralel) ===")
    results_df = personalization_dose_response_v2(full_df, k_values=range(0, 3),
                                                     checkpoint_dir="/tmp/personalization_v2_ckpt")

    print(f"\nToplam gozlem: {len(results_df)}")
    summary = summarize_by_k(results_df, n_boot=200)
    print(summary.to_string(index=False))

    results_df.to_csv("./outputs/personalization_v2_raw.csv", index=False)
    summary.to_csv("./outputs/personalization_v2_summary.csv", index=False)
    print("\nKaydedildi.")
