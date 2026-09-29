"""
table2_extra_column.py

Table 2'nin (kumulatif egitim verisi) eksik sutunu: her K degeri icin
egitim havuzundaki katilimcilarin GERCEKTE katkida bulundugu ortalama
gece sayisini hesaplar.

Bu, "K, izin verilen MAKSIMUM gece sayisidir, her katilimcinin kesin
sayisi degildir" notunu SAYISAL olarak destekler (Methods'ta soz
verilen aciklama).

Ayni 5-fold GroupKFold bolunmesini (subjects dizisi ayni sirada
oldugu surece deterministik) kullanarak, cumulative_training_experiment.py
ile TUTARLI sonuc uretir.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


def compute_avg_training_nights_per_k(full_df, k_values=range(1, 8), n_splits=5):
    """
    Her K icin, TUM fold'lardaki egitim katilimcilarinin ortalama
    KATKIDA BULUNDUGU (gercekte kullanilan) gece sayisini dondurur.
    """
    subjects = full_df["subject"].unique()
    gkf = GroupKFold(n_splits=n_splits)
    fold_splits = list(gkf.split(subjects, groups=subjects))

    results = []
    for k in k_values:
        nights_used_all = []
        for train_idx, _ in fold_splits:
            train_subjects = subjects[train_idx]
            for subj in train_subjects:
                subj_nights = sorted(full_df.loc[full_df["subject"] == subj, "night"].unique())
                n_used = min(len(subj_nights), k)
                nights_used_all.append(n_used)

        results.append({
            "K (max nights permitted)": k,
            "Mean nights actually used": round(np.mean(nights_used_all), 2),
            "Min nights used": int(np.min(nights_used_all)),
            "Max nights used": int(np.max(nights_used_all)),
            "N training-participant instances": len(nights_used_all),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys, os
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs"

    full_df = pd.read_parquet(f"{results_dir}/full_feature_table_v15.parquet")
    table = compute_avg_training_nights_per_k(full_df)
    print(table.to_string(index=False))

    out_dir = f"{results_dir}/final_submission_files"
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(f"{out_dir}/Table2_avg_training_nights.csv", index=False)
    print("\nKaydedildi.")
