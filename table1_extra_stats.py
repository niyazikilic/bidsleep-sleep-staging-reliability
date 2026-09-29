"""
table1_extra_stats.py

Table 1'e eklenecek ek istatistikler:
1. Katilimci basina gece sayisi: ortalama, SD, aralik (mevcut, ama SD eksikti)
2. Dort uyku sinifinin (Wake/Light/Deep/REM) epoch sayisi ve yuzdesi
"""

import pandas as pd
import numpy as np

STAGE_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}


def compute_table1_extra_stats(full_df, label_col="expert_label"):
    nights_per_subject = full_df.groupby("subject")["night"].nunique()

    night_stats = {
        "mean_nights_per_participant": round(nights_per_subject.mean(), 2),
        "sd_nights_per_participant": round(nights_per_subject.std(), 2),
        "min_nights_per_participant": int(nights_per_subject.min()),
        "max_nights_per_participant": int(nights_per_subject.max()),
    }

    class_counts = full_df[label_col].value_counts().sort_index()
    total_epochs = len(full_df)
    class_distribution = []
    for code, name in STAGE_NAMES.items():
        n = int(class_counts.get(code, 0))
        pct = round(100 * n / total_epochs, 1)
        class_distribution.append({"Sleep Stage": name, "N epochs": n, "% of total": pct})

    return night_stats, pd.DataFrame(class_distribution), total_epochs


if __name__ == "__main__":
    import sys, os
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs"

    full_df = pd.read_parquet(f"{results_dir}/full_feature_table_v15.parquet")
    night_stats, class_dist_df, total_epochs = compute_table1_extra_stats(full_df)

    print("=== Gece Istatistikleri ===")
    for k, v in night_stats.items():
        print(f"  {k}: {v}")

    print(f"\n=== Sinif Dagilimi (toplam {total_epochs} epoch) ===")
    print(class_dist_df.to_string(index=False))

    out_dir = f"{results_dir}/final_submission_files"
    os.makedirs(out_dir, exist_ok=True)
    class_dist_df.to_csv(f"{out_dir}/Table1_class_distribution.csv", index=False)
    print("\nKaydedildi.")
