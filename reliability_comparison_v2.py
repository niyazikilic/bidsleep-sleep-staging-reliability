"""
NAIF (ORTUSEN REFERANS) vs DUZELTILMIS (LEAVE-N-OUT) GUVENILIRLIK KARSILASTIRMASI

Bu modul, ayni per-night veri uzerinde IKI FARKLI tasarimi hesaplar:

1. NAIF TASARIM (yanlis/sisirilmis): Referans (R_i) katilimcinin TUM
   gecelerinin ortalamasi; tahmin (E_i^(N)) bu TUM gecelerin ICINDEN
   N tanesi. Tahmin ve referans ORTUSUR - bu, N arttikca yapay olarak
   r'yi 1'e yaklastirir (N=n_i oldugunda E=R, r=1 matematiksel zorunluluk).

2. DUZELTILMIS TASARIM (dogru): Referans, o tekrara ozel olarak SADECE
   tahminde KULLANILMAYAN (kalan) gecelerden hesaplanir. Tahmin ve
   referans ASLA ORTUSMEZ - bagimsiz, dogru bir guvenilirlik olcusu.

Ikisi de raporlanir: naif sonuc "yaniltici sisirilmis versiyon" olarak
gosterilip DUZELTMENIN GEREKCESI acikca sunulur.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)


def naive_reliability_curve(per_night_df, metric="kappa", n_boot=2000, rng=RNG):
    """NAIF (yanlis) tasarim: referans = TUM geceler, tahmin = bunlarin ICINDEN N tanesi."""
    subject_night_counts = per_night_df.groupby("subject")["night"].nunique()
    max_n = subject_night_counts.max()

    results = []
    for n_nights in range(1, max_n + 1):
        eligible = subject_night_counts[subject_night_counts >= n_nights].index
        if len(eligible) < 3:
            continue

        # Referans TEK SEFERDE, TUM geceler uzerinden hesaplanir (sabit)
        reference_values = {
            subj: per_night_df.loc[per_night_df["subject"] == subj, metric].mean()
            for subj in eligible
        }

        boot_correlations = []
        for _ in range(n_boot):
            estimated_values = []
            ref_list = []
            for subj in eligible:
                subj_nights = per_night_df.loc[per_night_df["subject"] == subj, ["night", metric]]
                all_nights = subj_nights["night"].values
                sampled = rng.choice(all_nights, size=n_nights, replace=False)
                est_value = subj_nights.loc[subj_nights["night"].isin(sampled), metric].mean()
                estimated_values.append(est_value)
                ref_list.append(reference_values[subj])

            if len(set(np.round(estimated_values, 6))) > 1:
                r = np.corrcoef(estimated_values, ref_list)[0, 1]
                if not np.isnan(r):
                    boot_correlations.append(r)

        if boot_correlations:
            results.append({
                "n_nights": n_nights,
                "n_eligible_subjects": len(eligible),
                "naive_correlation_mean": np.mean(boot_correlations),
                "naive_ci_lower": np.percentile(boot_correlations, 2.5),
                "naive_ci_upper": np.percentile(boot_correlations, 97.5),
            })
    return pd.DataFrame(results)


def corrected_reliability_curve(per_night_df, metric="kappa", n_boot=2000, rng=RNG):
    """DUZELTILMIS (dogru) tasarim: leave-N-out, referans HER TEKRARDA kalan gecelerden."""
    subject_night_counts = per_night_df.groupby("subject")["night"].nunique()
    max_n = subject_night_counts.max() - 1  # en az 1 gece referans icin kalmali

    results = []
    for n_nights in range(1, max_n + 1):
        eligible = subject_night_counts[subject_night_counts >= n_nights + 1].index
        if len(eligible) < 3:
            continue

        boot_correlations = []
        for _ in range(n_boot):
            estimated_values, reference_values = [], []
            for subj in eligible:
                subj_nights = per_night_df.loc[per_night_df["subject"] == subj, ["night", metric]]
                all_nights = subj_nights["night"].values
                sampled_est = rng.choice(all_nights, size=n_nights, replace=False)
                est_mask = subj_nights["night"].isin(sampled_est)
                ref_mask = ~est_mask

                estimated_values.append(subj_nights.loc[est_mask, metric].mean())
                reference_values.append(subj_nights.loc[ref_mask, metric].mean())

            if len(set(np.round(estimated_values, 6))) > 1 and len(set(np.round(reference_values, 6))) > 1:
                r = np.corrcoef(estimated_values, reference_values)[0, 1]
                if not np.isnan(r):
                    boot_correlations.append(r)

        if boot_correlations:
            results.append({
                "n_nights": n_nights,
                "n_eligible_subjects": len(eligible),
                "corrected_correlation_mean": np.mean(boot_correlations),
                "corrected_ci_lower": np.percentile(boot_correlations, 2.5),
                "corrected_ci_upper": np.percentile(boot_correlations, 97.5),
            })
    return pd.DataFrame(results)


def build_comparison_table(naive_df, corrected_df):
    """Iki tasarimi TEK bir tabloda birlestirir (N'e gore hizalanmis)."""
    merged = pd.merge(naive_df, corrected_df, on="n_nights", how="outer",
                        suffixes=("_naive", "_corrected"))
    merged = merged.sort_values("n_nights").reset_index(drop=True)
    return merged


def build_comparison_figure(naive_df, corrected_df, out_dir, filename_prefix="Figure_Reliability_Comparison_v2",
                              small_sample_threshold=15):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(naive_df["n_nights"], naive_df["naive_correlation_mean"],
             color="gray", linestyle="--", linewidth=1.5,
             label="Naive design (overlapping reference) — inflated", zorder=2)
    ax.fill_between(naive_df["n_nights"], naive_df["naive_ci_lower"], naive_df["naive_ci_upper"],
                      alpha=0.10, color="gray")

    ax.plot(corrected_df["n_nights"], corrected_df["corrected_correlation_mean"],
             color="black", linewidth=2,
             label="Corrected design (leave-N-out) — independent", zorder=3)
    ax.fill_between(corrected_df["n_nights"], corrected_df["corrected_ci_lower"], corrected_df["corrected_ci_upper"],
                      alpha=0.15, color="black")

    # Yeterli ornekli noktalar: dolu isaretciler (kare=naif, daire=duzeltilmis)
    naive_adequate = naive_df["n_eligible_subjects"] >= small_sample_threshold
    naive_small = ~naive_adequate
    corr_adequate = corrected_df["n_eligible_subjects"] >= small_sample_threshold
    corr_small = ~corr_adequate

    ax.scatter(naive_df.loc[naive_adequate, "n_nights"], naive_df.loc[naive_adequate, "naive_correlation_mean"],
                marker="s", s=55, color="gray", zorder=4)
    ax.scatter(corrected_df.loc[corr_adequate, "n_nights"], corrected_df.loc[corr_adequate, "corrected_correlation_mean"],
                marker="o", s=65, color="black", zorder=4)

    # Az ornekli noktalar: kirmizi kenarli, beyaz ic dolgulu (her iki egri icin de)
    small_label_added = False
    for df, xcol, ycol, marker in [(naive_df.loc[naive_small], "n_nights", "naive_correlation_mean", "s"),
                                      (corrected_df.loc[corr_small], "n_nights", "corrected_correlation_mean", "o")]:
        if len(df) > 0:
            lbl = f"Small sample (n<{small_sample_threshold}), unstable" if not small_label_added else None
            ax.scatter(df[xcol], df[ycol], marker=marker, s=90, facecolor="white",
                        edgecolor="red", linewidth=2, zorder=5, label=lbl)
            small_label_added = True

    ax.axhline(0.80, color="red", linestyle=":", linewidth=1, label="Conventional reliability threshold (r=0.80)")
    ax.set_xlabel("Number of nights (N)")
    ax.set_ylabel("Correlation with reference performance (r)")
    ax.set_ylim(-0.25, 1.1)
    max_n = max(naive_df["n_nights"].max(), corrected_df["n_nights"].max())
    ax.set_xlim(0.5, max_n + 0.5)
    ax.set_xticks(range(1, int(max_n) + 1))
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    png_path = f"{out_dir}/{filename_prefix}.png"
    tiff_path = f"{out_dir}/{filename_prefix}.tiff"
    plt.savefig(png_path, dpi=300)
    plt.savefig(tiff_path, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")


if __name__ == "__main__":
    per_night_df = pd.read_csv("./outputs/per_night_final_model.csv")

    print("=== Naif (yanlis) tasarim hesaplaniyor ===")
    naive_df = naive_reliability_curve(per_night_df, n_boot=300)
    print(naive_df.to_string(index=False))

    print("\n=== Duzeltilmis (dogru) tasarim hesaplaniyor ===")
    corrected_df = corrected_reliability_curve(per_night_df, n_boot=300)
    print(corrected_df.to_string(index=False))

    comparison = build_comparison_table(naive_df, corrected_df)
    print("\n=== KARSILASTIRMA TABLOSU ===")
    print(comparison.to_string(index=False))

    build_comparison_figure(naive_df, corrected_df, "/tmp")

    comparison.to_csv("./outputs/reliability_comparison.csv", index=False)
    print("\nKaydedildi.")
