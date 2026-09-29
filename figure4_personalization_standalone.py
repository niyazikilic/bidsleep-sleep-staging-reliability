"""
figure4_personalization_standalone.py

Figure 4 (bagimsiz, TIFF dahil): Kisisellestirme doz-yanit, iki panel
(A: degisken orneklem, B: sabit orneklem n=31), TUTARLI legend konumuyla.

Onceki versiyonda legend konumu matplotlib'in otomatik 'best' secimine
birakilmisti, bu da panel A ve B'de FARKLI konumlar secilmesine (Panel
B'de veri cizgilerinin ustune denk gelmesine) yol acmisti. Bu script,
HER IKI panelde de SABIT, ayni konumu (upper left, beyaz opak arka
planli) kullanir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 11


def build_figure4_personalization(results_dir, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)

    from personalization_dose_response_v2 import summarize_by_k

    pers_raw = pd.read_csv(f"{results_dir}/personalization_v2_raw.csv")

    summary_unfair = summarize_by_k(pers_raw, n_boot=1000, restrict_to_common_subjects=False)
    summary_fair = summarize_by_k(pers_raw, n_boot=1000, restrict_to_common_subjects=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, summary, title in zip(
        axes, [summary_unfair, summary_fair],
        ["A) Variable-sample analysis", "B) Sample-fixed analysis (n=31)"]
    ):
        ax.plot(summary["k"], summary["kappa_raw_mean"], marker="o", label="Raw",
                 color="darkorange", linewidth=1.8)
        ax.fill_between(summary["k"], summary["kappa_raw_ci_lower"], summary["kappa_raw_ci_upper"],
                          alpha=0.15, color="darkorange")
        ax.plot(summary["k"], summary["kappa_hmm_mean"], marker="s", label="HMM-smoothed",
                 color="darkviolet", linewidth=1.8)
        ax.fill_between(summary["k"], summary["kappa_hmm_ci_lower"], summary["kappa_hmm_ci_upper"],
                          alpha=0.15, color="darkviolet")
        ax.set_xlabel("Calibration nights (K)")
        ax.set_ylabel("Cohen's kappa")
        ax.set_title(title, fontsize=11)
        # SABIT legend konumu, HER IKI panelde AYNI, opak beyaz arka planli
        ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white",
                   framealpha=0.9, edgecolor="lightgray")

    plt.tight_layout()
    png_path = f"{out_dir}/Figure4_Personalization.png"
    tiff_path = f"{out_dir}/Figure4_Personalization.tiff"
    plt.savefig(png_path, dpi=300)
    plt.savefig(tiff_path, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs"
    out_dir = f"{results_dir}/final_submission_files"
    build_figure4_personalization(results_dir, out_dir)
