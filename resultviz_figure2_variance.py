"""
resultviz_figure2_variance.py

Figure 2 (bagimsiz): Kisiler-arasi vs kisi-ici varyans ayristirmasi.
Girdi: variance_decomposition_v15.csv (zaten V15_DIR'da mevcut).
Cikti: Fig2_Variance_Standalone.png / .tiff
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 11


def build_figure2_variance(results_dir, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    var_decomp_df = pd.read_csv(f"{results_dir}/variance_decomposition_v15.csv")

    pct_within = var_decomp_df["pct_variance_within_subject"].iloc[0]
    pct_between = 100 - pct_within
    icc_value = var_decomp_df["icc_from_mixed_model"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5.8))
    bars = ax.bar(["Between-subject", "Within-subject\n(night-to-night)"],
                    [pct_between, pct_within],
                    color=["dimgray", "lightgray"], edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("% of total performance variance")
    ax.set_ylim(0, 100)
    ax.set_title(f"Mixed-model ICC = {icc_value:.2f}", fontsize=10, style="italic", pad=10)
    for i, v in enumerate([pct_between, pct_within]):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=11)

    plt.tight_layout()
    png_path = f"{out_dir}/Fig2_Variance_Standalone.png"
    tiff_path = f"{out_dir}/Fig2_Variance_Standalone.tiff"
    plt.savefig(png_path, dpi=300)
    plt.savefig(tiff_path, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs"
    out_dir = f"{results_dir}/final_submission_files"
    build_figure2_variance(results_dir, out_dir)
