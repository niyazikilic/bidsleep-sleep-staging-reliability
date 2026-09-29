"""
SINYAL ORNEGI GORSELLESTIRMESI

1. Veri seti tanitimi icin: TEK bir temsili gece - ham IHR sinyali,
   ham ivmeolcer sinyali, ve karsilik gelen hipnogram (expert_label).
   "Temsili" gece, genel ortalama Kappa'ya EN YAKIN performansi
   gosteren gece olarak secilir (literatur konvansiyonuna uygun).

2. Gece-gece degiskenlik icin: AYNI katilimcinin FARKLI (en cok
   degisen) geceleri yan yana - hipnogram farklarini gorsel olarak
   gostermek icin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10

STAGE_NAMES_4CLASS = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}


def plot_single_night_example(base_dir, subject, night, out_path, label_scheme="4class"):
    """Tek bir gece icin ham IHR + ivmeolcer + hipnogram grafigi."""
    from data_loader import load_night

    night_dir = f"{base_dir}/{subject}/{night}"
    loaded = load_night(night_dir)

    hr_df = loaded["hr_df"]
    motion_df = loaded["motion_df"]
    labels_df = loaded["labels_df"].copy()

    if label_scheme == "4class":
        label_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
        labels_df["plot_label"] = labels_df["expert_label"].map(label_map)
        stage_names = STAGE_NAMES_4CLASS
    else:
        labels_df["plot_label"] = labels_df["expert_label"]
        stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    hr_time_hours = (hr_df["t"] - loaded["rec_start_unix"]) / 3600
    motion_df_copy = motion_df.copy()
    motion_time_hours = (motion_df_copy["Timestamp"] - loaded["rec_start_unix"]) / 3600
    motion_magnitude = np.sqrt(motion_df_copy["x"]**2 + motion_df_copy["y"]**2 + motion_df_copy["z"]**2)
    epoch_time_hours = (labels_df["epoch"] - 1) * 30 / 3600

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1, 0.6]})

    axes[0].plot(hr_time_hours, hr_df["hr"], color="crimson", linewidth=0.6)
    axes[0].set_ylabel("Heart Rate (bpm)")
    axes[0].set_title(f"Example Night: {subject}, Night {night}")

    axes[1].plot(motion_time_hours, motion_magnitude, color="darkorange", linewidth=0.4)
    axes[1].set_ylabel("Accel. Magnitude (g)")

    stage_order = list(stage_names.keys())
    axes[2].step(epoch_time_hours, labels_df["plot_label"], where="post", color="darkblue", linewidth=1.2)
    axes[2].set_yticks(stage_order)
    axes[2].set_yticklabels([stage_names[s] for s in stage_order])
    axes[2].invert_yaxis()
    axes[2].set_ylabel("Sleep Stage")
    axes[2].set_xlabel("Time (hours since recording start)")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_night_to_night_comparison(base_dir, subject, nights, out_path, label_scheme="4class"):
    """Ayni katilimcinin birden fazla gecesi icin hipnogram karsilastirmasi."""
    from data_loader import load_night

    label_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    stage_names = STAGE_NAMES_4CLASS if label_scheme == "4class" else \
        {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    fig, axes = plt.subplots(len(nights), 1, figsize=(12, 2.2 * len(nights)), sharex=False)
    if len(nights) == 1:
        axes = [axes]

    for ax, night in zip(axes, nights):
        night_dir = f"{base_dir}/{subject}/{night}"
        loaded = load_night(night_dir)
        labels_df = loaded["labels_df"].copy()

        if label_scheme == "4class":
            labels_df["plot_label"] = labels_df["expert_label"].map(label_map)
        else:
            labels_df["plot_label"] = labels_df["expert_label"]

        epoch_time_hours = (labels_df["epoch"] - 1) * 30 / 3600
        stage_order = list(stage_names.keys())

        ax.step(epoch_time_hours, labels_df["plot_label"], where="post", color="darkblue", linewidth=1.2)
        ax.set_yticks(stage_order)
        ax.set_yticklabels([stage_names[s] for s in stage_order])
        ax.invert_yaxis()
        ax.set_ylabel(f"Night {night}")

    axes[-1].set_xlabel("Time (hours since recording start)")
    fig.suptitle(f"Night-to-Night Hypnogram Variability: {subject}", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def select_representative_night(per_night_df, metric="kappa"):
    """Genel ortalama performansa EN YAKIN geceyi secer (literatur konvansiyonu)."""
    overall_mean = per_night_df[metric].mean()
    per_night_df = per_night_df.copy()
    per_night_df["dist_to_mean"] = (per_night_df[metric] - overall_mean).abs()
    best_row = per_night_df.sort_values("dist_to_mean").iloc[0]
    return best_row["subject"], int(best_row["night"])


def select_variable_subject(per_night_df, metric="kappa", min_nights=3):
    """Gece-gece EN FAZLA degisen (std'si en yuksek) katilimciyi secer."""
    night_counts = per_night_df.groupby("subject")["night"].nunique()
    eligible = night_counts[night_counts >= min_nights].index
    subset = per_night_df[per_night_df["subject"].isin(eligible)]
    stds = subset.groupby("subject")[metric].std().sort_values(ascending=False)
    return stds.index[0]
