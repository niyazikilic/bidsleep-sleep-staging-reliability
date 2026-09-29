"""
Tum gece klasorlerini tarar, her biri icin epoch-bazli ozellik tablosu
cikarir ve tek bir birlesik DataFrame olarak kaydeder.

Gercek veri geldiginde tek yapmaniz gereken BASE_DIR degiskenini
degistirmek - geri kalan her sey ayni sekilde calisir.
"""

import os
import pandas as pd
from data_loader import discover_nights, load_night
from feature_extraction import build_epoch_feature_table


def build_full_dataset(base_dir, output_path=None, verbose=True, apply_quality_filter=True,
                         window_size=11, apply_per_night_zscore=False, label_scheme="4class",
                         min_completeness=0.8):
    """
    min_completeness: bir gecenin dahil edilmesi icin gerekli minimum
    GECERLI (eksik olmayan HR+hareket verisi olan) epoch orani (0-1 arasi).
    None verilirse bu filtre uygulanmaz. Varsayilan 0.8 (%80).
    """
    nights = discover_nights(base_dir)
    if verbose:
        print(f"{len(nights)} gece bulundu, isleniyor... "
              f"(window={window_size}, zscore={apply_per_night_zscore}, sema={label_scheme})")

    all_frames = []
    errors = []
    excluded_short = []
    excluded_incomplete = []

    label_map_4class = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}  # Wake, Light(N1+N2), Deep(N3), REM

    for i, night_info in enumerate(nights):
        try:
            loaded = load_night(night_info["path"])

            if apply_quality_filter and loaded["below_min_duration"]:
                excluded_short.append({
                    "subject": night_info["subject"],
                    "night": night_info["night"],
                    "duration_hours": loaded["duration_hours"],
                    "n_epochs": loaded["n_epochs"],
                })
                continue

            feat_df = build_epoch_feature_table(
                loaded, night_info["subject"], night_info["night"],
                window_size=window_size, apply_per_night_zscore=apply_per_night_zscore
            )

            if label_scheme == "4class":
                feat_df["expert_label"] = feat_df["expert_label"].map(label_map_4class)
                feat_df["dreem_label"] = feat_df["dreem_label"].map(label_map_4class)
                feat_df = feat_df.dropna(subset=["expert_label"])  # 5 (Unknown) -> NaN -> dusur

            # VERI TAMLIGI (COMPLETENESS) FILTRESI: toplam sure yeterli olsa
            # bile, gecenin buyuk kismi eksik sensor verisi icerebilir (orn.
            # saatin gece ortasinda cikarilmasi). Bu, ICC/guvenilirlik
            # analizlerinde GERCEK fizyolojik degiskenlikle KARISABILECEK
            # yapay bir gurultu kaynagidir. Bu yuzden gecenin en az
            # min_completeness oraninda GECERLI (eksiksiz) epoch icermesini
            # sart kosuyoruz.
            if min_completeness is not None:
                base_feature_cols = ["hr_mean", "motion_mag_mean"]  # eksikligin en temel gostergesi
                valid_epochs = feat_df.dropna(subset=base_feature_cols).shape[0]
                completeness_ratio = valid_epochs / len(feat_df) if len(feat_df) > 0 else 0
                if completeness_ratio < min_completeness:
                    excluded_incomplete.append({
                        "subject": night_info["subject"], "night": night_info["night"],
                        "completeness_ratio": completeness_ratio, "valid_epochs": valid_epochs,
                        "total_epochs": len(feat_df),
                    })
                    continue

            all_frames.append(feat_df)
        except Exception as e:
            errors.append({"subject": night_info["subject"], "night": night_info["night"], "error": str(e)})
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(nights)} gece islendi...")

    full_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if verbose:
        if excluded_short:
            print(f"\nVERI KALITESI FILTRESI (SURE): {len(excluded_short)} gece 5 saat esiginin "
                  f"altinda oldugu icin haric tutuldu:")
            for e in excluded_short:
                print(f"  {e['subject']}/gece{e['night']}: {e['duration_hours']:.2f} saat")
        if excluded_incomplete:
            print(f"\nVERI KALITESI FILTRESI (TAMLIK): {len(excluded_incomplete)} gece "
                  f"%{min_completeness*100:.0f} gecerli veri esiginin altinda oldugu icin haric tutuldu:")
            for e in excluded_incomplete:
                print(f"  {e['subject']}/gece{e['night']}: %{e['completeness_ratio']*100:.1f} "
                      f"({e['valid_epochs']}/{e['total_epochs']} epoch)")
        if errors:
            print(f"\nUYARI: {len(errors)} gece islenemedi:")
            for e in errors:
                print(f"  {e['subject']} / gece {e['night']}: {e['error']}")

    if output_path:
        full_df.to_parquet(output_path, index=False)
        if verbose:
            print(f"\nKaydedildi: {output_path}")

    return full_df, errors, excluded_short, excluded_incomplete


if __name__ == "__main__":
    BASE_DIR = "./data/synthetic"
    OUTPUT_PATH = "./data/full_feature_table.parquet"

    full_df, errors, excluded_short, excluded_incomplete = build_full_dataset(BASE_DIR, OUTPUT_PATH)

    print(f"\n=== OZET ===")
    print(f"Toplam epoch sayisi: {len(full_df)}")
    print(f"Katilimci sayisi: {full_df['subject'].nunique()}")
    print(f"Katilimci basina ortalama gece: {full_df.groupby('subject')['night'].nunique().mean():.1f}")
    print(f"\nUyku evresi dagilimi (expert_label):")
    print(full_df['expert_label'].value_counts().sort_index())
