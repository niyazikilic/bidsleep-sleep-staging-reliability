"""
Epoch-bazli ozellik cikarimi: IHR (kalp hizi) ve ivmeolcerden.

Not: hr.csv verisi zaten "instantaneous heart rate" (BPM), ham PPG/EKG
sinyali degil. Bu yuzden klasik RR-interval tabanli HRV metrikleri
(RMSSD, SDNN gibi) burada IHR zaman serisi uzerinden yaklasik olarak
hesaplanir - bu, dusuk orneklemeli (0.2 Hz) IHR ile calisirken
literatürde kabul goren bir yaklasimdir, ama tam EKG-tabanli HRV ile
birebir ayni degildir. Bu sinirlama makalede acikca belirtilmelidir.
"""

import numpy as np
import pandas as pd
from scipy import stats


def extract_hr_features(hr_epoch):
    """Bir epoch'a ait HR degerlerinden ozellik cikarir."""
    if len(hr_epoch) == 0:
        return _nan_hr_features()

    hr_vals = hr_epoch["hr"].values
    feats = {
        "hr_mean": np.mean(hr_vals),
        "hr_std": np.std(hr_vals) if len(hr_vals) > 1 else 0.0,
        "hr_min": np.min(hr_vals),
        "hr_max": np.max(hr_vals),
        "hr_range": np.max(hr_vals) - np.min(hr_vals),
        "hr_median": np.median(hr_vals),
        "hr_iqr": np.percentile(hr_vals, 75) - np.percentile(hr_vals, 25) if len(hr_vals) > 1 else 0.0,
        "hr_n_samples": len(hr_vals),
    }
    # Ardisik farklar (successive differences) - RMSSD'nin IHR-tabanli yaklasik esdegeri
    if len(hr_vals) > 1:
        diffs = np.diff(hr_vals)
        feats["hr_rmssd_approx"] = np.sqrt(np.mean(diffs ** 2))
        feats["hr_mad_approx"] = np.mean(np.abs(diffs))
    else:
        feats["hr_rmssd_approx"] = np.nan
        feats["hr_mad_approx"] = np.nan
    return feats


def _nan_hr_features():
    keys = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_median",
            "hr_iqr", "hr_n_samples", "hr_rmssd_approx", "hr_mad_approx"]
    return {k: np.nan for k in keys}


def extract_motion_features(motion_epoch):
    """Bir epoch'a ait ivmeolcer verisinden ozellik cikarir."""
    if len(motion_epoch) == 0:
        return _nan_motion_features()

    x, y, z = motion_epoch["x"].values, motion_epoch["y"].values, motion_epoch["z"].values
    # Vektor buyuklugu (magnitude) - hareket yogunlugunun standart olcusu
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Yercekimini cikarilmis hareket enerjisi yaklasimi
    activity_energy = np.abs(magnitude - 1.0)

    feats = {
        "motion_mag_mean": np.mean(magnitude),
        "motion_mag_std": np.std(magnitude) if len(magnitude) > 1 else 0.0,
        "motion_activity_mean": np.mean(activity_energy),
        "motion_activity_std": np.std(activity_energy) if len(activity_energy) > 1 else 0.0,
        "motion_activity_max": np.max(activity_energy),
        "motion_n_samples": len(magnitude),
    }
    return feats


def _nan_motion_features():
    keys = ["motion_mag_mean", "motion_mag_std", "motion_activity_mean",
            "motion_activity_std", "motion_activity_max", "motion_n_samples"]
    return {k: np.nan for k in keys}


def build_epoch_feature_table(loaded_night, subject_id, night_id, window_size=11,
                                apply_per_night_zscore=False):
    """
    Bir gecelik yuklenmis veriden (data_loader.load_night ciktisi),
    her epoch icin bir satir olacak sekilde ozellik tablosu olusturur.

    Parameters
    ----------
    window_size : int
        Zamansal baglam penceresi (epoch sayisi, TEK sayi olmali).
        window_size=1 -> sadece mevcut epoch (baglamsiz, eski davranis)
        window_size=11 -> mevcut epoch + once/sonra 5'er epoch (330 saniye)
        NOT: Bu simetrik pencere GELECEK epoch'lari da kullanir - yani
        cevrimici/gercek-zamanli degil, cevrimdisi (offline) bir tasarimdir.
        Bu, makalede acikca belirtilmesi gereken bir tasarim karari.
    apply_per_night_zscore : bool
        True ise, ham ozellikler (window'dan once) HR ve motion icin
        gece-ici (within-night) z-skoruna donusturulur - kisiler-arasi
        bazal farklari (resting HR gibi) elemine etmek icin. Bu bir
        DUYARLILIK ANALIZI seceneigidir; varsayilan False (ham deger).

    Returns
    -------
    pd.DataFrame: her satir bir epoch, kolonlar ozellikler + etiketler + metadata
    """
    hr_df = loaded_night["hr_df"]
    motion_df = loaded_night["motion_df"]
    labels_df = loaded_night["labels_df"]
    n_epochs = loaded_night["n_epochs"]

    rows = []
    hr_grouped = dict(list(hr_df.groupby("epoch")))
    motion_grouped = dict(list(motion_df.groupby("epoch")))

    for epoch in range(1, n_epochs + 1):
        hr_epoch = hr_grouped.get(epoch, pd.DataFrame(columns=hr_df.columns))
        motion_epoch = motion_grouped.get(epoch, pd.DataFrame(columns=motion_df.columns))

        row = {"subject": subject_id, "night": night_id, "epoch": epoch}
        row.update(extract_hr_features(hr_epoch))
        row.update(extract_motion_features(motion_epoch))
        rows.append(row)

    feat_df = pd.DataFrame(rows)

    # --- OPSIYONEL: Gece-ici (within-night) z-score normalizasyonu ---
    # Bu, TUM geceden hesaplanan ortalama/std kullanir - yani cevrimdisi
    # bir islemdir (o epoch'un "gelecegini" de gorur). Offline analiz icin
    # kabul edilebilir ama makalede acikca belirtilmeli.
    if apply_per_night_zscore:
        numeric_feat_cols = [c for c in feat_df.columns
                              if c.startswith("hr_") or c.startswith("motion_")]
        for col in numeric_feat_cols:
            mu, sigma = feat_df[col].mean(), feat_df[col].std()
            if sigma > 0:
                feat_df[col] = (feat_df[col] - mu) / sigma
            else:
                feat_df[col] = 0.0

    # --- Zamansal baglam penceresi: onceki/sonraki epoch'larin ozellik
    # ortalamasini/std'sini ekstra kolonlar olarak ekle (RF icin) ---
    if window_size > 1:
        half_win = window_size // 2
        numeric_feat_cols = [c for c in feat_df.columns
                              if c.startswith("hr_") or c.startswith("motion_")]
        feat_df = feat_df.sort_values("epoch").reset_index(drop=True)
        for col in numeric_feat_cols:
            # Kayan pencere ortalamasi ve std'si (merkezi, gelecegi de kullanir)
            feat_df[f"{col}_win_mean"] = feat_df[col].rolling(
                window=window_size, center=True, min_periods=1).mean()
            feat_df[f"{col}_win_std"] = feat_df[col].rolling(
                window=window_size, center=True, min_periods=1).std().fillna(0)

    feat_df = feat_df.merge(labels_df, on="epoch", how="left")
    return feat_df


if __name__ == "__main__":
    from data_loader import discover_nights, load_night

    base_dir = "./data/synthetic"
    nights = discover_nights(base_dir)

    # Ilk gece uzerinde test
    night_info = nights[0]
    loaded = load_night(night_info["path"])
    feat_df = build_epoch_feature_table(loaded, night_info["subject"], night_info["night"])

    print(f"Ozellik tablosu boyutu: {feat_df.shape}")
    print(f"\nKolonlar:\n{list(feat_df.columns)}")
    print(f"\nIlk 5 satir:\n{feat_df.head()}")
    print(f"\nEksik deger sayisi:\n{feat_df.isna().sum()[feat_df.isna().sum() > 0]}")
