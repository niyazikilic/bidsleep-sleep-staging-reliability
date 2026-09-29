"""
BIDSleep veri yükleme ve epoch hizalama modülü.

README'deki tam formülü uygular:
    k = floor((t - recStart) / 30) + 1

Kritik nokta: recStart, Eastern Time'da insan-okunur string olarak
saklanır ve Unix time'a çevrilmesi gerekir (README'nin 1. adımı).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pytz
from dateutil import parser as dateparser


EASTERN = pytz.timezone("US/Eastern")


def parse_rec_start_to_unix(rec_start_raw):
    """
    labels.mat'teki recStart alanini Unix time'a cevirir.

    recStart farkli formatlarda gelebilir:
    - insan-okunur string (Eastern Time), ör. "01-Jan-2025 23:00:00"
    - zaten Unix time (float/int) olarak kaydedilmis olabilir

    Bu fonksiyon her iki durumu da ele alir, cunku gercek veri
    setinde hangi formatta oldugundan %100 emin degiliz.
    """
    # scipy.io.loadmat bazen string'i numpy array icinde dondurur, temizleyelim
    if isinstance(rec_start_raw, np.ndarray):
        rec_start_raw = rec_start_raw.item() if rec_start_raw.size == 1 else rec_start_raw[0]

    # Zaten sayisal (Unix time) ise dogrudan don
    if isinstance(rec_start_raw, (int, float, np.integer, np.floating)):
        return float(rec_start_raw)

    # String ise, Eastern Time olarak parse edip Unix time'a cevir
    rec_start_str = str(rec_start_raw).strip()
    dt_naive = dateparser.parse(rec_start_str)
    if dt_naive.tzinfo is None:
        dt_eastern = EASTERN.localize(dt_naive)
    else:
        dt_eastern = dt_naive.astimezone(EASTERN)
    return dt_eastern.timestamp()


def load_night(night_dir):
    """
    Bir gece klasöründen motion, hr ve label verilerini yükler,
    epoch numaralarıyla hizalar.

    Returns
    -------
    dict:
        - hr_df: columns [t, hr, epoch]
        - motion_df: columns [Timestamp, x, y, z, epoch]
        - labels_df: columns [epoch, dreem_label, expert_label]
        - rec_start_unix: float
        - n_epochs: int
    """
    hr_path = os.path.join(night_dir, "hr.csv")
    motion_path = os.path.join(night_dir, "motion.csv")
    labels_path = os.path.join(night_dir, "labels.mat")

    for p in (hr_path, motion_path, labels_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Beklenen dosya bulunamadi: {p}")

    # --- labels.mat yukle ---
    mat = loadmat(labels_path)
    rec_start_unix = parse_rec_start_to_unix(mat["recStart"])
    dreem_label = np.asarray(mat["dreem_label"]).flatten().astype(int)
    expert_label = np.asarray(mat["expert_label"]).flatten().astype(int)

    # GERCEK VERIDE GOZLEMLENEN BIR DURUM: dreem_label ve expert_label
    # bazen farkli uzunlukta olabiliyor (uzman, kaydin bir kismini
    # skorlanamaz/artefaktli bulup cikarmis olabilir, ya da kucuk bir
    # sinir/yuvarlama farki olabilir). Bu durumda KISA olan uzunluga
    # gore ikisini de kirpiyoruz ve bunu acikca logluyoruz - makalenin
    # "veri kalitesi / eksik veri" bolumunde raporlanmasi gereken bir
    # husustur (TRIPOD-AI checklist'inde de bu madde var).
    label_length_mismatch = None
    if len(dreem_label) != len(expert_label):
        original_dreem_len, original_expert_len = len(dreem_label), len(expert_label)
        min_len = min(len(dreem_label), len(expert_label))
        label_length_mismatch = {
            "dreem_len": original_dreem_len,
            "expert_len": original_expert_len,
            "truncated_to": min_len,
        }
        dreem_label = dreem_label[:min_len]
        expert_label = expert_label[:min_len]

    n_epochs = len(dreem_label)

    # VERI KALITESI FILTRESI (Song ve ark., 2026, IEEE TBME ile tutarli):
    # Orijinal veri seti makalesi, 5 saatten kisa kayitlari kalite sorunlari
    # (cihaz kapanmasi, pil tukenmesi, uzun uyanikliklar, bashligin
    # cikarilmasi) nedeniyle analiz disi birakmis. Ayni kriteri, bu
    # veri setiyle calisan herkesin karsilasacagi bilinen bir veri
    # kalitesi sorunu oldugu icin biz de uyguluyoruz.
    MIN_HOURS = 5.0
    MIN_EPOCHS = int(MIN_HOURS * 3600 / 30)  # 5 saat = 600 epoch
    duration_hours = n_epochs * 30 / 3600
    below_min_duration = n_epochs < MIN_EPOCHS

    labels_df = pd.DataFrame({
        "epoch": np.arange(1, n_epochs + 1),
        "dreem_label": dreem_label,
        "expert_label": expert_label,
    })

    # --- hr.csv yukle (header yok!) ---
    hr_df = pd.read_csv(hr_path, header=None, names=["t", "hr"])

    # AYKIRI DEGER TEMIZLIGI (Song ve ark., 2026 ile tutarli): z-skoru
    # esigi 3 kullanilarak IHR aykiri degerleri temizlenir. Bu, sensor
    # hatalarindan/ani sicramalardan kaynaklanan bozulmalari onler.
    if len(hr_df) > 1:
        hr_z = (hr_df["hr"] - hr_df["hr"].mean()) / hr_df["hr"].std()
        hr_df = hr_df[hr_z.abs() <= 3].reset_index(drop=True)

    hr_df["epoch"] = np.floor((hr_df["t"] - rec_start_unix) / 30).astype(int) + 1

    # --- motion.csv yukle (header var) ---
    motion_df = pd.read_csv(motion_path)

    # Ayni sekilde ivmeolcer icin de aykiri deger temizligi (magnitude uzerinden)
    if len(motion_df) > 1:
        magnitude = np.sqrt(motion_df["x"]**2 + motion_df["y"]**2 + motion_df["z"]**2)
        mag_z = (magnitude - magnitude.mean()) / magnitude.std()
        motion_df = motion_df[mag_z.abs() <= 3].reset_index(drop=True)

    motion_df["epoch"] = np.floor((motion_df["Timestamp"] - rec_start_unix) / 30).astype(int) + 1

    # Gecerli epoch araligi disindaki (negatif veya n_epochs'u asan) satirlari filtrele
    hr_df = hr_df[(hr_df["epoch"] >= 1) & (hr_df["epoch"] <= n_epochs)].reset_index(drop=True)
    motion_df = motion_df[(motion_df["epoch"] >= 1) & (motion_df["epoch"] <= n_epochs)].reset_index(drop=True)

    return {
        "hr_df": hr_df,
        "motion_df": motion_df,
        "labels_df": labels_df,
        "rec_start_unix": rec_start_unix,
        "n_epochs": n_epochs,
        "label_length_mismatch": label_length_mismatch,
        "duration_hours": duration_hours,
        "below_min_duration": below_min_duration,
    }


def discover_nights(base_dir):
    """
    base_dir altindaki tum BidslabXX/N klasorlerini tarar.

    Returns
    -------
    list of dict: [{"subject": "Bidslab00", "night": 1, "path": "..."}, ...]
    """
    records = []
    subject_dirs = sorted(glob.glob(os.path.join(base_dir, "Bidslab*")))
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        night_dirs = sorted(
            glob.glob(os.path.join(subj_dir, "*")),
            key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0
        )
        for night_dir in night_dirs:
            night_name = os.path.basename(night_dir)
            if night_name.isdigit():
                records.append({
                    "subject": subject_id,
                    "night": int(night_name),
                    "path": night_dir,
                })
    return records


if __name__ == "__main__":
    base_dir = "./data/synthetic"
    nights = discover_nights(base_dir)
    print(f"Toplam {len(nights)} gece bulundu.")
    print(f"Örnek: {nights[0]}")

    # Ilk geceyi yukleyip test et
    result = load_night(nights[0]["path"])
    print(f"\nrecStart (Unix): {result['rec_start_unix']}")
    print(f"Toplam epoch sayisi: {result['n_epochs']}")
    print(f"\nHR verisi (ilk 5 satir):\n{result['hr_df'].head()}")
    print(f"\nMotion verisi (ilk 5 satir):\n{result['motion_df'].head()}")
    print(f"\nLabels (ilk 5 satir):\n{result['labels_df'].head()}")

    # Epoch hizalamasi dogru mu kontrol et: her epoch'ta en az bir HR ornegi olmali
    epochs_with_hr = result["hr_df"]["epoch"].nunique()
    print(f"\nHR verisi olan epoch sayisi: {epochs_with_hr} / {result['n_epochs']}")
