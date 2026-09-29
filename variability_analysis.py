"""
Gece-gece degiskenlik analizi: ICC + karisik etkili model.

Bu, "Night-to-Night Variability" stratejisinin istatistiksel omurgasidir:
1. ICC (sinif-ici korelasyon katsayisi): performansin ne kadari kisi-ozel
   (sabit/tutarli), ne kadari gece-gece rastgele degisiyor?
2. Karisik etkili model (mixed-effects model): kisi-ici ve kisiler-arasi
   varyansi ayri ayri kestirir, ayrica gece sirasi (1., 2., ... gece) gibi
   sabit etkilerin anlamli olup olmadigini test eder.
"""

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")


def compute_icc(per_night_df, metric="kappa"):
    """
    ICC hesaplar: performans metriginin ne kadari katilimcinin
    "sabit" bir ozelligi (yuksek ICC = kisi-ici tutarlilik yuksek,
    dusuk ICC = gece-gece degiskenlik yuksek).

    ONEMLI METODOLOJIK NOT:
    pingouin.intraclass_corr klasik (Shrout & Fleiss) ICC formulasyonunu
    kullanir ve DENGELI tasarim gerektirir (her katilimcida ayni sayida
    olcum/gece). BIDSleep veri setinde katilimci basina gece sayisi
    degisken (3-7), yani tasarim yapisal olarak dengesizdir - bu
    eksik veri degil, veri setinin dogal ozelligidir.

    Bu durumda iki secenek var:
    (a) Sadece ortak minimum gece sayisina sahip alt kumeyi kullanip
        klasik ICC hesaplamak (veri kaybi pahasina),
    (b) Karisik etkili model tabanli ICC kullanmak (variance_decomposition
        fonksiyonu) - bu, dengesiz tasarimlari doganl olarak destekler
        ve TUM veriyi kullanir.

    Makalenizde (b) secenegini ana yontem olarak, (a)'yi ise duyarlilik
    analizi (sensitivity analysis) olarak sunmanizi oneririz - bu, hakemlere
    dengesizlik sorununun farkinda oldugunuzu ve dogru sekilde ele
    aldiginizi gosterir.
    """
    # (a) Duyarlilik analizi: dengeli alt-kume ile klasik ICC
    min_nights = per_night_df.groupby("subject")["night"].count().min()
    balanced_df = (
        per_night_df.sort_values(["subject", "night"])
        .groupby("subject")
        .head(min_nights)
        .copy()
    )
    # Her katilimci icin gece sirasini 1..min_nights olarak yeniden numaralandir
    balanced_df["night_rank"] = balanced_df.groupby("subject").cumcount() + 1

    df_icc = balanced_df[["subject", "night_rank", metric]].copy()
    df_icc.columns = ["targets", "raters", "ratings"]

    icc_result = pg.intraclass_corr(
        data=df_icc, targets="targets", raters="raters", ratings="ratings"
    )
    return icc_result, min_nights


def fit_mixed_model(per_night_df, metric="kappa"):
    """
    Karisik etkili model: metric ~ gece_sirasi + (1 | subject)

    Rastgele kesim (random intercept) katilimci bazinda kisiler-arasi
    varyansi modeller; kalan varyans (residual) kisi-ici / gece-gece
    degiskenligi temsil eder.
    """
    df = per_night_df.copy()
    # Gece sirasini normalize et (ilk gece = 0), "ilk gece etkisi"ni test etmek icin faydali
    df["night_order"] = df.groupby("subject")["night"].rank(method="first") - 1

    model = smf.mixedlm(f"{metric} ~ night_order", df, groups=df["subject"])
    result = model.fit()
    return result


def variance_decomposition(mixed_model_result):
    """
    Karisik etkili modelden varyans bilesenlerini cikarir:
    - between-subject variance (kisiler-arasi)
    - within-subject (residual) variance (gece-gece / kisi-ici)
    - ICC = between / (between + within)
    """
    var_between = mixed_model_result.cov_re.iloc[0, 0]
    var_within = mixed_model_result.scale
    icc = var_between / (var_between + var_within)
    return {
        "var_between_subject": var_between,
        "var_within_subject": var_within,
        "icc_from_mixed_model": icc,
        "pct_variance_within_subject": 100 * var_within / (var_between + var_within),
    }


if __name__ == "__main__":
    per_night_df = pd.read_csv("./outputs/per_night_final_model.csv")

    print("=== ICC Analizi (Kappa metrigi uzerinden) ===\n")
    icc_result, min_nights = compute_icc(per_night_df, metric="kappa")
    print(f"(Duyarlilik analizi: dengeli alt-kume kullanildi, katilimci basina {min_nights} gece)\n")
    print(icc_result)

    print("\n\n=== Karisik Etkili Model (mixedlm) ===\n")
    mixed_result = fit_mixed_model(per_night_df, metric="kappa")
    print(mixed_result.summary())

    print("\n\n=== Varyans Ayristirmasi ===\n")
    var_decomp = variance_decomposition(mixed_result)
    for k, v in var_decomp.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n\nYORUM: Varyansin %{var_decomp['pct_variance_within_subject']:.1f}'i "
          f"kisi-ici (gece-gece) kaynakli, geri kalani kisiler-arasi farkliliktan geliyor.")
    print("Bu sayi, makalenizin 'gece-gece degiskenlik onemli mi?' sorusuna dogrudan "
          "istatistiksel cevap oluyor.")

    # Sonuclari kaydet
    var_decomp_df = pd.DataFrame([var_decomp])
    var_decomp_df.to_csv("./outputs/variance_decomposition.csv", index=False)
    icc_result.to_csv("./outputs/icc_results.csv", index=False)
    print("\nSonuclar kaydedildi: outputs/variance_decomposition.csv, outputs/icc_results.csv")
