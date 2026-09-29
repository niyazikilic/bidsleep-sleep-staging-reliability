"""
Belirsizlik tahmini ve coklu karsilastirma duzeltmesi.

Bu modul 3 eksigi tamamlar:
1. Bootstrap guven araliklari (genel Kappa/F1 metrikleri icin)
2. Kisisellestirme iyilesmesinin bootstrap CI'i
3. Sinif-bazli (Wake/N1/N2/N3/REM) pooled vs personalized karsilastirmasi,
   coklu karsilastirma icin Benjamini-Hochberg (FDR) duzeltmesi ile

ONEMLI METODOLOJIK NOKTA: Veri hiyerarsik yapida (epoch'lar gece icinde,
geceler katilimci icinde ic ice). Bu yuzden NAIVE epoch-bazli bootstrap
(her epoch'u bagimsizmis gibi resample etmek) bagimlilik yapisini ihlal
eder ve guven araliklarini olduğundan dar (fazla iyimser) gosterir.
Burada KUMELI (cluster) bootstrap kullaniyoruz: her bootstrap orneginde
KATILIMCILAR (subject) yerine koyarak (with replacement) yeniden ornekleniyor,
o katilimcinin TUM epoklari birlikte tasiniyor. Bu, dogru varyans tahmini
icin sarttir.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(2026)
N_BOOT = 2000  # bootstrap tekrar sayisi

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ============================================================
# 1. KUMELI (CLUSTER) BOOTSTRAP - genel model performansi icin
# ============================================================

def cluster_bootstrap_ci(df, y_true_col, y_pred_col, metric_fn, cluster_col="subject",
                          n_boot=N_BOOT, ci=0.95, rng=RNG):
    """
    Katilimci-bazli kumeli bootstrap ile bir metrigin guven araligini hesaplar.

    Her bootstrap orneginde katilimcilar yerine-koyarak (with replacement)
    yeniden secilir; secilen her katilimcinin TUM satirlari (epoklari)
    birlikte alinir. Bu, epoklar arasi bagimliligi (ayni katilimcinin
    epoklari bagimsiz degildir) korur.
    """
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    boot_estimates = []

    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        # Secilen her katilimcinin verisini birlestir (tekrar eden katilimcilar birden fazla kez eklenir)
        boot_df = pd.concat([df[df[cluster_col] == c] for c in sampled_clusters], ignore_index=True)
        try:
            estimate = metric_fn(boot_df[y_true_col].values, boot_df[y_pred_col].values)
            boot_estimates.append(estimate)
        except Exception:
            continue  # bazen bootstrap orneginde bir sinif hic bulunmayabilir, atla

    boot_estimates = np.array(boot_estimates)
    alpha = 1 - ci
    lower = np.percentile(boot_estimates, 100 * alpha / 2)
    upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(df[y_true_col].values, df[y_pred_col].values)

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "boot_std": boot_estimates.std(),
        "n_boot_valid": len(boot_estimates),
    }


def kappa_metric(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def macro_f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def accuracy_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# ============================================================
# 2. KISISELLESTIRME IYILESMESI - bootstrap CI (kumeli, subject-bazli)
# ============================================================

def personalization_improvement_ci(raw_preds_df, metric_fn=kappa_metric,
                                     n_boot=N_BOOT, ci=0.95, rng=RNG):
    """
    Pooled ve personalized modeller arasindaki metrik farkinin (personalized - pooled)
    kumeli (subject-bazli) bootstrap guven araligini hesaplar.

    Her bootstrap orneginde: katilimcilar yerine-koyarak resample edilir,
    o katilimcinin TUM epoklari (butun geceleri) birlikte alinir, iki model
    icin de metrik hesaplanip fark alinir.
    """
    subjects = raw_preds_df["subject"].unique()
    n_subjects = len(subjects)
    boot_diffs = []

    for _ in range(n_boot):
        sampled_subjects = rng.choice(subjects, size=n_subjects, replace=True)
        boot_df = pd.concat([raw_preds_df[raw_preds_df["subject"] == s] for s in sampled_subjects],
                             ignore_index=True)
        try:
            pooled_metric = metric_fn(boot_df["y_true"].values, boot_df["pred_pooled"].values)
            personalized_metric = metric_fn(boot_df["y_true"].values, boot_df["pred_personalized"].values)
            boot_diffs.append(personalized_metric - pooled_metric)
        except Exception:
            continue

    boot_diffs = np.array(boot_diffs)
    alpha = 1 - ci
    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    point_pooled = metric_fn(raw_preds_df["y_true"].values, raw_preds_df["pred_pooled"].values)
    point_personalized = metric_fn(raw_preds_df["y_true"].values, raw_preds_df["pred_personalized"].values)
    point_diff = point_personalized - point_pooled

    # Bootstrap dagiliminin ne kadari 0'in ustunde (tek-tarafli "p-degeri" yaklasimi)
    prop_above_zero = np.mean(boot_diffs > 0)

    return {
        "point_diff": point_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "boot_std": boot_diffs.std(),
        "prop_boot_above_zero": prop_above_zero,
        "excludes_zero": (lower > 0) or (upper < 0),
    }


# ============================================================
# 3. SINIF-BAZLI KARSILASTIRMA + COKLU KARSILASTIRMA DUZELTMESI
# ============================================================

def per_class_comparison_with_fdr(raw_preds_df):
    """
    Her uyku evresi (Wake/N1/N2/N3/REM) icin ayri ayri pooled vs personalized
    F1 karsilastirmasi yapar (katilimci-bazli esli t-testi), sonra
    Benjamini-Hochberg (FDR) duzeltmesi uygular.

    NEDEN GEREKLI: 5 ayri hipotez testi yapiyoruz (her sinif icin bir tane).
    Duzeltme yapilmazsa, tip-1 hata orani (yanlislikla "anlamli" bulma riski)
    sise (0.05'ten cok daha yuksek gerceklesir). FDR-BH, coklu testler
    arasinda gucunu cok kaybetmeden hata oranini kontrol eder.
    """
    from scipy.stats import ttest_rel

    subjects = raw_preds_df["subject"].unique()
    results = []

    for stage_code, stage_name in STAGE_NAMES.items():
        pooled_f1_per_subject = []
        personalized_f1_per_subject = []

        for subj in subjects:
            subj_df = raw_preds_df[raw_preds_df["subject"] == subj]
            y_true = subj_df["y_true"].values

            # Bu sinif icin "one-vs-rest" F1 (o katilimcinin TUM epoklari uzerinden)
            y_true_binary = (y_true == stage_code).astype(int)
            pooled_pred_binary = (subj_df["pred_pooled"].values == stage_code).astype(int)
            personalized_pred_binary = (subj_df["pred_personalized"].values == stage_code).astype(int)

            pooled_f1 = f1_score(y_true_binary, pooled_pred_binary, zero_division=0)
            personalized_f1 = f1_score(y_true_binary, personalized_pred_binary, zero_division=0)

            pooled_f1_per_subject.append(pooled_f1)
            personalized_f1_per_subject.append(personalized_f1)

        pooled_arr = np.array(pooled_f1_per_subject)
        personalized_arr = np.array(personalized_f1_per_subject)

        # Esli t-testi (katilimci-bazli, n=5 kucuk ornek - sonuclari temkinli yorumlayin)
        # NOT: eger pooled ve personalized F1'ler TUM katilimcilarda birebir ayniysa
        # (varyans = 0), t-testi tanimsizdir (NaN doner) - bu istatistiksel olarak
        # "hicbir fark yok" anlamina gelir, hata degildir. Bu durumu p=1.0 olarak isaretliyoruz
        # ki FDR duzeltmesi bu satirdan etkilenmesin.
        diffs = personalized_arr - pooled_arr
        if np.allclose(diffs, 0):
            stat, pval = 0.0, 1.0
        else:
            stat, pval = ttest_rel(personalized_arr, pooled_arr)

        results.append({
            "sleep_stage": stage_name,
            "pooled_f1_mean": pooled_arr.mean(),
            "personalized_f1_mean": personalized_arr.mean(),
            "improvement": personalized_arr.mean() - pooled_arr.mean(),
            "t_statistic": stat,
            "p_value_raw": pval,
        })

    results_df = pd.DataFrame(results)

    # Benjamini-Hochberg (FDR) duzeltmesi - 5 test icin
    reject, pvals_corrected, _, _ = multipletests(
        results_df["p_value_raw"].values, alpha=0.05, method="fdr_bh"
    )
    results_df["p_value_fdr_corrected"] = pvals_corrected
    results_df["significant_after_fdr"] = reject

    return results_df


if __name__ == "__main__":
    raw_preds_df = pd.read_csv("./outputs/personalization_raw_predictions.csv")

    print(f"Toplam epoch sayisi (raw predictions): {len(raw_preds_df)}")
    print(f"Katilimci sayisi: {raw_preds_df['subject'].nunique()}\n")

    # --- 1. Genel model performansi - kumeli bootstrap CI ---
    print("=" * 60)
    print("1. GENEL MODEL PERFORMANSI - KUMELI BOOTSTRAP CI (%95)")
    print("=" * 60)

    for model_col, model_name in [("pred_pooled", "Pooled"), ("pred_personalized", "Personalized")]:
        print(f"\n--- {model_name} Model ---")
        for metric_fn, metric_name in [(kappa_metric, "Kappa"), (macro_f1_metric, "Macro-F1"),
                                         (accuracy_metric, "Accuracy")]:
            result = cluster_bootstrap_ci(raw_preds_df, "y_true", model_col, metric_fn, n_boot=N_BOOT)
            print(f"  {metric_name}: {result['point_estimate']:.4f} "
                  f"[%95 GA: {result['ci_lower']:.4f} - {result['ci_upper']:.4f}] "
                  f"(bootstrap std: {result['boot_std']:.4f}, gecerli tekrar: {result['n_boot_valid']})")

    # --- 2. Kisisellestirme iyilesmesi - bootstrap CI ---
    print("\n" + "=" * 60)
    print("2. KISISELLESTIRME IYILESMESI - KUMELI BOOTSTRAP CI")
    print("=" * 60)

    for metric_fn, metric_name in [(kappa_metric, "Kappa"), (macro_f1_metric, "Macro-F1")]:
        result = personalization_improvement_ci(raw_preds_df, metric_fn=metric_fn, n_boot=N_BOOT)
        sig_marker = "✓ ANLAMLI (CI sifiri icermiyor)" if result["excludes_zero"] else "✗ anlamli degil (CI sifiri iceriyor)"
        print(f"\n  {metric_name} farki (personalized - pooled): {result['point_diff']:.4f}")
        print(f"  %95 GA: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  -->  {sig_marker}")
        print(f"  Bootstrap orneklerinin %{100*result['prop_boot_above_zero']:.1f}'i 0'in ustunde")

    # --- 3. Sinif-bazli karsilastirma + FDR duzeltmesi ---
    print("\n" + "=" * 60)
    print("3. SINIF-BAZLI KARSILASTIRMA (FDR-BH DUZELTMELI)")
    print("=" * 60)

    per_class_df = per_class_comparison_with_fdr(raw_preds_df)
    print(f"\n{per_class_df.to_string(index=False)}")

    print(f"\nNOT: n={raw_preds_df['subject'].nunique()} katilimci ile sinif-bazli testler "
          f"cok dusuk istatistiksel guce (power) sahiptir - gercek veride (n=47) "
          f"cok daha guvenilir sonuclar beklenir. Bu sentetik veride bulgular "
          f"sadece PIPELINE DOGRULAMASI amaclidir.")

    # Sonuclari kaydet
    per_class_df.to_csv("./outputs/per_class_comparison_fdr.csv", index=False)
    print("\nSonuclar kaydedildi: outputs/per_class_comparison_fdr.csv")
