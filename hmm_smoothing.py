"""
HMM/VITERBI ZAMANSAL DUZELTME

RF, her epoch'u BAGIMSIZ tahmin eder - uyku evrelerinin otokorelasyonlu
yapisini (bir Deep blogu ~20 dakika surer, ani Wake<->Deep gecisleri
fizyolojik olarak imkansizdir) goz ardi eder.

Bu modul, RF'in olasilik ciktilarini (predict_proba) bir Gizli Markov
Modeli (HMM) ile son-isleyerek, EGITIM VERISINDEN OGRENILEN GECIS
OLASILIKLARINA gore en olasi TUM diziyi (Viterbi algoritmasi) bulur.

ONEMLI: Bu bir "duzeltme garantisi" degildir - sadece bir DENEYDIR.
Sonuc, onceden belirlenmis kritere gore degerlendirilir:
  - Kappa'da >=0.02 anlamli iyilesme (bootstrap CI ile) -> ise yariyor
  - Iyilesme yok/anlamsiz -> ise yaramiyor, dürüstçe raporlanir
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(2026)


def estimate_transition_matrix(y_sequences, n_states=4, smoothing=1.0):
    """
    Egitim verisindeki GERCEK etiket dizilerinden (katilimci/gece bazinda
    ayri ayri, gece sinirlarini karistirmadan) durum gecis matrisini tahmin eder.

    smoothing: Laplace duzeltmesi (sifir-olasilikli gecisleri onlemek icin)
    """
    trans_counts = np.full((n_states, n_states), smoothing)
    for seq in y_sequences:
        for i in range(len(seq) - 1):
            trans_counts[seq[i], seq[i + 1]] += 1
    trans_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return trans_matrix


def estimate_start_probs(y_sequences, n_states=4, smoothing=1.0):
    """Her dizinin ILK epoch'undaki durum dagilimindan baslangic olasiligini tahmin eder."""
    start_counts = np.full(n_states, smoothing)
    for seq in y_sequences:
        if len(seq) > 0:
            start_counts[seq[0]] += 1
    return start_counts / start_counts.sum()


def viterbi_smooth(proba_matrix, transition_matrix, start_probs):
    """
    RF'in predict_proba ciktisini (emission probabilities olarak kullanarak)
    Viterbi algoritmasi ile en olasi durum dizisine cozer.

    proba_matrix: (n_epochs, n_states) - RF'in her epoch/sinif icin olasiligi
    """
    n_epochs, n_states = proba_matrix.shape
    log_trans = np.log(transition_matrix + 1e-10)
    log_start = np.log(start_probs + 1e-10)
    log_emission = np.log(proba_matrix + 1e-10)

    viterbi_matrix = np.zeros((n_epochs, n_states))
    backpointer = np.zeros((n_epochs, n_states), dtype=int)

    viterbi_matrix[0] = log_start + log_emission[0]

    for t in range(1, n_epochs):
        for s in range(n_states):
            scores = viterbi_matrix[t - 1] + log_trans[:, s]
            backpointer[t, s] = np.argmax(scores)
            viterbi_matrix[t, s] = np.max(scores) + log_emission[t, s]

    best_path = np.zeros(n_epochs, dtype=int)
    best_path[-1] = np.argmax(viterbi_matrix[-1])
    for t in range(n_epochs - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path


def run_rf_with_hmm_comparison(full_df, feature_cols, label_col="expert_label",
                                 n_splits=5, n_estimators=150, random_state=42,
                                 checkpoint_dir=None):
    """
    5-Fold CV: her fold'da RF egitilir, hem HAM tahminler hem HMM-duzeltilmis
    tahminler kaydedilir - adil, ayni test verisi uzerinde karsilastirma.

    checkpoint_dir verilirse, her fold TAMAMLANDIGINDA diske kaydedilir.
    Fonksiyon tekrar cagrildiginda, zaten tamamlanmis fold'lari ATLAR ve
    kaldigi yerden devam eder - Colab baglanti kopmalarina karsi koruma.
    """
    import os

    df = full_df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    subjects = df["subject"].unique()
    n_states = int(df[label_col].max()) + 1

    gkf = GroupKFold(n_splits=n_splits)
    fold_splits = list(gkf.split(subjects, groups=subjects))

    all_results = []

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        checkpoint_path = f"{checkpoint_dir}/fold_{fold_idx}.csv" if checkpoint_dir else None

        # Bu fold zaten tamamlanmis mi kontrol et
        if checkpoint_path and os.path.exists(checkpoint_path):
            fold_result = pd.read_csv(checkpoint_path)
            all_results.append(fold_result)
            print(f"  Fold {fold_idx+1}/{n_splits} zaten tamamlanmis, checkpoint'ten yuklendi (atlaniyor)")
            continue

        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]

        train_df = df[df["subject"].isin(train_subjects)]
        test_df = df[df["subject"].isin(test_subjects)]

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values.astype(int)
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10,
                                       class_weight="balanced", random_state=random_state, n_jobs=-1)
        clf.fit(X_train_s, y_train)

        train_sequences = [g.sort_values("epoch")[label_col].values.astype(int)
                            for _, g in train_df.groupby(["subject", "night"])]
        trans_matrix = estimate_transition_matrix(train_sequences, n_states=n_states)
        start_probs = estimate_start_probs(train_sequences, n_states=n_states)

        test_df_copy = test_df.copy()
        proba_all = clf.predict_proba(X_test_s)
        raw_preds = clf.predict(X_test_s)
        test_df_copy["pred_raw"] = raw_preds

        hmm_preds = np.zeros(len(test_df_copy), dtype=int)
        for (subj, night), g in test_df_copy.groupby(["subject", "night"]):
            night_idx = g.sort_values("epoch").index
            night_proba = proba_all[[test_df_copy.index.get_loc(i) for i in night_idx]]
            smoothed = viterbi_smooth(night_proba, trans_matrix, start_probs)
            hmm_preds[[test_df_copy.index.get_loc(i) for i in night_idx]] = smoothed

        test_df_copy["pred_hmm"] = hmm_preds

        fold_rows = []
        for subj, g in test_df_copy.groupby("subject"):
            fold_rows.append({
                "fold": fold_idx, "subject": subj,
                "kappa_raw": cohen_kappa_score(g[label_col], g["pred_raw"]),
                "kappa_hmm": cohen_kappa_score(g[label_col], g["pred_hmm"]),
                "f1_macro_raw": f1_score(g[label_col], g["pred_raw"], average="macro", zero_division=0),
                "f1_macro_hmm": f1_score(g[label_col], g["pred_hmm"], average="macro", zero_division=0),
            })
        fold_result = pd.DataFrame(fold_rows)
        all_results.append(fold_result)

        if checkpoint_path:
            fold_result.to_csv(checkpoint_path, index=False)

        print(f"  Fold {fold_idx+1}/{n_splits} tamamlandi" + (" (kaydedildi)" if checkpoint_path else ""))

    return pd.concat(all_results, ignore_index=True)


def bootstrap_improvement_ci(results_df, metric_pair=("kappa_hmm", "kappa_raw"),
                               n_boot=1000, ci=0.95, rng=RNG):
    """Kumeli (subject-bazli) bootstrap ile HMM-RF farkinin guven araligini hesaplar."""
    subjects = results_df["subject"].unique()
    boot_diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        vals_hmm = np.concatenate([results_df.loc[results_df["subject"]==s, metric_pair[0]].values for s in sampled])
        vals_raw = np.concatenate([results_df.loc[results_df["subject"]==s, metric_pair[1]].values for s in sampled])
        boot_diffs.append(vals_hmm.mean() - vals_raw.mean())

    boot_diffs = np.array(boot_diffs)
    point_diff = results_df[metric_pair[0]].mean() - results_df[metric_pair[1]].mean()
    return {
        "point_diff": point_diff,
        "ci_lower": np.percentile(boot_diffs, 2.5),
        "ci_upper": np.percentile(boot_diffs, 97.5),
        "meets_threshold_0.02": point_diff >= 0.02 and np.percentile(boot_diffs, 2.5) > 0,
    }


if __name__ == "__main__":
    from cumulative_training_experiment import get_feature_cols

    full_df = pd.read_parquet("./data/full_feature_table.parquet")
    feature_cols = get_feature_cols(full_df)

    print("=== RF (HAM) vs RF+HMM/Viterbi Karsilastirmasi ===\n")
    results_df = run_rf_with_hmm_comparison(full_df, feature_cols)

    print("\n=== Ozet ===")
    print(results_df[["kappa_raw", "kappa_hmm", "f1_macro_raw", "f1_macro_hmm"]].mean())

    ci_result = bootstrap_improvement_ci(results_df)
    print(f"\nKappa iyilesmesi (HMM - RAW): {ci_result['point_diff']:.4f}")
    print(f"%95 GA: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
    print(f"Onceden belirlenen esigi (>=0.02, anlamli) karsiliyor mu: {ci_result['meets_threshold_0.02']}")

    results_df.to_csv("./outputs/hmm_comparison_raw.csv", index=False)
    print("\nSonuclar kaydedildi.")
