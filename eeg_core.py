# Fonctions de traitement EEG (enrichi avec stratégies de validation, équilibrage, intra-patient et métriques S-match)
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from matio import load_from_mat

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL = os.path.join(DATA_DIR, "Crises absences.xlsx")
CACHE = os.path.join(DATA_DIR, "cache_app")
RANDOM_STATE = 42

CANAUX_EEG = ['EEGFp1','EEGFp2','EEGF7','EEGF3','EEGFz','EEGF4','EEGF8','EEGT3','EEGC3',
              'EEGCz','EEGC4','EEGT4','EEGT5','EEGP3','EEGPz','EEGP4','EEGT6','EEGO1','EEGO2']
EXCLUS = {"200625A-F", "211222B-G", "210406A-F"}
WIN_S, OVERLAP = 2, 0.5
BANDES = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
FEATURES = ["Mean","Std","Variance","Min","Max","Amplitude","RMS","Energy","P_Delta","P_Theta","P_Alpha","P_Beta"]
LOG_FEATS = ["Variance", "Energy", "P_Delta", "P_Theta", "P_Alpha", "P_Beta"]

FEATURES_TEMPORELLES = ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy", "LineLength"]
FEATURES_FREQ_SPECTRALES = ["P_Delta", "P_Theta", "P_Alpha", "P_Beta", "R_Delta", "R_Theta", "R_Alpha", "R_Beta", "P_SW_2.5-4.5", "Freq_dominante"]
FEATURES_SPATIALES = ["Synchronie"]
FEATURES_AVANCEES = ["R_Delta","R_Theta","R_Alpha","R_Beta","P_SW_2.5-4.5","Freq_dominante","LineLength","Synchronie"]
TOUTES_FEATURES = FEATURES + FEATURES_AVANCEES


# ---------- Chargement ----------
def charger_eeg(patient):
    # Renvoie (signal [n_echantillons, 19 canaux], fs)
    f = os.path.join(DATA_DIR, patient.replace("-", "_") + "_0000d.mat")
    tt = load_from_mat(f)["output_d"]
    fs = np.ravel(tt.iloc[0, 0]).size
    X = np.stack([np.concatenate([np.ravel(v) for v in tt[c]]) for c in CANAUX_EEG], axis=1)
    return X.astype("float32"), fs


def filtrer(X, fs, low=0.5, high=40, ordre=4):
    sos = butter(ordre, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=0).astype("float32")


def lire_excel():
    xl = pd.read_excel(EXCEL)
    xl = xl.rename(columns={xl.columns[0]: "ID"})
    xl["ID"] = xl["ID"].astype(str).str.strip()
    xl = xl[xl["ID"].str.match(r"\d{6}[AB]-[A-Z]")]
    return xl.drop_duplicates("ID").reset_index(drop=True)


def crises_excel(row):
    # Liste des (début, fin) en secondes depuis minuit
    deb = [c for c in row.index if c.startswith("Deb")]
    fin = [c for c in row.index if c.startswith("Fin")]
    return [(row[d] * 86400, row[f] * 86400) for d, f in zip(deb, fin)
            if pd.notna(row[d]) and pd.notna(row[f])]


def annotations_h(patient):
    h = load_from_mat(os.path.join(DATA_DIR, patient.replace("-", "_") + "_0000h.mat"))["output_h"]
    return pd.DataFrame({"t": h.index.total_seconds(), "texte": h["Annotations"].astype(str).values})


# ---------- Recalage Excel -> temps du signal ----------
def score_pointes_ondes(X, fs):
    n = X.shape[0] // fs
    coupes = X[:n * fs].reshape(n, fs, X.shape[1])
    p3 = np.abs(np.fft.rfft(coupes, axis=1)[:, int(2.5 * fs / fs):int(4.5 * fs / fs), :]).mean((1, 2))
    return np.convolve(p3, np.ones(5) / 5, mode="same")


def recaler(crises_ex, score, anc_h, rayon=300):
    if not crises_ex: return 0
    c0 = crises_ex[0][0]
    candidats = [int(round(c0 - t)) for t in anc_h] if len(anc_h) else []
    if not candidats:
        t_max = int(np.argmax(score))
        candidats = [int(round(c0 - t_max))]
    meilleur_t0, meilleur_s = candidats[0], -1
    for t0_c in candidats:
        for dt in range(-rayon, rayon + 1, 5):
            t0 = t0_c + dt
            s = 0
            for d, f in crises_ex:
                i0, i1 = max(0, int(d - t0)), min(len(score), int(f - t0))
                if i1 > i0: s += score[i0:i1].mean()
            if s > meilleur_s:
                meilleur_s, meilleur_t0 = s, t0
    return meilleur_t0


# ---------- Découpage en fenêtres et calcul des features ----------
def vecteur_classe(n_echantillons, fs, crises):
    y = np.zeros(n_echantillons, dtype="uint8")
    for d, f in crises:
        i0, i1 = max(0, int(d * fs)), min(n_echantillons, int(f * fs))
        y[i0:i1] = 1
    return y


def segmenter(X, y, fs):
    win = int(WIN_S * fs)
    hop = int((WIN_S - OVERLAP) * fs)
    idx = np.arange(0, len(X) - win + 1, hop)
    fen = np.lib.stride_tricks.sliding_window_view(X, (win, X.shape[1]))[idx, 0]
    y_fen = (np.lib.stride_tricks.sliding_window_view(y, win)[idx].mean(1) >= 0.5).astype(int)
    t_deb = idx / fs
    return fen, y_fen, t_deb


def extraire_features(fen, fs):
    # fen: (N, win, 19)
    win = fen.shape[1]
    hann = np.hanning(win)[None, :, None]
    F = {"Mean": fen.mean(1), "Std": fen.std(1), "Variance": fen.var(1),
         "Min": fen.min(1), "Max": fen.max(1)}
    F["Amplitude"] = F["Max"] - F["Min"]
    F["RMS"] = np.sqrt((fen**2).mean(1))
    F["Energy"] = (fen.astype("float64")**2).sum(1)
    spec = np.abs(np.fft.rfft(fen * hann, axis=1))**2 / win
    f = np.fft.rfftfreq(win, 1/fs)
    for nom, (lo, hi) in BANDES.items():
        F["P_" + nom] = spec[:, (f >= lo) & (f < hi), :].sum(1)
    out = pd.DataFrame({k: v.mean(1) for k, v in F.items()})

    # Features avancées
    tot = spec[:, (f >= 0.5) & (f < 30), :].sum(1) + 1e-9
    G = {}
    for nom, (lo, hi) in BANDES.items():
        G["R_" + nom] = spec[:, (f >= lo) & (f < hi), :].sum(1) / tot
    G["P_SW_2.5-4.5"] = np.log10(spec[:, (f >= 2.5) & (f <= 4.5), :].sum(1) + 1e-6)
    sl = (f >= 0.5) & (f < 30)
    G["Freq_dominante"] = f[sl][spec[:, sl, :].argmax(1)]
    G["LineLength"] = np.abs(np.diff(fen, axis=1)).sum(1)
    for k, v in G.items():
        out[k] = v.mean(1)
    z = (fen - fen.mean(1, keepdims=True)) / (fen.std(1, keepdims=True) + 1e-9)
    C = np.einsum("nti,ntj->nij", z, z) / win
    k = fen.shape[2]
    out["Synchronie"] = (C.sum((1, 2)) - k) / (k * (k - 1))
    return out


def preparer_X(dataset, feature_set="all"):
    """
    feature_set:
      - 'all' : Toutes les 20 features
      - 'base' : 12 features de base
      - 'temporelles' : Descripteurs temporels (Mean, Std, Var, Min, Max, Amplitude, RMS, Energy, LineLength)
      - 'frequentielles' : Descripteurs spectraux (Delta, Theta, Alpha, Beta, Ratios, SW 3Hz, Freq_dom)
      - 'top_k' : Top features sélectionnées par pertinence
    """
    cols_base = [c for c in FEATURES if c in dataset.columns]
    X_base = dataset[cols_base].copy()
    log_cols = [c for c in LOG_FEATS if c in X_base.columns]
    X_base[log_cols] = np.log10(X_base[log_cols] + 1e-6)

    cols_av = [c for c in FEATURES_AVANCEES if c in dataset.columns]
    X_all = pd.concat([X_base, dataset[cols_av]], axis=1)

    if feature_set == "all":
        return X_all
    elif feature_set == "base":
        return X_base
    elif feature_set == "temporelles":
        dispo = [c for c in FEATURES_TEMPORELLES if c in X_all.columns]
        return X_all[dispo]
    elif feature_set == "frequentielles":
        dispo = [c for c in FEATURES_FREQ_SPECTRALES if c in X_all.columns]
        return X_all[dispo]
    elif feature_set == "top_k":
        top = ["P_SW_2.5-4.5", "R_Theta", "LineLength", "Variance", "Synchronie", "P_Theta", "RMS", "Energy"]
        dispo = [c for c in top if c in X_all.columns]
        return X_all[dispo]
    return X_all


# ---------- Traitement d'un patient complet (avec cache disque) ----------
def construire_base(progress=None):
    os.makedirs(CACHE, exist_ok=True)
    f_info, f_data = os.path.join(CACHE, "patients.pkl"), os.path.join(CACHE, "dataset.pkl")
    if os.path.exists(f_info) and os.path.exists(f_data):
        return pd.read_pickle(f_info), pd.read_pickle(f_data)

    csv_data = os.path.join(DATA_DIR, "dataset_fenetres_eeg.csv")
    if os.path.exists(csv_data) and os.path.exists(f_info):
        return pd.read_pickle(f_info), pd.read_csv(csv_data)

    xl = lire_excel()
    xl = xl[~xl.ID.isin(EXCLUS)].reset_index(drop=True)
    infos, lignes = [], []
    for i, r in xl.iterrows():
        if progress: progress(i / len(xl), f"Patient {r.ID} ({i+1}/{len(xl)})")
        Xb, fs = charger_eeg(r.ID)
        X = filtrer(Xb, fs)
        h = annotations_h(r.ID)
        anc = h.t[h.texte.str.lower().str.contains("absence|absnce|crise")].values
        t0 = recaler(crises_excel(r), score_pointes_ondes(X, fs), anc)
        crises = [(d - t0, f - t0) for d, f in crises_excel(r)]
        fen, yw, tw = segmenter(X, vecteur_classe(len(X), fs, crises), fs)
        feat = extraire_features(fen, fs)
        feat.insert(0, "Patient", r.ID); feat.insert(1, "t_debut", tw); feat["Class"] = yw
        lignes.append(feat)
        infos.append({"ID": r.ID, "Sexe": r.Sexe, "Age": r.Age, "fs": fs,
                      "duree_s": len(X) / fs, "t0": t0, "crises": crises})
    infos, dataset = pd.DataFrame(infos), pd.concat(lignes, ignore_index=True)
    infos.to_pickle(f_info); dataset.to_pickle(f_data)
    if progress: progress(1.0, "Terminé")
    return infos, dataset


# ---------- Modèles & Gestion du Déséquilibre ----------
def creer_modele(nom, equilibrage="balanced"):
    """
    equilibrage:
      - 'balanced' : pondération automatique des classes (class_weight='balanced')
      - 'none' : aucune correction (poids naturels)
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cw = "balanced" if equilibrage == "balanced" else None
    cw_rf = "balanced_subsample" if equilibrage == "balanced" else None

    modeles = {
        "Random Forest amélioré": RandomForestClassifier(
            n_estimators=200, class_weight=cw_rf, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=cw_rf, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight=cw, random_state=RANDOM_STATE
        ),
        "KNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance" if equilibrage == "balanced" else "uniform")
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale", class_weight=cw, probability=True, random_state=RANDOM_STATE)
        ),
    }
    return modeles[nom]


NOMS_MODELES = ["Random Forest amélioré", "Random Forest", "Decision Tree", "KNN", "SVM"]


def reechantillonner(X_tr, y_tr, methode="none", ratio=0.25):
    """
    Méthodes de rééchantillonnage pour équilibrer Crise vs Hors crise :
      - 'none' : données originales
      - 'undersampling' : sous-échantillonnage aléatoire de la classe majoritaire
      - 'oversampling' : sur-échantillonnage aléatoire avec remplacement de la classe minoritaire
      - 'smote' : interpolation synthétique (SMOTE)
    """
    if methode == "none":
        return X_tr, y_tr

    X_np = np.asarray(X_tr)
    y_np = np.asarray(y_tr)
    idx_pos = np.where(y_np == 1)[0]
    idx_neg = np.where(y_np == 0)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos == 0 or n_neg == 0:
        return X_tr, y_tr

    rng = np.random.default_rng(RANDOM_STATE)

    if methode == "undersampling":
        # Conserver tous les positifs et sous-échantillonner les négatifs
        taille_neg = min(n_neg, int(n_pos / ratio))
        idx_neg_sub = rng.choice(idx_neg, size=taille_neg, replace=False)
        idx_all = np.concatenate([idx_pos, idx_neg_sub])
        rng.shuffle(idx_all)
        if isinstance(X_tr, pd.DataFrame):
            return X_tr.iloc[idx_all].reset_index(drop=True), y_np[idx_all]
        return X_np[idx_all], y_np[idx_all]

    elif methode == "oversampling":
        # Dupliquer les exemples de crise pour atteindre le ratio
        taille_pos_cible = min(n_neg, int(n_neg * ratio))
        idx_pos_over = rng.choice(idx_pos, size=max(n_pos, taille_pos_cible), replace=True)
        idx_all = np.concatenate([idx_neg, idx_pos_over])
        rng.shuffle(idx_all)
        if isinstance(X_tr, pd.DataFrame):
            return X_tr.iloc[idx_all].reset_index(drop=True), y_np[idx_all]
        return X_np[idx_all], y_np[idx_all]

    elif methode == "smote":
        # Tenter d'utiliser imbalanced-learn s'il est présent, sinon fallback sur duplication avec bruit gaussien
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=ratio)
            return sm.fit_resample(X_tr, y_tr)
        except Exception:
            # Synthèse par interpolation k-plus proches voisins simple
            taille_pos_cible = min(n_neg, int(n_neg * ratio))
            n_a_generer = max(0, taille_pos_cible - n_pos)
            if n_a_generer > 0:
                X_pos = X_np[idx_pos]
                # Choisir deux points positifs aléatoires et interpoler
                idx1 = rng.choice(len(X_pos), size=n_a_generer, replace=True)
                idx2 = rng.choice(len(X_pos), size=n_a_generer, replace=True)
                lambdas = rng.uniform(0.1, 0.9, size=(n_a_generer, 1))
                X_synth = X_pos[idx1] + lambdas * (X_pos[idx2] - X_pos[idx1])
                # Ajouter un léger bruit pour éviter les doublons stricts
                X_synth += rng.normal(0, 1e-4, size=X_synth.shape)
                X_res = np.vstack([X_np, X_synth])
                y_res = np.concatenate([y_np, np.ones(n_a_generer, dtype=int)])
                idx_mix = np.arange(len(y_res))
                rng.shuffle(idx_mix)
                if isinstance(X_tr, pd.DataFrame):
                    return pd.DataFrame(X_res[idx_mix], columns=X_tr.columns), y_res[idx_mix]
                return X_res[idx_mix], y_res[idx_mix]
            return X_tr, y_tr

    return X_tr, y_tr


# ---------- Reconstruction temporelle & Métriques S-match ----------
def reconstruire_crises(t_debut, pred, duree_min=3, trou_max=1):
    """
    Fusionne les fenêtres consécutives prédites positives en événements temporels (début, fin).
    """
    inter = []
    for t, p in zip(t_debut, pred):
        if p != 1: continue
        d, f = t, t + WIN_S
        if inter and d <= inter[-1][1] + trou_max:
            inter[-1][1] = max(inter[-1][1], f)
        else:
            inter.append([d, f])
    return [(d, f) for d, f in inter if f - d >= duree_min]


def evaluer_smatch(reelles, detectees, duree_totale_heures=None, min_iou=0.0):
    """
    Évaluation S-match (Event-based & Sample-to-Match evaluation) :
      - Matching bi-directionnel entre intervalles réels et intervalles détectés
      - Sensibilité par événement (Event Recall) = Crises réelles trouvées / Total crises réelles
      - Précision par événement (Event Precision) = Détections valides / Total détections
      - Faux positifs par heure (FP / heure)
      - F1-score événementiel
    """
    n_reelles = len(reelles)
    n_detectees = len(detectees)

    if n_reelles == 0:
        return {
            "Total_Reelles": 0,
            "Total_Detectees": n_detectees,
            "Reelles_Trouvees": 0,
            "Faux_Positifs": n_detectees,
            "Event_Recall": 0.0,
            "Event_Precision": 0.0 if n_detectees > 0 else 1.0,
            "Event_F1": 0.0,
            "FP_par_heure": n_detectees / (duree_totale_heures if duree_totale_heures and duree_totale_heures > 0 else 1.0),
            "Detail": pd.DataFrame()
        }

    lignes = []
    utilisees = set()

    for k, (d_r, f_r) in enumerate(reelles, 1):
        # Match si chevauchement temporel
        matches = [j for j, (d_p, f_p) in enumerate(detectees) if (min(f_r, f_p) - max(d_r, d_p)) > min_iou]
        if matches:
            utilisees.update(matches)
            # Prendre le meilleur match (plus grand recouvrement)
            best_j = max(matches, key=lambda j: (min(f_r, detectees[j][1]) - max(d_r, detectees[j][0])))
            d_p, f_p = detectees[best_j]
            overlap = max(0, min(f_r, f_p) - max(d_r, d_p))
            duree_r = f_r - d_r
            overlap_pct = (overlap / duree_r) * 100
            lignes.append({
                "Crise": k,
                "Début réel (s)": round(d_r, 1),
                "Fin réelle (s)": round(f_r, 1),
                "Début détecté (s)": round(d_p, 1),
                "Fin détectée (s)": round(f_p, 1),
                "Couverture (%)": round(overlap_pct, 1),
                "Statut": "✅ Détectée"
            })
        else:
            lignes.append({
                "Crise": k,
                "Début réel (s)": round(d_r, 1),
                "Fin réelle (s)": round(f_r, 1),
                "Début détecté (s)": None,
                "Fin détectée (s)": None,
                "Couverture (%)": 0.0,
                "Statut": "❌ Ratée"
            })

    trouvees = sum(1 for lig in lignes if lig["Statut"] == "✅ Détectée")
    fausses = [detectees[j] for j in range(n_detectees) if j not in utilisees]
    n_fp = len(fausses)

    rec = trouvees / n_reelles if n_reelles > 0 else 0.0
    prec = trouvees / n_detectees if n_detectees > 0 else 1.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    fp_h = (n_fp / duree_totale_heures) if (duree_totale_heures and duree_totale_heures > 0) else float(n_fp)

    return {
        "Total_Reelles": n_reelles,
        "Total_Detectees": n_detectees,
        "Reelles_Trouvees": trouvees,
        "Faux_Positifs": n_fp,
        "Fausses_Alarmes_Intervalles": fausses,
        "Event_Recall": rec,
        "Event_Precision": prec,
        "Event_F1": f1,
        "FP_par_heure": round(fp_h, 2),
        "Detail": pd.DataFrame(lignes)
    }


def comparer(reelles, detectees):
    """Compatibilité descendante pour l'interface existante"""
    sm = evaluer_smatch(reelles, detectees)
    detail = sm["Detail"].rename(columns={"Statut": "Résultat"})
    return detail, sm.get("Fausses_Alarmes_Intervalles", [])
