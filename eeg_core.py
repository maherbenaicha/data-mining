# Fonctions de traitement EEG (reprises du notebook Projet_EEG_Crises_Absence.ipynb)
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
    seg = X[: n*fs].reshape(n, fs, X.shape[1])
    spec = np.abs(np.fft.rfft(seg, axis=1))**2
    f = np.fft.rfftfreq(fs, 1/fs)
    s = np.log(spec[:, (f >= 2.5) & (f <= 4.5), :].sum(1) + 1e-6).mean(1)
    return (s - np.median(s)) / s.std()


def recaler(crises_abs, score, ancres=()):
    n = len(score)
    premier = min(d for d, f in crises_abs); dernier = max(f for d, f in crises_abs)
    meilleur = (-np.inf, None)
    for t0 in np.arange(np.floor(dernier - n), np.ceil(premier) + 1):
        idx = np.concatenate([np.arange(round(d - t0), round(f - t0) + 1) for d, f in crises_abs])
        idx = idx[(idx >= 0) & (idx < n)]
        s = score[idx].mean()
        if len(ancres):
            s += 2 * np.mean([any(d-t0-5 <= a <= f-t0+15 for d, f in crises_abs) for a in ancres])
        if s > meilleur[0]: meilleur = (s, t0)
    return meilleur[1]


# ---------- Segmentation et features ----------
def vecteur_classe(n, fs, crises):
    t = np.arange(n) / fs
    y = np.zeros(n, dtype=int)
    for d, f in crises:
        y[(t >= d) & (t <= f)] = 1
    return y


def segmenter(X, y, fs):
    win = int(WIN_S * fs); step = int(win * (1 - OVERLAP))
    starts = np.arange(0, len(X) - win + 1, step)
    fen = np.stack([X[s:s+win] for s in starts])
    frac = np.array([y[s:s+win].mean() for s in starts])
    return fen, (frac >= 0.5).astype(int), starts / fs


def extraire_features(fen, fs):
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

    # Features avancées (Partie XII du notebook)
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


FEATURES_AVANCEES = ["R_Delta","R_Theta","R_Alpha","R_Beta","P_SW_2.5-4.5","Freq_dominante","LineLength","Synchronie"]


def preparer_X(dataset, avancees=True):
    X = dataset[FEATURES].copy()
    X[LOG_FEATS] = np.log10(X[LOG_FEATS] + 1e-6)
    if avancees:
        X = pd.concat([X, dataset[FEATURES_AVANCEES]], axis=1)
    return X


# ---------- Traitement d'un patient complet (avec cache disque) ----------
def construire_base(progress=None):
    # Construit (ou relit depuis le cache) la table des patients et le dataset de fenêtres
    os.makedirs(CACHE, exist_ok=True)
    f_info, f_data = os.path.join(CACHE, "patients.pkl"), os.path.join(CACHE, "dataset.pkl")
    if os.path.exists(f_info) and os.path.exists(f_data):
        return pd.read_pickle(f_info), pd.read_pickle(f_data)

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


# ---------- Modèles ----------
def creer_modele(nom):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "Random Forest amélioré": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                         min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale")),
    }[nom]


NOMS_MODELES = ["Random Forest amélioré", "Random Forest", "Decision Tree", "KNN", "SVM"]


def reconstruire_crises(t_debut, pred, duree_min=3, trou_max=1):
    inter = []
    for t, p in zip(t_debut, pred):
        if p != 1: continue
        d, f = t, t + WIN_S
        if inter and d <= inter[-1][1] + trou_max:
            inter[-1][1] = max(inter[-1][1], f)
        else:
            inter.append([d, f])
    return [(d, f) for d, f in inter if f - d >= duree_min]


def comparer(reelles, detectees):
    lignes, utilisees = [], set()
    for k, (d, f) in enumerate(reelles, 1):
        match = [j for j, (a, b) in enumerate(detectees) if a < f and b > d]
        if match:
            utilisees.update(match); a, b = detectees[match[0]]
            lignes.append({"Crise": k, "Début réel (s)": d, "Fin réelle (s)": f,
                           "Début détecté (s)": a, "Fin détectée (s)": b, "Résultat": "✅ Détectée"})
        else:
            lignes.append({"Crise": k, "Début réel (s)": d, "Fin réelle (s)": f, "Résultat": "❌ Ratée"})
    fp = [detectees[j] for j in range(len(detectees)) if j not in utilisees]
    return pd.DataFrame(lignes).round(1), fp
