# Interface avancée et interactive pour le projet "Détection des crises d'absence"
# Supporte : Stratégies de validation, Gestion du déséquilibre, Comparaison des Features,
# Modèles Intra-Patient personnalisés et Évaluation S-match.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     LeaveOneGroupOut, GroupKFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)

import eeg_core as core

st.set_page_config(page_title="Crises d'absence - EEG Analytics", page_icon="🧠", layout="wide")


# ---------- Données & Cache ----------
@st.cache_data(show_spinner=False)
def base():
    barre = st.progress(0.0, "Préparation des données (mise en cache)...")
    infos, dataset = core.construire_base(lambda p, txt: barre.progress(p, txt))
    barre.empty()
    return infos, dataset


@st.cache_data(show_spinner="Chargement du signal EEG...", max_entries=4)
def signal(patient):
    X, fs = core.charger_eeg(patient)
    return core.filtrer(X, fs), fs


@st.cache_data(show_spinner=False)
def detecter(patient, nom_modele, equilibrage="balanced", methode_sampling="none", feature_set="all", seuil=0.5):
    # Entraîne le modèle sur tous les AUTRES patients (Leave-One-Patient-Out)
    _, dataset = base()
    X = core.preparer_X(dataset, feature_set=feature_set)
    y = dataset["Class"].values
    masque = (dataset.Patient == patient).values

    X_train, y_train = X[~masque], y[~masque]
    X_train_res, y_train_res = core.reechantillonner(X_train, y_train, methode=methode_sampling)

    clf = core.creer_modele(nom_modele, equilibrage=equilibrage)
    clf.fit(X_train_res, y_train_res)

    d = dataset[masque].sort_values("t_debut")
    X_test = X.loc[d.index]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= seuil).astype(int)
    else:
        pred = clf.predict(X_test)
        proba = pred.astype(float)

    return d.t_debut.values, d.Class.values, pred, proba


@st.cache_data(show_spinner=False)
def entrainer_intra_patient(patient, nom_modele, train_ratio=0.6, equilibrage="balanced", methode_sampling="none", feature_set="all"):
    # Modèle dédié et personnalisé pour un seul patient
    _, dataset = base()
    df_p = dataset[dataset.Patient == patient].sort_values("t_debut").reset_index(drop=True)
    n_total = len(df_p)
    n_train = max(10, int(n_total * train_ratio))

    df_train = df_p.iloc[:n_train]
    df_test = df_p.iloc[n_train:]

    X_train = core.preparer_X(df_train, feature_set=feature_set)
    y_train = df_train["Class"].values
    X_test = core.preparer_X(df_test, feature_set=feature_set)
    y_test = df_test["Class"].values

    X_train_res, y_train_res = core.reechantillonner(X_train, y_train, methode=methode_sampling)

    clf = core.creer_modele(nom_modele, equilibrage=equilibrage)
    clf.fit(X_train_res, y_train_res)

    pred_test = clf.predict(X_test)

    # Métriques
    metrics = {
        "Accuracy": accuracy_score(y_test, pred_test),
        "Precision": precision_score(y_test, pred_test, zero_division=0),
        "Recall": recall_score(y_test, pred_test, zero_division=0),
        "F1-score": f1_score(y_test, pred_test, zero_division=0),
        "Split_Train_s": df_train.t_debut.max(),
        "N_Train": n_train,
        "N_Test": len(df_test),
        "Crises_Train": int(y_train.sum()),
        "Crises_Test": int(y_test.sum())
    }
    return df_test.t_debut.values, y_test, pred_test, metrics, confusion_matrix(y_test, pred_test)


@st.cache_data(show_spinner=False)
def comparer_strategies_validation(nom_modele="Random Forest amélioré", feature_set="all", equilibrage="balanced"):
    # Compare 3 stratégies : Holdout 80/20, Stratified 5-Fold, Group 5-Fold (par patients)
    _, dataset = base()
    X = core.preparer_X(dataset, feature_set=feature_set)
    y = dataset["Class"].values
    groups = dataset["Patient"].values

    resultats = []

    # 1. Holdout 80/20 standard
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=core.RANDOM_STATE)
    clf = core.creer_modele(nom_modele, equilibrage=equilibrage).fit(X_tr, y_tr)
    p = clf.predict(X_te)
    resultats.append({
        "Stratégie": "Holdout 80/20 (Aléatoire)",
        "Type": "Fenêtres mélangées",
        "Accuracy": accuracy_score(y_te, p),
        "Precision": precision_score(y_te, p, zero_division=0),
        "Recall": recall_score(y_te, p, zero_division=0),
        "F1-score": f1_score(y_te, p, zero_division=0)
    })

    # 2. Stratified 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=core.RANDOM_STATE)
    recs, precs, f1s, accs = [], [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        clf = core.creer_modele(nom_modele, equilibrage=equilibrage).fit(X.iloc[tr_idx], y[tr_idx])
        p = clf.predict(X.iloc[te_idx])
        accs.append(accuracy_score(y[te_idx], p))
        precs.append(precision_score(y[te_idx], p, zero_division=0))
        recs.append(recall_score(y[te_idx], p, zero_division=0))
        f1s.append(f1_score(y[te_idx], p, zero_division=0))
    resultats.append({
        "Stratégie": "5-Fold Stratifié",
        "Type": "Fenêtres mélangées",
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "F1-score": np.mean(f1s)
    })

    # 3. Group 5-Fold (Inter-Patient strict)
    gkf = GroupKFold(n_splits=5)
    recs, precs, f1s, accs = [], [], [], []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        clf = core.creer_modele(nom_modele, equilibrage=equilibrage).fit(X.iloc[tr_idx], y[tr_idx])
        p = clf.predict(X.iloc[te_idx])
        accs.append(accuracy_score(y[te_idx], p))
        precs.append(precision_score(y[te_idx], p, zero_division=0))
        recs.append(recall_score(y[te_idx], p, zero_division=0))
        f1s.append(f1_score(y[te_idx], p, zero_division=0))
    resultats.append({
        "Stratégie": "Group 5-Fold (Inter-Patients)",
        "Type": "Généralisation stricte",
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "F1-score": np.mean(f1s)
    })

    return pd.DataFrame(resultats).set_index("Stratégie")


@st.cache_data(show_spinner=False)
def evaluer_importance_features(nom_modele="Random Forest"):
    _, dataset = base()
    X = core.preparer_X(dataset, feature_set="all")
    y = dataset["Class"].values
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=core.RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp


# ---------- Affichage Graphique ----------
def courbe_eeg(X, fs, crises, canal, detections=None, t0=None, t1=None):
    t = np.arange(len(X)) / fs
    i0, i1 = int((t0 or 0) * fs), int((t1 or t[-1]) * fs)
    pas = max(1, (i1 - i0) // 40000)
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(t[i0:i1:pas], X[i0:i1:pas, core.CANAUX_EEG.index(canal)], lw=.4, color="k")
    haut = (.5, 1) if detections is not None else (0, 1)
    for k, (d, f) in enumerate(crises):
        ax.axvspan(d, f, ymin=haut[0], ymax=haut[1], color="tab:green" if detections is not None else "red",
                   alpha=.35, label="Crise réelle (Excel)" if k == 0 else None)
    for k, (d, f) in enumerate(detections or []):
        ax.axvspan(d, f, ymin=0, ymax=.5, color="tab:red", alpha=.35, label="Crise détectée par l'IA" if k == 0 else None)
    ax.set(xlabel="Temps (secondes)", ylabel="µV", xlim=(t[i0], t[i1 - 1]))
    ax.grid(alpha=.3); ax.legend(loc="upper right")
    return fig


def zoom_multicanal(X, fs, crises, t0, t1, ecart=250):
    i0, i1 = max(0, int(t0 * fs)), min(len(X), int(t1 * fs))
    tt = np.arange(i0, i1) / fs
    fig, ax = plt.subplots(figsize=(14, 7))
    for k in range(X.shape[1]):
        ax.plot(tt, X[i0:i1, k] - k * ecart, lw=.6, color="k")
    for d, f in crises:
        if f > t0 and d < t1:
            ax.axvspan(max(d, t0), min(f, t1), color="red", alpha=.15)
    ax.set_yticks([-k * ecart for k in range(X.shape[1])], [c.replace("EEG", "") for c in core.CANAUX_EEG])
    ax.set(xlabel="Temps (secondes)", xlim=(tt[0], tt[-1]))
    return fig


# ---------- Menu Principal ----------
st.sidebar.title("🧠 EEG Crises d'Absence")
page = st.sidebar.radio("Navigation du projet", [
    "🏠 0. Vue d'ensemble & Données",
    "👀 1. Exploration du signal EEG",
    "🧪 2. Stratégies de Validation",
    "⚖️ 3. Résolution du Déséquilibre",
    "📊 4. Comparaison des Features",
    "🎯 5. Modèle Intra-Patient (Dédié)",
    "🔎 6. Détection Clinique & S-match"
])
st.sidebar.markdown("---")
st.sidebar.caption("Système avancé d'évaluation EEG & Machine Learning.")

infos, dataset = base()
patients = infos.ID.tolist()


# ========================================================
# 0. ACCUEIL
# ========================================================
if page == "🏠 0. Vue d'ensemble & Données":
    st.title("Système Intelligent de Détection des Crises d'Absence 🧠")
    st.markdown("""
Cette plateforme intègre désormais l'ensemble des modules d'évaluation avancée en data mining :
- **Stratégies de Validation :** Comparaison rigoureuse Holdout vs K-Fold vs GroupKFold (inter-patients).
- **Équilibrage de Classes :** Méthodes de pondération cost-sensitive, sous-échantillonnage et sur-échantillonnage (SMOTE).
- **Feature Engineering & Importance :** Comparaison des ensembles temporels, spectraux et avancés (Synchronie, Pointes-ondes).
- **Modèle Intra-Patient :** Entraînement dédié et personnalisé pour un patient isolé.
- **Évaluation S-match :** Métriques cliniques par événement (sensibilité réelle, faux positifs par heure, recouvrement temporel).
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients analysés", len(infos))
    c2.metric("Crises totales", int(infos.crises.map(len).sum()))
    c3.metric("Fenêtres totales (2s)", f"{len(dataset):,}".replace(",", " "))
    taux_crise = 100 * dataset.Class.mean()
    c4.metric("Taux de crise (déséquilibre)", f"{taux_crise:.2f} %", delta=f"{100 - taux_crise:.1f}% normal", delta_color="inverse")

    tab = infos[["ID", "Sexe", "Age"]].copy()
    tab["Durée (min)"] = (infos.duree_s / 60).round(1)
    tab["Nombre de crises"] = infos.crises.map(len)
    st.subheader("Base des patients")
    st.dataframe(tab, hide_index=True, width="stretch")


# ========================================================
# 1. EXPLORATION DU SIGNAL
# ========================================================
elif page == "👀 1. Exploration du signal EEG":
    st.title("👀 Exploration du signal EEG")
    c1, c2 = st.columns(2)
    patient = c1.selectbox("Sélection du patient", patients)
    canal = c2.selectbox("Électrode à visualiser", core.CANAUX_EEG, index=core.CANAUX_EEG.index("EEGF3"),
                         format_func=lambda c: c.replace("EEG", ""))
    info = infos.set_index("ID").loc[patient]
    X, fs = signal(patient)
    crises = info.crises

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sexe / Âge", f"{info.Sexe} / {info.Age}")
    m2.metric("Durée", f"{info.duree_s / 60:.1f} min")
    m3.metric("Crises", len(crises))
    m4.metric("Durée moy. crise", f"{np.mean([f - d for d, f in crises]):.1f} s" if crises else "N/A")

    st.subheader("Tracé continu de l'enregistrement")
    st.pyplot(courbe_eeg(X, fs, crises, canal))

    if crises:
        st.subheader("🔬 Zoom multi-électrodes sur une crise")
        k = st.selectbox("Sélectionner la crise", range(1, len(crises) + 1), format_func=lambda i: f"Crise n°{i}")
        d, f = crises[k - 1]
        st.pyplot(zoom_multicanal(X, fs, crises, d - 8, f + 8))


# ========================================================
# 2. STRATÉGIES DE VALIDATION
# ========================================================
elif page == "🧪 2. Stratégies de Validation":
    st.title("🧪 Comparaison des Stratégies de Validation")
    st.markdown("""
En traitement de signaux biomédicaux, le choix du schéma de validation est crucial pour éviter la **fuite d'information (data leakage)** :
1. **Holdout 80/20 standard :** Mélange aléatoire des fenêtres de tous les patients (surestimation optimiste).
2. **5-Fold Stratifié :** Validation croisée avec préservation des proportions de classe mais mélange des patients.
3. **Group 5-Fold (Inter-Patients) :** Les patients d'un bloc de test ne sont **jamais** vus en entraînement. C'est l'évaluation clinique réelle.
""")
    c1, c2 = st.columns(2)
    modele = c1.selectbox("Modèle à évaluer", core.NOMS_MODELES, index=0)
    feat_mode = c2.selectbox("Descripteurs utilisés", ["all", "base", "temporelles", "frequentielles", "top_k"], index=0)

    if st.button("🚀 Lancer l'évaluation comparative", type="primary"):
        with st.spinner("Exécution des schémas de validation croisée..."):
            df_val = comparer_strategies_validation(nom_modele=modele, feature_set=feat_mode)
        st.success("Évaluation terminée !")
        cols_float = ["Accuracy", "Precision", "Recall", "F1-score"]
        fmt_dict = {col: "{:.3f}" for col in cols_float}
        st.dataframe(df_val.style.format(fmt_dict).highlight_max(subset=cols_float, axis=0, color="#d4edda"), width="stretch")

        st.subheader("Impact sur le F1-score et le Rappel")
        st.bar_chart(df_val[["Recall", "Precision", "F1-score"]])
        st.info("💡 **Constat clinique :** Le score en **Group 5-Fold** est généralement plus bas mais reflète fidèlement la capacité à généraliser sur un patient inconnu.")


# ========================================================
# 3. RÉSOLUTION DU DÉSÉQUILIBRE
# ========================================================
elif page == "⚖️ 3. Résolution du Déséquilibre":
    st.title("⚖️ Résolution du Déséquilibre de Classes")
    st.markdown("""
Les crises ne représentent que **~3% des données**. Sans traitement adapté, un modèle prédisant systématiquement « Normal » obtiendrait 97% d'accuracy tout en ratant 100% des crises !

Ici, nous comparons les approches pour traiter ce déséquilibre :
- **Cost-Sensitive Learning :** Ajustement des poids des classes (`class_weight='balanced'`).
- **Undersampling :** Réduction de la classe majoritaire pour équilibrer le ratio.
- **SMOTE / Oversampling :** Synthèse d'exemples de crises dans l'espace des descripteurs.
""")
    c1, c2 = st.columns(2)
    modele = c1.selectbox("Modèle d'IA", ["Random Forest", "Decision Tree", "SVM"], index=0)
    split_p = c2.slider("Pourcentage de test", 10, 40, 20)

    if st.button("⚡ Comparer les méthodes d'équilibrage", type="primary"):
        with st.spinner("Entraînement des variantes avec et sans équilibrage..."):
            X = core.preparer_X(dataset, feature_set="all")
            y = dataset["Class"].values
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=split_p / 100.0, stratify=y, random_state=core.RANDOM_STATE)

            configs = [
                ("Sans équilibrage (Baseline)", "none", "none"),
                ("Pondération (Class Weights)", "balanced", "none"),
                ("Sous-échantillonnage (Undersampling)", "none", "undersampling"),
                ("Sur-échantillonnage synthétique (SMOTE)", "none", "smote"),
                ("Pondération + SMOTE combinés", "balanced", "smote")
            ]
            resultats_eq = []
            mats = {}

            for titre, eq, samp in configs:
                X_samp, y_samp = core.reechantillonner(Xtr, ytr, methode=samp, ratio=0.3)
                clf = core.creer_modele(modele, equilibrage=eq).fit(X_samp, y_samp)
                p = clf.predict(Xte)
                resultats_eq.append({
                    "Méthode d'équilibrage": titre,
                    "Recall (Crises détectées)": recall_score(yte, p, zero_division=0),
                    "Precision": precision_score(yte, p, zero_division=0),
                    "F1-score": f1_score(yte, p, zero_division=0),
                    "Accuracy": accuracy_score(yte, p)
                })
                mats[titre] = confusion_matrix(yte, p)

            df_eq = pd.DataFrame(resultats_eq).set_index("Méthode d'équilibrage")
            st.session_state["res_eq"] = (df_eq, mats)

    if "res_eq" in st.session_state:
        df_eq, mats = st.session_state["res_eq"]
        st.dataframe(df_eq.style.format("{:.3f}").highlight_max(subset=["Recall (Crises détectées)", "F1-score"], color="#d4edda"), width="stretch")
        st.bar_chart(df_eq[["Recall (Crises détectées)", "Precision", "F1-score"]])

        st.subheader("Matrices de confusion comparatives")
        cols = st.columns(len(mats))
        for col, (titre, M) in zip(cols, mats.items()):
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            ConfusionMatrixDisplay(M, display_labels=["Norm", "Crise"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(titre.split("(")[0].strip(), fontsize=8)
            col.pyplot(fig)


# ========================================================
# 4. COMPARAISON DES FEATURES
# ========================================================
elif page == "📊 4. Comparaison des Features":
    st.title("📊 Comparaison et Importance des Features")
    st.markdown("""
Le jeu de données intègre 20 descripteurs :
- **Temporels :** Moyenne, Variance, RMS, Énergie, LineLength...
- **Spectraux (FFT) :** Puissances et ratios par bande (Delta, Theta, Alpha, Beta).
- **Spécifiques Absence :** Puissance des pointes-ondes 3 Hz (`P_SW_2.5-4.5`) et **Synchronie inter-électrodes**.
""")

    tab1, tab2 = st.tabs(["🏆 Importance des Features", "🔄 Performance par sous-ensemble"])

    with tab1:
        if st.button("Calculer l'importance globale (Random Forest)"):
            with st.spinner("Calcul des importances Gini..."):
                imp = evaluer_importance_features()
                st.session_state["feat_imp"] = imp

        if "feat_imp" in st.session_state:
            imp = st.session_state["feat_imp"]
            st.subheader("Top des descripteurs les plus discriminants")
            st.bar_chart(imp)
            st.write(pd.DataFrame({"Importance normalisée": imp}).head(10))

    with tab2:
        st.markdown("Comparaison des performances selon le groupe de features employé :")
        if st.button("Lancer le benchmark par groupe de descripteurs"):
            with st.spinner("Entraînement des différents sous-ensembles..."):
                groupes = {
                    "Tous les descripteurs (20)": "all",
                    "Descripteurs de base (12)": "base",
                    "Temporels uniquement": "temporelles",
                    "Fréquentiels / Spectraux": "frequentielles",
                    "Top 8 descripteurs ciblés": "top_k"
                }
                res_groupes = []
                y = dataset["Class"].values
                for nom_g, fset in groupes.items():
                    X = core.preparer_X(dataset, feature_set=fset)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=core.RANDOM_STATE)
                    clf = core.creer_modele("Random Forest", equilibrage="balanced").fit(Xtr, ytr)
                    p = clf.predict(Xte)
                    res_groupes.append({
                        "Groupe": nom_g,
                        "Nombre de features": X.shape[1],
                        "Recall": recall_score(yte, p, zero_division=0),
                        "Precision": precision_score(yte, p, zero_division=0),
                        "F1-score": f1_score(yte, p, zero_division=0)
                    })
                df_grp = pd.DataFrame(res_groupes).set_index("Groupe")
                st.dataframe(df_grp.style.format({"Recall": "{:.3f}", "Precision": "{:.3f}", "F1-score": "{:.3f}"}), width="stretch")
                st.bar_chart(df_grp[["F1-score", "Recall"]])


# ========================================================
# 5. MODÈLE INTRA-PATIENT (DÉDIÉ)
# ========================================================
elif page == "🎯 5. Modèle Intra-Patient (Dédié)":
    st.title("🎯 Entraînement d'un Modèle Dédié pour un Seul Patient")
    st.markdown("""
Dans un contexte de **suivi ambulatoire personnalisé**, un modèle est entraîné sur le début de l'enregistrement d'un patient donné, puis déployé sur la suite de ses signaux.

Cette approche permet de capturer la signature électrophysiologique unique de chaque individu.
""")
    # Filtrer les patients ayant au moins 2 crises pour permettre un train/test pertinent
    patients_eligibles = [p for p in patients if len(infos.set_index("ID").loc[p].crises) >= 2]

    c1, c2, c3 = st.columns(3)
    patient_sel = c1.selectbox("Patient sélectionné", patients_eligibles, index=0)
    modele_sel = c2.selectbox("Modèle", core.NOMS_MODELES, index=0)
    split_train = c3.slider("Part d'entraînement temporel (%)", 40, 80, 60)

    if st.button("🏋️ Entraîner le modèle personnalisé", type="primary"):
        with st.spinner("Entraînement intra-patient en cours..."):
            t_test, y_test, p_test, met, M = entrainer_intra_patient(
                patient=patient_sel, nom_modele=modele_sel, train_ratio=split_train / 100.0,
                equilibrage="balanced", methode_sampling="undersampling"
            )
            st.session_state["intra_res"] = (t_test, y_test, p_test, met, M, patient_sel)

    if "intra_res" in st.session_state and st.session_state["intra_res"][5] == patient_sel:
        t_test, y_test, p_test, met, M, p_id = st.session_state["intra_res"]
        st.success(f"Modèle personnalisé entraîné avec succès pour le patient **{p_id}** !")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-score Intra", f"{met['F1-score']:.3f}")
        c2.metric("Rappel (Sensibilité)", f"{met['Recall']:.3f}")
        c3.metric("Précision", f"{met['Precision']:.3f}")
        c4.metric("Fenêtres Test (non vues)", met['N_Test'])

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Matrice de confusion sur la phase Test")
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ConfusionMatrixDisplay(M, display_labels=["Normal", "Crise"]).plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")
            st.pyplot(fig)

        with col_right:
            st.subheader("Reconstruction des événements Test")
            crises_detectees = core.reconstruire_crises(t_test, p_test)
            crises_reelles_tot = infos.set_index("ID").loc[p_id].crises
            # Filtrer les crises réelles dans la période de test
            t_debut_test = t_test[0] if len(t_test) > 0 else 0
            crises_reelles_test = [(d, f) for d, f in crises_reelles_tot if f > t_debut_test]

            eval_test = core.evaluer_smatch(crises_reelles_test, crises_detectees, duree_totale_heures=(t_test[-1] - t_test[0]) / 3600.0 if len(t_test) else 1)
            st.metric("Crises réelles détectées", f"{eval_test['Reelles_Trouvees']} / {eval_test['Total_Reelles']}")
            st.metric("Fausses alarmes sur la phase test", eval_test['Faux_Positifs'])


# ========================================================
# 6. DÉTECTION CLINIQUE & S-MATCH
# ========================================================
elif page == "🔎 6. Détection Clinique & S-match":
    st.title("🔎 Détection Clinique & Évaluation Événementielle (S-match)")
    st.markdown("""
Le modèle s'entraîne sur **tous les autres patients**, puis diagnostique le patient sélectionné (généralisation stricte).
L'évaluation utilise les métriques **S-match** recommandées en clinique (taux de crises réelles identifiées, taux de recouvrement temporel, et fausses alarmes par heure).
""")
    c1, c2, c3 = st.columns(3)
    patient = c1.selectbox("Patient à tester", patients, index=patients.index("230718A-A") if "230718A-A" in patients else 0)
    nom_modele = c2.selectbox("Algorithme d'IA", core.NOMS_MODELES, index=0)
    seuil_proba = c3.slider("Seuil de décision (Threshold)", 0.2, 0.8, 0.5, step=0.05)

    c4, c5 = st.columns(2)
    methode_eq = c4.selectbox("Stratégie d'équilibrage", ["balanced", "none"], index=0)
    sampling_mode = c5.selectbox("Rééchantillonnage Train", ["none", "undersampling", "smote", "oversampling"], index=0)

    if st.button("🔬 Diagnostiquer le patient", type="primary"):
        st.session_state["detect_smatch"] = (patient, nom_modele, seuil_proba, methode_eq, sampling_mode)

    if st.session_state.get("detect_smatch") == (patient, nom_modele, seuil_proba, methode_eq, sampling_mode):
        with st.spinner("Diagnostic du patient en cours..."):
            t_deb, y_vrai, pred, proba = detecter(
                patient, nom_modele, equilibrage=methode_eq, methode_sampling=sampling_mode, seuil=seuil_proba
            )
        reelles = infos.set_index("ID").loc[patient].crises
        detectees = core.reconstruire_crises(t_deb, pred)
        duree_h = (infos.set_index("ID").loc[patient].duree_s) / 3600.0

        # Évaluation S-match complète
        res_smatch = core.evaluer_smatch(reelles, detectees, duree_totale_heures=duree_h)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sensibilité Crises (Recall)", f"{res_smatch['Event_Recall']*100:.1f} %", f"{res_smatch['Reelles_Trouvees']}/{res_smatch['Total_Reelles']} crises")
        m2.metric("Précision Événements", f"{res_smatch['Event_Precision']*100:.1f} %")
        m3.metric("F1-score Événementiel", f"{res_smatch['Event_F1']:.3f}")
        m4.metric("Faux Positifs / Heure", f"{res_smatch['FP_par_heure']:.1f} FP/h", delta=f"{res_smatch['Faux_Positifs']} au total", delta_color="inverse")

        X, fs = signal(patient)
        st.subheader("Signal EEG et Détections")
        st.caption("🟩 Vert (haut) : crises réelles (Excel)   🟥 Rouge (bas) : crises détectées par le modèle.")
        st.pyplot(courbe_eeg(X, fs, reelles, "EEGF3", detections=detectees))

        st.subheader("📋 Tableau de Correspondance S-match (Crise par Crise)")
        st.dataframe(res_smatch["Detail"], hide_index=True, width="stretch")

        if res_smatch["Fausses_Alarmes_Intervalles"]:
            st.warning("⚠️ Fausses alarmes (intervalles détectés hors des annotations Excel) : " +
                       ", ".join(f"{a:.0f}–{b:.0f} s" for a, b in res_smatch["Fausses_Alarmes_Intervalles"]))

        st.subheader("🔬 Zoom interactif sur le signal")
        duree = float(len(X) / fs)
        borne_init = (max(0.0, reelles[0][0] - 10), min(duree, reelles[0][1] + 10)) if reelles else (0.0, min(60.0, duree))
        a, b = st.slider("Fenêtre temporelle d'inspection (s)", 0.0, duree, borne_init, step=1.0)
        if b - a > 120:
            st.info("💡 Veuillez sélectionner une fenêtre de moins de 120 secondes pour un affichage optimal.")
        else:
            fig = zoom_multicanal(X, fs, reelles, a, b)
            for d, f in detectees:
                if f > a and d < b:
                    fig.axes[0].axvspan(max(d, a), min(f, b), ymin=0, ymax=.03, color="tab:red")
            st.pyplot(fig)
            st.caption("Zone rouge transparent = crise réelle. Barre rouge en bas = détection du modèle.")
