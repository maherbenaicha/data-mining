# Interface simple pour le projet "Détection des crises d'absence"
# Lancement : double-clic sur Lancer_application.bat  (ou : python -m streamlit run app.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

import eeg_core as core

st.set_page_config(page_title="Crises d'absence - EEG", page_icon="🧠", layout="wide")


# ---------- Données (mises en cache pour aller vite) ----------
@st.cache_data(show_spinner=False)
def base():
    barre = st.progress(0.0, "Préparation des données (une seule fois, ~2 minutes)...")
    infos, dataset = core.construire_base(lambda p, txt: barre.progress(p, txt))
    barre.empty()
    return infos, dataset


@st.cache_data(show_spinner="Chargement du signal EEG...", max_entries=3)
def signal(patient):
    X, fs = core.charger_eeg(patient)
    return core.filtrer(X, fs), fs


@st.cache_data(show_spinner=False)
def detecter(patient, nom_modele):
    # Entraîne le modèle sur tous les AUTRES patients, puis prédit le patient choisi
    _, dataset = base()
    X = core.preparer_X(dataset)
    y = dataset["Class"].values
    masque = (dataset.Patient == patient).values
    m = core.creer_modele(nom_modele).fit(X[~masque], y[~masque])
    d = dataset[masque].sort_values("t_debut")
    pred = m.predict(X.loc[d.index])
    return d.t_debut.values, d.Class.values, pred


@st.cache_data(show_spinner=False)
def comparer_modeles():
    _, dataset = base()
    X, y = core.preparer_X(dataset), dataset["Class"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=core.RANDOM_STATE)
    res, mats = [], {}
    for nom in core.NOMS_MODELES:
        p = core.creer_modele(nom).fit(Xtr, ytr).predict(Xte)
        res.append({"Modèle": nom, "Accuracy": accuracy_score(yte, p), "Precision": precision_score(yte, p),
                    "Recall": recall_score(yte, p), "F1-score": f1_score(yte, p)})
        mats[nom] = confusion_matrix(yte, p)
    return pd.DataFrame(res).set_index("Modèle"), mats


def courbe_eeg(X, fs, crises, canal, detections=None, t0=None, t1=None):
    t = np.arange(len(X)) / fs
    i0, i1 = int((t0 or 0) * fs), int((t1 or t[-1]) * fs)
    pas = max(1, (i1 - i0) // 40000)                     # on allège l'affichage
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


# ---------- Menu ----------
st.sidebar.title("🧠 Crises d'absence")
page = st.sidebar.radio("Que veux-tu faire ?", [
    "🏠 Accueil",
    "👀 1. Voir le signal d'un patient",
    "🧪 2. Comparer les modèles",
    "🔎 3. Détecter les crises d'un patient",
])
st.sidebar.markdown("---")
st.sidebar.caption("Clique sur une étape dans l'ordre. Les calculs longs ne sont faits qu'une seule fois.")

infos, dataset = base()
patients = infos.ID.tolist()


# ---------- Pages ----------
if page == "🏠 Accueil":
    st.title("Détection automatique des crises d'absence 🧠")
    st.markdown("""
Cette application refait le travail du notebook, **avec des boutons**, sans écrire de code.

**L'idée en 3 phrases :**
1. On enregistre l'activité électrique du cerveau (**EEG**) avec 19 électrodes sur la tête.
2. Pendant une **crise d'absence**, le signal devient très ample et régulier (≈ 3 oscillations par seconde).
3. On apprend à un **modèle d'intelligence artificielle** à reconnaître ces moments, pour trouver le **début et la fin de chaque crise** automatiquement.

**Comment utiliser l'application :** choisis une étape dans le menu à gauche 👈
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(infos))
    c2.metric("Crises annotées", int(infos.crises.map(len).sum()))
    c3.metric("Fenêtres de 2 s", f"{len(dataset):,}".replace(",", " "))
    c4.metric("Fenêtres en crise", f"{100 * dataset.Class.mean():.1f} %")

    tab = infos[["ID", "Sexe", "Age"]].copy()
    tab["Durée (min)"] = (infos.duree_s / 60).round(1)
    tab["Nombre de crises"] = infos.crises.map(len)
    st.subheader("Liste des patients")
    st.dataframe(tab, hide_index=True, width="stretch")


elif page == "👀 1. Voir le signal d'un patient":
    st.title("👀 Voir le signal EEG d'un patient")
    c1, c2 = st.columns(2)
    patient = c1.selectbox("Choisis un patient", patients)
    canal = c2.selectbox("Choisis une électrode", core.CANAUX_EEG, index=core.CANAUX_EEG.index("EEGF3"),
                         format_func=lambda c: c.replace("EEG", ""))
    info = infos.set_index("ID").loc[patient]
    X, fs = signal(patient)
    crises = info.crises

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sexe / Âge", f"{info.Sexe} / {info.Age}")
    m2.metric("Durée de l'enregistrement", f"{info.duree_s / 60:.1f} min")
    m3.metric("Nombre de crises", len(crises))
    m4.metric("Durée moyenne d'une crise", f"{np.mean([f - d for d, f in crises]):.1f} s")

    st.subheader("Tout l'enregistrement")
    st.caption("Les zones **rouges** sont les crises indiquées dans le fichier Excel.")
    st.pyplot(courbe_eeg(X, fs, crises, canal))

    st.subheader("Les crises de ce patient")
    tab = pd.DataFrame(crises, columns=["Début (s)", "Fin (s)"]).round(1)
    tab["Durée (s)"] = (tab["Fin (s)"] - tab["Début (s)"]).round(1)
    tab.index = np.arange(1, len(tab) + 1); tab.index.name = "Crise"
    st.dataframe(tab, width="stretch")

    st.subheader("🔬 Zoom sur une crise (les 19 électrodes)")
    k = st.selectbox("Quelle crise ?", range(1, len(crises) + 1), format_func=lambda i: f"Crise n°{i}")
    d, f = crises[k - 1]
    st.pyplot(zoom_multicanal(X, fs, crises, d - 8, f + 8))
    st.info("👉 Pendant la crise (zone rouge), **toutes les électrodes** s'agitent en même temps "
            "avec des « pointes-ondes » régulières, environ 3 par seconde.")


elif page == "🧪 2. Comparer les modèles":
    st.title("🧪 Comparer les 5 modèles d'IA")
    st.markdown("""
On donne **80 %** des fenêtres de 2 s au modèle pour **apprendre**, puis on le **teste** sur les 20 % restants.

- **Recall** = part des crises trouvées (le plus important : on ne veut pas rater de crise)
- **Precision** = quand le modèle dit « crise », a-t-il raison ?
- **F1-score** = la note globale qui combine les deux ➜ **c'est elle qui désigne le meilleur modèle**
""")
    if st.button("▶️ Lancer la comparaison (environ 1 minute)", type="primary"):
        st.session_state["comparer"] = True
    if st.session_state.get("comparer"):
        with st.spinner("Entraînement des modèles..."):
            res, mats = comparer_modeles()
        meilleur = res["F1-score"].idxmax()
        st.success(f"🏆 Meilleur modèle : **{meilleur}** (F1 = {res.loc[meilleur, 'F1-score']:.3f})")
        st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#c8f7c5"), width="stretch")
        st.bar_chart(res[["Precision", "Recall", "F1-score"]])

        st.subheader("Matrices de confusion")
        st.caption("Ligne du bas = vraies crises : à gauche celles **ratées**, à droite celles **trouvées**.")
        cols = st.columns(len(mats))
        for col, (nom, M) in zip(cols, mats.items()):
            fig, ax = plt.subplots(figsize=(3, 3))
            ConfusionMatrixDisplay(M, display_labels=["Normal", "Crise"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.set_title(nom, fontsize=9)
            col.pyplot(fig)


elif page == "🔎 3. Détecter les crises d'un patient":
    st.title("🔎 Détecter automatiquement les crises")
    st.markdown("Le modèle apprend sur **tous les autres patients**, puis analyse le patient choisi "
                "**comme s'il ne l'avait jamais vu** (comme un vrai nouveau patient à l'hôpital).")
    c1, c2 = st.columns(2)
    patient = c1.selectbox("Choisis un patient", patients, index=patients.index("230718A-A") if "230718A-A" in patients else 0)
    nom_modele = c2.selectbox("Choisis un modèle", core.NOMS_MODELES)

    if st.button("🔎 Détecter les crises", type="primary"):
        st.session_state["detect"] = (patient, nom_modele)
    if st.session_state.get("detect") == (patient, nom_modele):
        with st.spinner("Le modèle apprend puis analyse le signal (10 à 60 secondes)..."):
            t_deb, y_vrai, pred = detecter(patient, nom_modele)
        reelles = infos.set_index("ID").loc[patient].crises
        detectees = core.reconstruire_crises(t_deb, pred)
        tab, fausses = core.comparer(reelles, detectees)
        trouvees = (tab["Résultat"] == "✅ Détectée").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Crises trouvées", f"{trouvees} / {len(reelles)}")
        m2.metric("Crises ratées", len(reelles) - trouvees)
        m3.metric("Fausses alarmes", len(fausses))

        X, fs = signal(patient)
        st.subheader("Résultat sur tout l'enregistrement")
        st.caption("🟩 En haut : crises réelles (Excel)   🟥 En bas : crises détectées par l'IA. "
                   "Quand les deux sont alignés, c'est réussi !")
        st.pyplot(courbe_eeg(X, fs, reelles, "EEGF3", detections=detectees))

        st.subheader("Détail crise par crise")
        st.dataframe(tab, hide_index=True, width="stretch")
        if fausses:
            st.warning("⚠️ Fausses alarmes (détections sans crise dans l'Excel) : " +
                       ", ".join(f"{a:.0f}–{b:.0f} s" for a, b in fausses) +
                       ". Parfois, c'est une vraie crise oubliée dans l'Excel ! Regarde-la avec le zoom ci-dessous.")

        st.subheader("🔬 Zoom libre")
        duree = float(len(X) / fs)
        a, b = st.slider("Choisis la zone à regarder (en secondes)", 0.0, duree,
                         (max(0.0, reelles[0][0] - 10), min(duree, reelles[0][1] + 10)), step=1.0)
        if b - a > 120:
            st.info("Choisis une zone de moins de 120 secondes pour bien voir.")
        else:
            fig = zoom_multicanal(X, fs, reelles, a, b)
            for d, f in detectees:
                if f > a and d < b:
                    fig.axes[0].axvspan(max(d, a), min(f, b), ymin=0, ymax=.03, color="tab:red")
            st.pyplot(fig)
            st.caption("Zone rose = crise réelle. Barre rouge en bas = détection de l'IA.")
