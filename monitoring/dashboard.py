from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "monitoring" / "logs" / "production_logs.jsonl"
X_TRAIN_PATH = PROJECT_ROOT / "data" / "X_train.pkl"
EVIDENTLY_REPORT_PATH = PROJECT_ROOT / "reports" / "evidently_data_drift_report.html"

MODEL_NAME = "LightGBM optimisé"
DECISION_THRESHOLD = 0.5
PSI_WARNING_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2


st.set_page_config(
    page_title="Credit Scoring Monitoring",
    layout="wide"
)

# Style
st.markdown(

    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
 
    .stApp {
        background: #f8fafc;
        color: #0f172a;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 26px;
        margin-bottom: 14px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        height: 100%;
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }

    .kpi-help {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 6px;
    }

    .status-ok {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #22c55e;
        border-radius: 16px;
        padding: 22px;
    }

    .status-warning {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 6px solid #f97316;
        border-radius: 16px;
        padding: 22px;
    }

    .sidebar-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 16px;
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Chargement des données
@st.cache_data
def load_training_data():
    return joblib.load(X_TRAIN_PATH)


@st.cache_data
def load_logs():
    if not LOG_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    logs = pd.read_json(LOG_PATH, lines=True)
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])

    features = pd.json_normalize(logs["features"])

    return logs, features


# PSI
def calculate_psi(reference, current, buckets=10):
    epsilon = 1e-4

    reference = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna()
    current = pd.Series(current).replace([np.inf, -np.inf], np.nan).dropna()

    if reference.dtype == "bool":
        reference = reference.astype(int)

    if current.dtype == "bool":
        current = current.astype(int)

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return np.nan

    bornes = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    bornes = np.unique(bornes)

    if len(bornes) <= 2:
        return np.nan

    distribution_reference = (
        pd.cut(reference, bins=bornes, include_lowest=True)
        .value_counts(normalize=True)
    )

    distribution_production = (
        pd.cut(current, bins=bornes, include_lowest=True)
        .value_counts(normalize=True)
    )

    psi = 0

    for intervalle in distribution_reference.index:
        proportion_reference = max(distribution_reference.get(intervalle, epsilon), epsilon)
        proportion_production = max(distribution_production.get(intervalle, epsilon), epsilon)

        psi += (
            (proportion_production - proportion_reference)
            * np.log(proportion_production / proportion_reference)
        )

    return psi


def interpret_psi(psi):
    if pd.isna(psi):
        return "Non calculable"
    if psi < PSI_WARNING_THRESHOLD:
        return "Stable"
    if psi < PSI_ALERT_THRESHOLD:
        return "À surveiller"
    return "Drift significatif"


def build_psi_table(reference_data, production_data):
    production_data = production_data.reindex(
        columns=reference_data.columns,
        fill_value=0
    )

    results = []

    for column in reference_data.columns:
        psi = calculate_psi(reference_data[column], production_data[column])

        results.append({
            "Variable": column,
            "PSI": round(psi, 4) if not pd.isna(psi) else np.nan,
            "Statut": interpret_psi(psi)
        })

    return (
        pd.DataFrame(results)
        .sort_values("PSI", ascending=False)
        .reset_index(drop=True)
    )

# Helpers UI
def kpi_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


# Graphiques
def plot_scores(logs):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(logs["score"], bins=20, color="#93c5fd", edgecolor="white")
    ax.axvline(
        DECISION_THRESHOLD,
        color="#334155",
        linestyle="--",
        linewidth=2,
        label="Seuil de décision"
    )
    ax.set_title("Distribution des scores prédits", fontsize=13, fontweight="medium")
    ax.set_xlabel("Score de risque")
    ax.set_ylabel("Nombre de prédictions")
    ax.legend()
    style_axis(ax)
    st.pyplot(fig)

def plot_scores_over_time(logs):
    logs_sorted = logs.sort_values("timestamp").copy()
    logs_sorted["numero_requete"] = range(1, len(logs_sorted) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        logs_sorted["numero_requete"],
        logs_sorted["score"],
        color="#8593cf",
        linewidth=2,
        marker="o",
        markersize=3,
        alpha=0.8
    )

    ax.axhline(
        DECISION_THRESHOLD,
        color="#334155",
        linestyle="--",
        linewidth=2,
        label="Seuil de décision"
    )

    ax.set_title("Évolution des scores au fil des requêtes", fontsize=13, fontweight="medium")
    ax.set_xlabel("Numéro de requête")
    ax.set_ylabel("Score prédit")
    ax.legend()
    style_axis(ax)

    st.pyplot(fig)

def plot_decisions(logs):
    accepted = int((logs["prediction"] == 0).sum())
    risky = int((logs["prediction"] == 1).sum())

    fig, ax = plt.subplots(figsize=(4.5,4.5))

    ax.pie(
        [accepted, risky],
        labels=["Acceptés", "Risque élevé"],
        autopct="%1.1f%%",
        startangle=90,
        radius=0.82,
        colors=["#dae6bb", "#8593cf"],
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white"
        },
        textprops={
            "fontsize": 6,
            "fontweight": "medium",
            "color": "#334155"
        }
    )

    ax.text(
        0,
        0,
        f"Total\n{len(logs)}",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="medium",
        color="#0f172a"
    )
    ax.set_title("Répartition des décisions", fontsize=8, fontweight="medium")


    st.pyplot(fig)

def plot_psi_status_summary(resume):
    categories = ["Stable", "À surveiller", "Drift significatif", "Non calculable"]
    values = [int(resume.get(category, 0)) for category in categories]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        categories,
        values,
        color=["#dae6bb", "#f7d9a8", "#e9a5a5", "#d9dee8"]
    )

    ax.set_title(
        "Répartition des variables selon le PSI",
        fontsize=13,
        fontweight="medium"
    )
    ax.set_ylabel("Nombre de variables")
    style_axis(ax)

    st.pyplot(fig)

def plot_latency(logs):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(logs["inference_time_ms"], bins=20, color="#c4b5fd", edgecolor="white")
    ax.set_title("Distribution des temps d'inférence", fontsize=13, fontweight="medium")
    ax.set_xlabel("Temps d'inférence (ms)")
    ax.set_ylabel("Nombre de requêtes")
    style_axis(ax)
    st.pyplot(fig)


def plot_psi_distribution(psi_table):
    psi_valides = psi_table.dropna(subset=["PSI"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(psi_valides["PSI"], bins=30, color="#bfdbfe", edgecolor="white")
    ax.axvline(PSI_WARNING_THRESHOLD, color="#f59e0b", linestyle="--", linewidth=2, label="Surveillance")
    ax.axvline(PSI_ALERT_THRESHOLD, color="#ef4444", linestyle="--", linewidth=2, label="Alerte")
    ax.set_title("Distribution globale des PSI", fontsize=13, fontweight="medium")
    ax.set_xlabel("PSI")
    ax.set_ylabel("Nombre de variables")
    ax.legend()
    style_axis(ax)
    st.pyplot(fig)


def plot_psi_zoom(psi_table):
    psi_valides = psi_table.dropna(subset=["PSI"])
    psi_zoom = psi_valides[psi_valides["PSI"] <= 0.5]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(psi_zoom["PSI"], bins=20, color="#dbeafe", edgecolor="white")
    ax.axvline(PSI_WARNING_THRESHOLD, color="#f59e0b", linestyle="--", linewidth=2, label="Surveillance")
    ax.axvline(PSI_ALERT_THRESHOLD, color="#ef4444", linestyle="--", linewidth=2, label="Alerte")
    ax.set_title("Vue zoomée des PSI entre 0 et 0.5", fontsize=13, fontweight="medium")
    ax.set_xlabel("PSI")
    ax.set_ylabel("Nombre de variables")
    ax.legend()
    style_axis(ax)
    st.pyplot(fig)


def plot_top_psi(psi_table):
    top20 = psi_table.dropna(subset=["PSI"]).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20["Variable"], top20["PSI"], color="#fdba74")
    ax.axvline(PSI_ALERT_THRESHOLD, color="#ef4444", linestyle="--", linewidth=2, label="Seuil alerte")
    ax.set_title("Top 20 des variables avec le PSI le plus élevé", fontsize=13, fontweight="medium")
    ax.set_xlabel("PSI")
    ax.invert_yaxis()
    ax.legend()
    style_axis(ax)
    st.pyplot(fig)


# Interface
st.markdown('<div class="main-title">Dashboard de Monitoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Suivi des prédictions, de la performance API et du data drift du modèle de scoring.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Credit Scoring Monitoring")

    st.markdown("### Informations modèle")
    st.markdown(
        f"""
        <div class="sidebar-box">
        <b>Modèle</b><br>{MODEL_NAME}<br><br>
        <b>Variables utilisées</b><br>304
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Seuils")
    st.markdown(
        f"""
        <div class="sidebar-box">
        <b>Seuil décision</b><br>{DECISION_THRESHOLD}<br><br>
        <b>PSI surveillance</b><br>{PSI_WARNING_THRESHOLD}<br><br>
        <b>PSI alerte</b><br>{PSI_ALERT_THRESHOLD}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Sources")
    st.markdown(
        """
        <div class="sidebar-box">
        <b>Logs de production</b><br>
        monitoring/logs/production_logs.jsonl<br><br>
        <b>Données de référence</b><br>
        data/X_train.pkl
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Dashboard mis à jour automatiquement au lancement.")


X_train = load_training_data()
logs, production_data = load_logs()

if logs.empty:
    st.warning("Aucun log de production trouvé. Lance l'API puis génère des prédictions.")
    st.stop()

psi_table = build_psi_table(X_train, production_data)

score_moyen = logs["score"].mean()
taux_risque = (logs["prediction"] == 1).mean() * 100
temps_moyen = logs["inference_time_ms"].mean()
dernier_log = logs["timestamp"].max().strftime("%d/%m/%Y %H:%M:%S")

resume = psi_table["Statut"].value_counts()
variables_critiques = psi_table[psi_table["PSI"] >= PSI_ALERT_THRESHOLD]

tabs = st.tabs([
    "Vue d'ensemble",
    "Prédictions",
    "Performance API",
    "Data Drift",
    "Recommandations"
])

# Page 1 - Vue d'ensemble
with tabs[0]:
    st.markdown('<div class="section-title">Vue d’ensemble</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card("Prédictions totales", len(logs), "Nombre d'appels API loggés")
    with c2:
        kpi_card("Score moyen", f"{score_moyen:.3f}", "Probabilité moyenne prédite")
    with c3:
        kpi_card("Taux dossiers risqués", f"{taux_risque:.1f} %", "Prédictions au-dessus du seuil")
    with c4:
        kpi_card("Inférence moyenne", f"{temps_moyen:.2f} ms", "Temps moyen de prédiction")
    with c5:
        kpi_card("Dernière prédiction", dernier_log, "Dernier appel enregistré")

    st.markdown('<div class="section-title">Synthèse rapide</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        plot_scores_over_time(logs)

    with col2:
        plot_psi_status_summary(resume)

    st.markdown('<div class="section-title">Résumé du data drift</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        kpi_card("Variables stables", int(resume.get("Stable", 0)), "PSI < 0.1")
    with d2:
        kpi_card("À surveiller", int(resume.get("À surveiller", 0)), "0.1 ≤ PSI < 0.2")
    with d3:
        kpi_card("Drift significatif", int(resume.get("Drift significatif", 0)), "PSI ≥ 0.2")
    with d4:
        kpi_card("Non calculables", int(resume.get("Non calculable", 0)), "Constantes ou trop peu variables")

# Page 2 - Prédictions
with tabs[1]:

    col1, col2 = st.columns([1.4, 1])

    with col1:
        plot_scores(logs)

    with col2:
        plot_decisions(logs)

    st.markdown('<div class="section-title">Dernières prédictions</div>', unsafe_allow_html=True)

    st.dataframe(
        logs[["timestamp", "score", "prediction", "inference_time_ms"]]
        .sort_values("timestamp", ascending=False)
        .head(20),
        use_container_width=True,
        hide_index=True
    )

# Page 3 - API
with tabs[2]:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        plot_latency(logs)

    with col2:
        latency_summary = pd.DataFrame({
            "Métrique": ["Moyenne", "Médiane", "95e percentile", "Maximum", "Minimum"],
            "Temps (ms)": [
                round(logs["inference_time_ms"].mean(), 2),
                round(logs["inference_time_ms"].median(), 2),
                round(logs["inference_time_ms"].quantile(0.95), 2),
                round(logs["inference_time_ms"].max(), 2),
                round(logs["inference_time_ms"].min(), 2),
            ]
        })

        st.markdown("#### Synthèse latence")
        st.dataframe(latency_summary, use_container_width=True, hide_index=True)


# Page 4 - Drift
with tabs[3]:
    st.markdown('<div class="section-title">Rapport Evidently</div>', unsafe_allow_html=True)

    e1, e2 = st.columns([1, 2])

    with e1:
        kpi_card(
            "Rapport HTML",
            "Disponible" if EVIDENTLY_REPORT_PATH.exists() else "Absent"
        )

    with e2:
        if EVIDENTLY_REPORT_PATH.exists():
            with open(EVIDENTLY_REPORT_PATH, "rb") as report_file:
                st.download_button(
                    label="Télécharger le rapport Evidently",
                    data=report_file,
                    file_name="evidently_data_drift_report.html",
                    mime="text/html"
                )
        else:
            st.warning("Rapport Evidently introuvable dans le dossier reports.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analyse PSI</div>', unsafe_allow_html=True)


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Variables stables", int(resume.get("Stable", 0)), "PSI < 0.1")
    with c2:
        kpi_card("À surveiller", int(resume.get("À surveiller", 0)), "0.1 ≤ PSI < 0.2")
    with c3:
        kpi_card("Drift significatif", int(resume.get("Drift significatif", 0)), "PSI ≥ 0.2")
    with c4:
        kpi_card("Non calculables", int(resume.get("Non calculable", 0)), "Variables constantes")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        plot_psi_distribution(psi_table)

    with col2:
        plot_psi_zoom(psi_table)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        plot_top_psi(psi_table)

    with col2:
        st.markdown("#### Variables en alerte")

        if variables_critiques.empty:
            st.success("Aucune variable ne dépasse le seuil d'alerte PSI.")
        else:
            st.warning(f"{len(variables_critiques)} variables dépassent le seuil PSI de {PSI_ALERT_THRESHOLD}.")
            st.dataframe(
                variables_critiques,
                use_container_width=True,
                hide_index=True
            )

    with st.expander("Voir le tableau complet des PSI"):
        st.dataframe(psi_table, use_container_width=True, hide_index=True)

# Page 5 - Décision
with tabs[4]:
    st.markdown('<div class="section-title">Décision & recommandations</div>', unsafe_allow_html=True)

    nb_critiques = len(variables_critiques)
    nb_surveillance = len(
        psi_table[
            (psi_table["PSI"] >= PSI_WARNING_THRESHOLD)
            & (psi_table["PSI"] < PSI_ALERT_THRESHOLD)
        ]
    )

    if nb_critiques == 0:
        st.markdown(
            """
            <div class="status-ok">
            <h3>État actuel : stable</h3>
            Aucun réentraînement immédiat n'est nécessaire.
            Le monitoring doit être maintenu sur les prochaines fenêtres de production.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="status-warning">
            <h3>État actuel : surveillance renforcée</h3>
            {nb_critiques} variables présentent un drift significatif (PSI ≥ 0.2).<br>
            {nb_surveillance} variables présentent un drift modéré à surveiller.<br><br>
            Aucun réentraînement automatique n'est déclenché immédiatement.
            Un réentraînement pourra être envisagé si ces dérives persistent sur plusieurs fenêtres
            ou si une baisse de performance métier est observée.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Variables prioritaires à suivre</div>', unsafe_allow_html=True)
    st.dataframe(
        psi_table.dropna(subset=["PSI"]).head(10),
        use_container_width=True,
        hide_index=True
    )