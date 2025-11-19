# pages/03_Mes-Resultats_et_Plan_action.py
# Synthese DISC + plan d'action EverINSIGHT

import os
import json
from datetime import datetime
import io
import math
import tempfile

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from fpdf import FPDF

st.title("Mes resultats & plan d'action")

# -------------------------------------------------------------------
# 1. Recuperation de l'email (session OU saisie manuelle)
# -------------------------------------------------------------------
session_email = (st.session_state.get("email") or "").strip().lower()

st.markdown(
    """
Pour consulter votre profil DISC, nous avons besoin de l’adresse e-mail utilisee
lorsque vous avez rempli le questionnaire.
"""
)

if session_email:
    st.success(f"Email detecte depuis l’onglet Accueil : **{session_email}**")
else:
    st.info(
        "Vous n’avez pas encore renseigne vos informations dans l’onglet **Accueil** "
        "ou vous avez recharge la page. Vous pouvez saisir directement votre e-mail ci-dessous."
    )

email = st.text_input(
    "Votre adresse e-mail (celle utilisee pour le questionnaire DISC)",
    value=session_email,
).strip().lower()

if not email:
    st.stop()

# -------------------------------------------------------------------
# 2. Chargement du dernier resultat correspondant a cet e-mail
# -------------------------------------------------------------------
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PAGES_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "Data", "logs")
LOG_PATH = os.path.join(LOG_DIR, "disc_forced_sessions.jsonl")

if not os.path.exists(LOG_PATH):
    st.error("Aucun resultat trouve pour l’instant. Le fichier de reponses n’existe pas encore.")
    st.stop()

records = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        user_field = (rec.get("user") or "").strip().lower()
        if user_field == email:
            records.append(rec)

if not records:
    st.warning(
        "Aucun resultat DISC trouve pour cet e-mail. "
        "Vous n’avez peut-etre pas encore valide le questionnaire, "
        "ou vous avez utilise une autre adresse."
    )
    st.stop()

last_rec = records[-1]

scores = last_rec.get("scores", {})
style_code = last_rec.get("style", "")
top_dims = last_rec.get("top_dims", [])

# 👉 Recuperation eventuelle du prenom / nom (si la nouvelle version du questionnaire les enregistre)
prenom = (last_rec.get("prenom") or "").strip()
nom = (last_rec.get("nom") or "").strip()

for k in ["D", "I", "S", "C"]:
    scores.setdefault(k, 0)

# -------------------------------------------------------------------
# 3. Table des scores + radar
# -------------------------------------------------------------------
DIM_LABELS = {
    "D": ("Dominance", "Resultats / decision / vitesse"),
    "I": ("Influence",  "Relation / energie / inspiration"),
    "S": ("Stabilite",  "Cooperation / patience / fiabilite"),
    "C": ("Conformite", "Qualite / precision / normes"),
}

st.subheader("Vos scores DISC")

df = pd.DataFrame(
    [
        {
            "Dimension": k,
            "Libelle": DIM_LABELS[k][0],
            "Score": scores[k],
            "Description": DIM_LABELS[k][1],
        }
        for k in ["D", "I", "S", "C"]
    ]
).sort_values("Score", ascending=False).reset_index(drop=True)

st.dataframe(df, use_container_width=True)

chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Libelle:N", sort="-y"),
    y="Score:Q",
    tooltip=["Libelle", "Score", "Description"],
).properties(height=260)

st.altair_chart(chart, use_container_width=True)

if len(df) >= 2:
    top1, top2 = df.iloc[0], df.iloc[1]
    st.success(
        f"Votre profil est principalement **{top1['Libelle']} ({top1['Dimension']})**, "
        f"avec une energie secondaire **{top2['Libelle']} ({top2['Dimension']})**."
    )

# ---------- Radar / spider chart ----------
st.subheader("Votre profil DISC (radar)")

COLOR = {"D": "#E41E26", "I": "#FFC107", "S": "#2ECC71", "C": "#2E86DE"}
ANGLE_DEG = {"D": 45, "I": 135, "S": 225, "C": 315}

def pol2xy(angle_deg, r):
    a = math.radians(angle_deg)
    return (r * math.cos(a), r * math.sin(a))

def xy2pol(x, y):
    r = math.hypot(x, y)
    a = (math.degrees(math.atan2(y, x)) + 360) % 360
    return a, r

def scale_r(score, rmin=0.10, rmax=0.95, max_score=25):
    score = max(0, min(score, max_score))
    return rmin + (rmax - rmin) * (score / max_score)

rD = scale_r(scores["D"])
rI = scale_r(scores["I"])
rS = scale_r(scores["S"])
rC = scale_r(scores["C"])

radar_pts = {
    "D": pol2xy(45, rD),
    "I": pol2xy(135, rI),
    "S": pol2xy(225, rS),
    "C": pol2xy(315, rC),
}

ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
dims_top2 = [ordered[0][0], ordered[1][0]]
x1, y1 = radar_pts[dims_top2[0]]
x2, y2 = radar_pts[dims_top2[1]]
xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
marker_angle_deg, marker_r = xy2pol(xm, ym)

fig = plt.figure(figsize=(4.8, 4.8))
ax = plt.subplot(111, projection="polar")
plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)

sectors = {
    "D": (math.radians(0), math.radians(90)),
    "I": (math.radians(90), math.radians(180)),
    "S": (math.radians(180), math.radians(270)),
    "C": (math.radians(270), math.radians(360)),
}
for k, (start, end) in sectors.items():
    theta = [start + t * (end - start) / 120 for t in range(121)]
    rr = [1.0] * len(theta)
    ax.fill(theta, rr, alpha=0.24, color=COLOR[k], edgecolor="none")

for r, lw in [(0.30, 1), (0.42, 1), (0.90, 1.2)]:
    ax.plot([0, 2 * math.pi], [r, r], color="#bdbdbd", linewidth=lw)

for ang in [45, 135, 225, 315]:
    ax.plot(
        [math.radians(ang), math.radians(ang)],
        [0, 1],
        color="#d9d9d9",
        linewidth=1,
        linestyle="--",
        zorder=2,
    )

ax.spines["polar"].set_visible(False)

dominant_dim = ordered[0][0]
radar_color = COLOR[dominant_dim]
thetas = [
    math.radians(45),
    math.radians(135),
    math.radians(225),
    math.radians(315),
    math.radians(45),
]
radii = [rD, rI, rS, rC, rD]

ax.fill(thetas, radii, color=radar_color, alpha=0.10, zorder=3)
ax.plot(thetas, radii, color=radar_color, linewidth=1.8, zorder=4)
ax.scatter(thetas[:-1], [rD, rI, rS, rC], s=28, c=radar_color, zorder=5)

ax.plot([math.radians(45), math.radians(45)], [0, rD], color=radar_color, linewidth=1.0)
ax.plot([math.radians(135), math.radians(135)], [0, rI], color=radar_color, linewidth=1.0)
ax.plot([math.radians(225), math.radians(225)], [0, rS], color=radar_color, linewidth=1.0)
ax.plot([math.radians(315), math.radians(315)], [0, rC], color=radar_color, linewidth=1.0)

# 👉 Affichage du prenom si disponible, sinon email, sinon "participant"
if prenom:
    name_display = prenom
else:
    name_display = email or "participant"

ax.scatter(
    math.radians(marker_angle_deg),
    marker_r,
    s=170,
    c="#D32F2F",
    edgecolors="none",
    zorder=6,
)
label_r = max(0.05, marker_r - 0.08)
ax.text(
    math.radians(marker_angle_deg),
    label_r,
    name_display,
    ha="center",
    va="top",
    fontsize=11,
    color="#333",
    zorder=7,
)

ax.text(math.radians(45), 1.03, "D", color=COLOR["D"], ha="center", va="center",
        fontsize=14, fontweight="bold")
ax.text(math.radians(135), 1.03, "I", color=COLOR["I"], ha="center", va="center",
        fontsize=14, fontweight="bold")
ax.text(math.radians(225), 1.03, "S", color=COLOR["S"], ha="center", va="center",
        fontsize=14, fontweight="bold")
ax.text(math.radians(315), 1.03, "C", color=COLOR["C"], ha="center", va="center",
        fontsize=14, fontweight="bold")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rticks([])
ax.set_thetagrids([])
ax.set_rlim(0, 1.05)

# ---> Sauvegarde du radar en PNG (pour le PDF)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
st.session_state["radar_png"] = buf.getvalue()

left, mid, right = st.columns([1, 2, 1])
with mid:
    st.pyplot(fig, clear_figure=True)

st.caption(
    "Le point rouge est place au **milieu** entre vos deux energies les plus fortes. "
    "Le radar colore represente l’intensite relative de chaque dimension DISC."
)

# -------------------------------------------------------------------
# 4. Lecture de profil + axes de reflexion
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Lecture de votre profil")

DIM_NATURAL_STRENGTHS = {
    "D": "Vous aimez relever des defis, aller vite et orienter les decisions.",
    "I": "Vous mettez facilement de l’energie et du lien dans le groupe.",
    "S": "Vous favorisez la cooperation, l’ecoute et un climat stable.",
    "C": "Vous apportez de la rigueur, de la precision et le sens des normes.",
}

DIM_EXCESS = {
    "D": "En exces, vous pouvez aller trop vite, imposer vos vues ou prendre peu de temps pour ecouter.",
    "I": "En exces, vous pouvez beaucoup parler, vous disperser ou perdre de vue l’objectif.",
    "S": "En exces, vous pouvez eviter les conflits, trop vous adapter ou avoir du mal a dire non.",
    "C": "En exces, vous pouvez sur-structurer, rechercher trop de details ou avoir du mal a decider.",
}

DIM_DEV = {
    "D": "Gagner a ecouter davantage, poser des questions et partager la decision quand c’est utile.",
    "I": "Gagner a structurer vos messages, prioriser et conclure plus clairement.",
    "S": "Gagner a exprimer vos desaccords, poser des limites et oser dire non.",
    "C": "Gagner a simplifier, aller a l’essentiel et accepter une part d’incertitude.",
}

st.write(
    f"Vous avez un profil principalement **{DIM_LABELS[ordered[0][0]][0]} ({ordered[0][0]})**, "
    f"avec une energie secondaire **{DIM_LABELS[ordered[1][0]][0]} ({ordered[1][0]})**."
)

st.markdown(
    "Concretement, dans votre maniere naturelle d’agir et de communiquer, cela se traduit souvent ainsi :"
)

for dim in [ordered[0][0], ordered[1][0]]:
    st.markdown(f"- **{DIM_LABELS[dim][0]} ({dim})** : {DIM_NATURAL_STRENGTHS[dim]}")

st.subheader("Vos points forts naturels")

for dim in [ordered[0][0], ordered[1][0]]:
    st.markdown(f"- **{DIM_LABELS[dim][0]} ({dim})** : {DIM_NATURAL_STRENGTHS[dim]}")

st.subheader("Axes de réflexion pour progresser")

# On sépare clairement forces et énergies moins naturelles
strong_dims = [d for d, s in scores.items() if s >= 6]   # ≥ 6
weak_dims   = [d for d, s in scores.items() if s < 6]    # < 6

# Textes risques d’excès pour les énergies fortes
RISK_TEXT = {
    "D": "En excès, vous pouvez aller trop vite, imposer vos vues ou prendre peu de temps pour écouter.",
    "I": "En excès, vous pouvez beaucoup parler, vous disperser ou perdre de vue l’objectif.",
    "S": "En excès, vous pouvez éviter les conflits, trop vous adapter et avoir du mal à dire non.",
    "C": "En excès, vous pouvez être trop dans le détail, ralentir la décision ou manquer de flexibilité."
}

# Textes de développement pour les énergies moins naturelles
GROWTH_TEXT = {
    "D": "Développer davantage la Dominance (D) vous aiderait à prendre plus facilement des décisions, tenir vos positions et oser vous affirmer dans les moments clés.",
    "I": "Développer davantage l’Influence (I) vous aiderait à partager vos idées, créer plus de lien et embarquer plus facilement les autres.",
    "S": "Développer davantage la Stabilité (S) vous aiderait à mieux réfléchir aux conséquences de vos actions, prendre en compte l’ensemble des acteurs et installer un climat de confiance.",
    "C": "Développer davantage la Conformité (C) vous aiderait à structurer vos démarches, sécuriser les points de détail importants et fiabiliser vos décisions."
}

# 1. Utiliser vos forces sans tomber dans leurs excès
st.markdown("**1. Utiliser vos forces sans tomber dans leurs excès**")
for d in strong_dims:
    label = DIM_LABELS[d][0]  # ex. "Dominance"
    st.markdown(f"- **Énergie {label} ({d})** : {RISK_TEXT[d]}")

# 2. Développer davantage vos énergies moins naturelles
st.markdown("**2. Développer davantage vos énergies moins naturelles**")
if weak_dims:
    for d in weak_dims:
        st.markdown(f"- {GROWTH_TEXT[d]}")
else:
    st.markdown("Vous mobilisez déjà les 4 énergies de façon assez équilibrée. L’enjeu principal est surtout de doser vos forces en fonction des situations.")

# -------------------------------------------------------------------
# 5. Plan d'action – micro-comportements
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Pistes de plan d’action personnel")

st.markdown(
    """
L’objectif est de relier votre profil DISC a **des situations concretes**.

**1. Situation ou vous avez atteint un bon resultat**  
- Quelle etait la situation ?  
- Qu’avez-vous fait concretement ?  
- Quelles energies DISC avez-vous mobilisees (D, I, S, C) ?  
- Quels micro-comportements aimeriez-vous reutiliser plus souvent ?

**2. Situation plus difficile / moins satisfaisante**  
- Quelle etait la situation ?  
- Comment avez-vous reagi spontaneement ?  
- Quelle autre energie DISC auriez-vous pu activer ?  
- Quels micro-comportements pourriez-vous tester la prochaine fois ?
"""
)

col1, col2 = st.columns(2)

with col1:
    situation_success = st.text_area(
        "Plan d'action – Situation reussie",
        height=220,
        placeholder=(
            "Decrivez une situation ou vous avez obtenu un bon resultat.\n"
            "- Ce qui s'est passe\n"
            "- Ce que vous avez fait concretement\n"
            "- Les energies DISC mobilisees\n"
            "- Les micro-comportements a garder"
        ),
    )

with col2:
    situation_difficult = st.text_area(
        "Plan d'action – Situation difficile",
        height=220,
        placeholder=(
            "Decrivez une situation plus difficile.\n"
            "- Ce qui s'est passe\n"
            "- Votre reaction spontanee\n"
            "- L'energie DISC que vous pourriez activer autrement\n"
            "- Les micro-comportements a tester"
        ),
    )

# -------------------------------------------------------------------
# 6. Export PDF de synthese (avec radar)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Exporter ma synthese en PDF")

def sanitize(text: str) -> str:
    """Convertit tout texte en latin-1 compatible pour FPDF."""
    if text is None:
        return ""
    return text.encode("latin-1", "ignore").decode("latin-1")

def build_pdf(
    email: str,
    scores: dict,
    ordered_dims,
    situation_success: str,
    situation_difficult: str,
    radar_png: bytes | None = None,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize("Profil DISC - Synthese personnelle"), ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, sanitize(f"Email : {email}"), ln=True)
    pdf.ln(2)

    # Scores
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize("Scores detaillees :"), ln=True)
    pdf.set_font("Arial", "", 11)
    scores_line = (
        f"D : {scores['D']}, I : {scores['I']}, "
        f"S : {scores['S']}, C : {scores['C']}"
    )
    pdf.cell(0, 8, sanitize(scores_line), ln=True)
    pdf.ln(4)

    # Points forts
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize("Vos points forts naturels :"), ln=True)
    pdf.set_font("Arial", "", 11)
    for dim in [ordered_dims[0][0], ordered_dims[1][0]]:
        txt = f"- {DIM_LABELS[dim][0]} ({dim}) : {DIM_NATURAL_STRENGTHS[dim]}"
        pdf.multi_cell(0, 6, sanitize(txt))
    pdf.ln(2)

    # Axes de reflexion
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize("Axes de reflexion pour progresser :"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, sanitize("Utiliser vos forces sans tomber dans leurs exces :"))
    for dim in [ordered_dims[0][0], ordered_dims[1][0]]:
        txt = f"- {DIM_LABELS[dim][0]} ({dim}) : {DIM_EXCESS[dim]}"
        pdf.multi_cell(0, 6, sanitize(txt))
    pdf.ln(1)
    pdf.multi_cell(0, 6, sanitize("Developper davantage vos energies moins naturelles :"))
    for dim in ["D", "I", "S", "C"]:
        if dim not in [ordered_dims[0][0], ordered_dims[1][0]]:
            txt = f"- {DIM_LABELS[dim][0]} ({dim}) : {DIM_DEV[dim]}"
            pdf.multi_cell(0, 6, sanitize(txt))
    pdf.ln(2)

    # Page radar (si dispo)
    if radar_png:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, sanitize("Votre profil DISC (radar)"), ln=True)
        pdf.ln(4)

        # Enregistrer temporairement l'image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(radar_png)
            tmp_path = tmp.name

        # Inserer l'image (largeur ~160 mm, centree)
        pdf.image(tmp_path, x=25, y=None, w=160)
        # Nettoyage du fichier temporaire
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Page plan d'action
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize("Plan d'action - Situation reussie :"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, sanitize(situation_success or "(non renseigne)"))
    pdf.ln(1)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, sanitize("Plan d'action - Situation difficile :"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, sanitize(situation_difficult or "(non renseigne)"))

    # Generation finale
    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    return pdf_bytes

if st.button("Generer le PDF de ma synthese"):
    radar_png = st.session_state.get("radar_png")
    pdf_bytes = build_pdf(
        email=email,
        scores=scores,
        ordered_dims=ordered,
        situation_success=situation_success,
        situation_difficult=situation_difficult,
        radar_png=radar_png,
    )
    st.download_button(
        "⬇️ Telecharger le PDF",
        data=io.BytesIO(pdf_bytes),
        file_name="profil_disc_synthese.pdf",
        mime="application/pdf",
    )
