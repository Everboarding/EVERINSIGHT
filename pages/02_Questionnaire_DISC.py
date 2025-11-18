# pages/02_Questionnaire_DISC.py
# DISC — Autodiagnostic

import io
import json
import math
from datetime import datetime
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Questionnaire DISC", page_icon="🧭", layout="wide")
st.title("DISC — Autodiagnostic")

# ---------- Vérification des infos (email) ----------
if "user_email" not in st.session_state or not st.session_state["user_email"]:
    st.warning(
        "Merci de commencer par l’onglet **Accueil** pour renseigner votre prénom, nom et adresse e-mail."
    )
    st.stop()

email = st.session_state["user_email"].strip().lower()
display_name = st.session_state.get("user_firstname") or email.split("@")[0]

st.markdown(f"Vous répondez au questionnaire en tant que **{display_name}** ({email}).")

ROOT = Path(__file__).parent.parent  # dossier EVERINSIGHT

# Flag pour savoir si on a demandé une réinitialisation
st.session_state.setdefault("disc_reset", False)


def get_log_path() -> Path:
    path = ROOT / "data" / "logs" / "disc_forced_sessions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_last_session(user_email: str):
    """Retourne la dernière session pour cet e-mail (ou None)."""
    log_path = get_log_path()
    if not log_path.exists():
        return None

    last = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if (rec.get("user") or rec.get("email") or "").lower().strip() == user_email.lower().strip():
                last = rec
    return last


# ---------- Radar (spider chart) ----------
def render_radar(scores: dict, name: str):
    if not scores:
        return

    totals = {k: scores.get(k, 0) for k in ["D", "I", "S", "C"]}

    COLOR = {"D": "#E41E26", "I": "#FFC107", "S": "#2ECC71", "C": "#2E86DE"}
    ANGLES = {"D": 45, "I": 135, "S": 225, "C": 315}

    def scale_r(score, rmin=0.10, rmax=0.95, max_score=25):
        score = max(0, min(score, max_score))
        return rmin + (rmax - rmin) * (score / max_score)

    def pol2xy(angle_deg, r):
        a = math.radians(angle_deg)
        return (r * math.cos(a), r * math.sin(a))

    def xy2pol(x, y):
        r = math.hypot(x, y)
        a = (math.degrees(math.atan2(y, x)) + 360) % 360
        return a, r

    rD = scale_r(totals["D"])
    rI = scale_r(totals["I"])
    rS = scale_r(totals["S"])
    rC = scale_r(totals["C"])

    radar_pts = {
        "D": pol2xy(ANGLES["D"], rD),
        "I": pol2xy(ANGLES["I"], rI),
        "S": pol2xy(ANGLES["S"], rS),
        "C": pol2xy(ANGLES["C"], rC),
    }

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
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
        name,
        ha="center",
        va="top",
        fontsize=11,
        color="#333",
        zorder=7,
    )

    ax.text(math.radians(45), 1.03, "D", color=COLOR["D"], ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(math.radians(135), 1.03, "I", color=COLOR["I"], ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(math.radians(225), 1.03, "S", color=COLOR["S"], ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(math.radians(315), 1.03, "C", color=COLOR["C"], ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.set_rlim(0, 1.05)

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.pyplot(fig, clear_figure=True)

    st.caption(
        "Le point rouge correspond au milieu entre vos deux énergies les plus fortes. "
        "Le radar coloré représente votre profil global."
    )


# ---------------- ITEMS (25) EMBARQUÉS ----------------
ITEMS = [
    {"id": 1, "stem": "Mon collègue me présente un compte-rendu de notre dernier projet", "options": [
        {"label": "Je suis à son écoute, attentif", "dim": "S"},
        {"label": "Je lui fais comprendre que la décision finale m’appartient", "dim": "D"},
        {"label": "Je m’attache aux détails de son exposé", "dim": "C"},
        {"label": "Je le coupe souvent avec des anecdotes", "dim": "I"}]},
    {"id": 2, "stem": "Dans la vie de tous les jours", "options": [
        {"label": "J’aime relever des défis, il me faut de l’action", "dim": "D"},
        {"label": "Avec moi les gens ne s’ennuient jamais, j’aime divertir les autres", "dim": "I"},
        {"label": "Je suis plutôt compréhensif, je n’aime pas blesser les autres", "dim": "S"},
        {"label": "Je suis prudent, je ne donne pas ma confiance facilement", "dim": "C"}]},
    {"id": 3, "stem": "Dans une réunion", "options": [
        {"label": "Je suis à l’écoute des avis de chacun afin d’éviter les conflits", "dim": "S"},
        {"label": "Je suis coopératif, tant que tout le monde se conforme aux règles", "dim": "C"},
        {"label": "Mon avis est primordial, je ne lâche pas de terrain", "dim": "D"},
        {"label": "Je séduis les autres pour les convaincre de me suivre", "dim": "I"}]},
    {"id": 4, "stem": "Lors de la signature d’un contrat", "options": [
        {"label": "Je m’entoure de précautions et vérifie scrupuleusement tous les termes", "dim": "C"},
        {"label": "Je suis ferme et déterminé sur les clauses de l’accord", "dim": "D"},
        {"label": "J’influence mon interlocuteur pour le convaincre de faire un geste supplémentaire", "dim": "I"},
        {"label": "Je m’arrange pour que tout le monde soit satisfait quitte à faire une concession", "dim": "S"}]},
    {"id": 5, "stem": "Avec mes supérieurs", "options": [
        {"label": "Je suis facile à guider, j’obéis aux règles", "dim": "C"},
        {"label": "Je fais preuve d’audace", "dim": "D"},
        {"label": "Ils savent qu’ils peuvent vraiment compter sur moi", "dim": "S"},
        {"label": "Je suis aimable, je fais tout pour les charmer", "dim": "I"}]},
    {"id": 6, "stem": "Dans un travail de groupe", "options": [
        {"label": "Je suis plein de bonne volonté, la cohésion du groupe est importante", "dim": "S"},
        {"label": "J’ai la tête sur les épaules et j’impose mon point de vue", "dim": "D"},
        {"label": "Je me conforme aux règles et vérifie que toutes les normes sont respectées", "dim": "C"},
        {"label": "Je m’attache à ce que tout se passe dans la bonne humeur", "dim": "I"}]},
    {"id": 7, "stem": "Si je devais classer mes qualités, ce serait :", "options": [
        {"label": "La fiabilité, je suis méticuleux et ponctuel", "dim": "C"},
        {"label": "La détermination, je dois atteindre mes objectifs", "dim": "D"},
        {"label": "L’altruisme, j’aime rendre service", "dim": "S"},
        {"label": "La sociabilité, j’ai le contact facile", "dim": "I"}]},
    {"id": 8, "stem": "Dans les conversations, ce qui me caractérise le plus :", "options": [
        {"label": "J’aime quand les gens sont précis", "dim": "C"},
        {"label": "J’aime la convivialité, discuter de choses concrètes sans trop se prendre au sérieux", "dim": "I"},
        {"label": "J’écoute plus que je ne parle", "dim": "S"},
        {"label": "Je parle plus que je n’écoute", "dim": "D"}]},
    {"id": 9, "stem": "Je suis une personne :", "options": [
        {"label": "D’humeur égale, calme, difficilement irritable", "dim": "S"},
        {"label": "Joviale qui aime plaisanter", "dim": "I"},
        {"label": "Précise et exacte", "dim": "C"},
        {"label": "Fonceuse, audacieuse, qui déborde d’énergie", "dim": "D"}]},
    {"id": 10, "stem": "J’aime les gens :", "options": [
        {"label": "Disciplinés, qui savent se dominer", "dim": "C"},
        {"label": "Généreux, qui désirent partager", "dim": "I"},
        {"label": "Animés et sociables, qui s’expriment par gestes", "dim": "D"},
        {"label": "Persévérants, qui n’abandonnent pas et vont jusqu’au bout", "dim": "S"}]},
    {"id": 11, "stem": "Ce qui me caractérise le plus :", "options": [
        {"label": "J’ai l’esprit de compétition, je suis un battant", "dim": "D"},
        {"label": "Je suis expansif, sociable, j’ai confiance en moi", "dim": "I"},
        {"label": "Je suis attentionné et prévenant", "dim": "S"},
        {"label": "J’ai le goût de la perfection", "dim": "C"}]},
    {"id": 12, "stem": "Ce qui me reflète le mieux :", "options": [
        {"label": "J’aime les compliments, les éloges", "dim": "I"},
        {"label": "Je suis bienveillant, prêt à donner ou à aider", "dim": "S"},
        {"label": "Je suis formel et garde mes distances", "dim": "C"},
        {"label": "J’ai de la force de caractère", "dim": "D"}]},
    {"id": 13, "stem": "Les qualités que j’aime :", "options": [
        {"label": "L’empathie, comprendre les sentiments de l’autre", "dim": "S"},
        {"label": "La précision et la perfection", "dim": "C"},
        {"label": "La détermination et la force", "dim": "D"},
        {"label": "Le sens de l’humour, une certaine philosophie de la vie", "dim": "I"}]},
    {"id": 14, "stem": "Parmi les métiers proposés, je choisirais celui de :", "options": [
        {"label": "Infirmier, pour son dévouement aux autres", "dim": "S"},
        {"label": "Entrepreneur, pour son sens du challenge", "dim": "D"},
        {"label": "Comptable ou juriste, pour sa précision", "dim": "C"},
        {"label": "Journaliste ou écrivain, pour son côté investigateur", "dim": "I"}]},
    {"id": 15, "stem": "En règle générale, je suis plutôt :", "options": [
        {"label": "Respectueux des règles", "dim": "C"},
        {"label": "Entreprenant et aventurier", "dim": "D"},
        {"label": "Optimiste et positif", "dim": "I"},
        {"label": "Prêt à aider les autres et arrangeant", "dim": "S"}]},
    {"id": 16, "stem": "Les qualités qui me caractérisent le plus :", "options": [
        {"label": "Je suis courageux et fais preuve de bravoure", "dim": "D"},
        {"label": "Je sais stimuler les autres et les inspirer", "dim": "I"},
        {"label": "Je me conforme aux règles et aux lois", "dim": "C"},
        {"label": "Je suis paisible, j’aime le calme", "dim": "S"}]},
    {"id": 17, "stem": "Pour résoudre un problème avec mon équipe :", "options": [
        {"label": "Je m’adapte et fais preuve de flexibilité", "dim": "S"},
        {"label": "J’aime la confrontation, je sais ce qu’il faut faire", "dim": "D"},
        {"label": "Je suis décontracté, j’adore convaincre", "dim": "I"},
        {"label": "Je leur rappelle les règles à respecter pour surmonter la crise", "dim": "C"}]},
    {"id": 18, "stem": "C’est dimanche, j’ai prévu :", "options": [
        {"label": "D’organiser une petite fête avec les amis et voisins", "dim": "I"},
        {"label": "De faire ce que j’aime sans m’occuper des autres", "dim": "D"},
        {"label": "De m’occuper de ceux qui ont besoin d’aide", "dim": "S"},
        {"label": "De mettre de l’ordre dans mes papiers afin d’avoir l’esprit libre", "dim": "C"}]},
    {"id": 19, "stem": "Le plus souvent, vous êtes :", "options": [
        {"label": "Content de vous et satisfait de vos actions", "dim": "I"},
        {"label": "Confiant, vous avez foi dans les autres", "dim": "S"},
        {"label": "Attentif au travail bien fait", "dim": "C"},
        {"label": "Affirmatif, vous n’admettez pas le doute", "dim": "D"}]},
    {"id": 20, "stem": "Face à une nouvelle situation, vous êtes :", "options": [
        {"label": "Aventureux, vous aimez relever les défis", "dim": "D"},
        {"label": "Ouvert aux suggestions, réceptif aux idées des autres", "dim": "I"},
        {"label": "Chaleureux, vous allez connaître de nouvelles personnes", "dim": "S"},
        {"label": "Modéré, vous évitez les extrêmes et respectez les conventions", "dim": "C"}]},
    {"id": 21, "stem": "Ce que les autres apprécient chez vous :", "options": [
        {"label": "Votre calme et votre patience", "dim": "S"},
        {"label": "Votre goût du détail, vous êtes bien documenté", "dim": "C"},
        {"label": "Votre vigueur, vous êtes énergique", "dim": "D"},
        {"label": "Votre convivialité, vous aimez la compagnie", "dim": "I"}]},
    {"id": 22, "stem": "Dans les conversations, vous êtes plutôt :", "options": [
        {"label": "Loquace, vous aimez parler de sujets variés", "dim": "I"},
        {"label": "A l’écoute, vous savez vous contrôler", "dim": "S"},
        {"label": "A l’écoute, chaque mot a son importance", "dim": "C"},
        {"label": "Loquace, vous aimez diriger la conversation", "dim": "D"}]},
    {"id": 23, "stem": "Dans la vie, il faut se lever le matin, pour :", "options": [
        {"label": "Rechercher l’excellence… Faire mieux qu’hier !", "dim": "C"},
        {"label": "Créer de nouveaux contacts… Agrandir son cercle de relations", "dim": "I"},
        {"label": "Vivre un nouveau défi… Chaque jour est un nouveau challenge", "dim": "D"},
        {"label": "Travailler en équipe… Avancer ensemble et en paix", "dim": "S"}]},
    {"id": 24, "stem": "Pour réussir, il faut savoir :", "options": [
        {"label": "Être diplomate et avoir du tact", "dim": "S"},
        {"label": "Prendre des risques et être intrépide", "dim": "D"},
        {"label": "Être beau parleur et brillant en société", "dim": "I"},
        {"label": "Être réfléchi et analytique", "dim": "C"}]},
    {"id": 25, "stem": "On vous qualifie le plus souvent de :", "options": [
        {"label": "Hyperactif, vous ne tenez pas en place", "dim": "D"},
        {"label": "Populaire, vous êtes apprécié par la plupart", "dim": "I"},
        {"label": "Amical, vous êtes à l’écoute des autres", "dim": "S"},
        {"label": "Ordonné, vous êtes soigneux et organisé", "dim": "C"}]},
]

DIM_LABELS = {
    "D": ("Dominance", "Résultats / décision / vitesse"),
    "I": ("Influence", "Relation / énergie / inspiration"),
    "S": ("Stabilité", "Coopération / patience / fiabilité"),
    "C": ("Conformité", "Qualité / précision / normes"),
}

last_session = load_last_session(email)

# Si on n'a pas demandé de reset, on pré-remplit
preselected_choices = {}
if last_session and not st.session_state.get("disc_reset", False):
    for c in last_session.get("choices", []):
        preselected_choices[c["qid"]] = c["choice"]

# Bouton de réinitialisation (efface juste le pré-remplissage)
if st.button("Réinitialiser le questionnaire"):
    st.session_state["disc_reset"] = True
    st.rerun()

# ---------- FORM ----------
answers = {}
with st.form("disc_forced"):
    for it in ITEMS:
        st.markdown(f"**{it['id']}. {it['stem']}**")
        labels = [f"{i+1}. {opt['label']}" for i, opt in enumerate(it["options"])]

        default_index = None
        if it["id"] in preselected_choices:
            target = preselected_choices[it["id"]]
            for i, opt in enumerate(it["options"]):
                if opt["label"] == target:
                    default_index = i
                    break

        choice = st.radio(
            "",
            labels,
            key=f"q_{it['id']}",
            index=default_index,
        )
        answers[it["id"]] = choice
        st.divider()

    submitted = st.form_submit_button("Valider mes réponses")

# ---------- Si soumission : scoring + sauvegarde ----------
if submitted:
    st.session_state["disc_reset"] = False
    missing = [qid for qid, c in answers.items() if c is None]
    if missing:
        st.error("Vous n'avez pas répondu aux blocs : " + ", ".join(map(str, missing)))
        st.stop()

    totals = {"D": 0, "I": 0, "S": 0, "C": 0}
    picked = []
    for it in ITEMS:
        idx = int(answers[it["id"]].split(".")[0]) - 1
        dim = it["options"][idx]["dim"]
        label = it["options"][idx]["label"]
        totals[dim] += 1
        picked.append({"qid": it["id"], "choice": label, "dim": dim})

    ordered_dims = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top1_dim, top1_score = ordered_dims[0]
    top2_dim, top2_score = ordered_dims[1]
    style = top1_dim + top2_dim  # ex: "DI"

    st.success(
        f"Merci, vos réponses ont bien été enregistrées ✅ "
        f"(profil **{style}**). Vous pouvez consulter le débrief dans l’onglet **Mes résultats & plan d’action**."
    )

    # TABLEAU + BAR CHART
    df = pd.DataFrame(
        [
            {
                "Dimension": k,
                "Libellé": DIM_LABELS[k][0],
                "Score": v,
                "Description": DIM_LABELS[k][1],
            }
            for k, v in totals.items()
        ]
    ).sort_values("Score", ascending=False).reset_index(drop=True)
    st.subheader("Résultats")
    st.dataframe(df, use_container_width=True)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Libellé:N", sort="-y"),
        y="Score:Q",
        tooltip=["Libellé", "Score", "Description"],
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    # RADAR
    render_radar(totals, display_name)

    # EXPORT + LOG
    export = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user": email.lower(),
        "scores": totals,
        "style": style,
        "top_dims": [top1_dim, top2_dim],
        "choices": picked,
    }

    st.download_button(
        "⬇️ Export JSON",
        io.BytesIO(json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")),
        file_name="disc_result.json",
        mime="application/json",
    )

    log_path = get_log_path()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(export, ensure_ascii=False) + "\n")

else:
    # Pas de soumission → si on a déjà une session, on affiche les derniers résultats
    if last_session:
        st.subheader("Derniers résultats enregistrés")
        totals = last_session.get("scores", {}) or {}
        if totals:
            df = pd.DataFrame(
                [
                    {
                        "Dimension": k,
                        "Libellé": DIM_LABELS[k][0],
                        "Score": v,
                        "Description": DIM_LABELS[k][1],
                    }
                    for k, v in totals.items()
                ]
            ).sort_values("Score", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Libellé:N", sort="-y"),
                y="Score:Q",
                tooltip=["Libellé", "Score", "Description"],
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

            render_radar(totals, display_name)
            st.caption(
                "Ces résultats proviennent de votre dernière passation du questionnaire."
            )
    else:
        st.info(
            "Répondez aux 25 situations puis cliquez sur **Valider mes réponses** pour voir votre profil DISC."
        )
