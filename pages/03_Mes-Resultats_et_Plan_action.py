from pathlib import Path
import json
import math
from datetime import datetime
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fpdf import FPDF
import smtplib
from email.message import EmailMessage


# =========================================================
# Utilitaires
# =========================================================

def clean_text(text: str) -> str:
    """Nettoie les caracteres non supportes par latin-1 (apostrophes typographiques, tirets, etc.)."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
    )


# =========================================================
# Chargement des donnees
# =========================================================

@st.cache_data
def load_whitelist():
    """Charge la liste des emails autorises (profils_etudiants.csv)."""
    base_dir = Path(__file__).parent.parent
    candidates = [
        base_dir / "data" / "profils_etudiants.csv",
        base_dir / "Data" / "profils_etudiants.csv",
    ]
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "email" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "email"})
    df["email"] = df["email"].astype(str).str.lower().str.strip()
    return df


@st.cache_data
def load_disc_sessions():
    """Charge les reponses DISC (disc_forced_sessions.jsonl)."""
    base_dir = Path(__file__).parent.parent
    candidates = [
        base_dir / "data" / "logs" / "disc_forced_sessions.jsonl",
        base_dir / "Data" / "logs" / "disc_forced_sessions.jsonl",
    ]
    log_path = None
    for p in candidates:
        if p.exists():
            log_path = p
            break
    if log_path is None:
        return []

    sessions = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            email = str(rec.get("user") or rec.get("email") or "").lower().strip()
            scores = rec.get("scores", {}) or {}
            style = rec.get("style") or ""
            ts = rec.get("ts", "")
            sessions.append(
                {
                    "email": email,
                    "profile": style,
                    "scores": scores,
                    "timestamp": ts,
                }
            )
    return sessions


# =========================================================
# Dictionnaires DISC
# =========================================================

DISC_DESCRIPTIONS = {
    "D": "Dominant : oriente resultats, direct, aime decider.",
    "I": "Influent : sociable, expressif, aime convaincre et inspirer.",
    "S": "Stable : pose, a l'ecoute, recherche l'harmonie et la cooperation.",
    "C": "Conforme : structure, rigoureux, aime la qualite et la precision.",
}

DIM_LABELS = {
    "D": "Dominance",
    "I": "Influence",
    "S": "Stabilite",
    "C": "Conformite",
}

DIM_STRENGTHS = {
    "D": "Vous etes a l'aise pour decider, trancher et faire avancer les sujets.",
    "I": "Vous savez creer du lien, mettre de l'energie et embarquer les autres.",
    "S": "Vous contribuez a stabiliser le groupe, ecouter et installer un climat serein.",
    "C": "Vous aimez structurer, verifier et garantir la qualite du travail.",
}

DIM_EXCES = {
    "D": "Poussee trop loin, cette energie peut vous rendre pressant, impatient, voire tranchant pour les autres.",
    "I": "En exces, vous pouvez beaucoup parler, vous disperser ou perdre de vue l'objectif.",
    "S": "En exces, vous pouvez eviter les conflits, trop vous adapter et avoir du mal a dire non.",
    "C": "En exces, vous pouvez etre tres exigeant, pointilleux et ralentir les decisions.",
}

DIM_DEV = {
    "D": "Vous pourriez gagner a affirmer davantage vos idees et a prendre plus d'initiatives dans les moments cles.",
    "I": "Vous pourriez gagner a partager plus vos idees, proposer des pistes et prendre plus souvent la parole.",
    "S": "Vous pourriez gagner a prendre le temps d'ecouter, soutenir vos collegues et stabiliser les situations tendues.",
    "C": "Vous pourriez gagner a structurer davantage vos demarches et a securiser les points de detail importants.",
}


# =========================================================
# Affichages: badges + radar
# =========================================================

def render_disc_badges(profile: str):
    """Affiche les badges DISC."""
    if not isinstance(profile, str) or profile.strip() == "":
        st.write("_Votre profil DISC n'a pas encore ete calcule._")
        return
    letters = [c.upper() for c in profile if c.upper() in ["D", "I", "S", "C"]]
    if not letters:
        st.write("_Votre profil DISC n'a pas encore ete calcule._")
        return

    cols = st.columns(len(letters))
    for col, letter in zip(cols, letters):
        with col:
            st.markdown(
                f"""
                <div style="
                    display:inline-block;
                    padding:4px 10px;
                    border-radius:999px;
                    font-weight:600;
                    font-size:0.9rem;
                    text-align:center;
                    border:1px solid #ddd;
                ">
                    {letter}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(" ")
    for letter in letters:
        desc = DISC_DESCRIPTIONS.get(letter, "")
        if desc:
            st.markdown(f"- **{letter}** – {desc}")


def create_radar_figure(scores: dict, name: str = "vous"):
    """Cree le radar DISC et renvoie la figure (sans l'afficher)."""
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

    return fig


def render_radar(scores: dict, name: str = "vous"):
    """Affiche le radar dans Streamlit."""
    if not scores:
        return
    fig = create_radar_figure(scores, name)
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.pyplot(fig, clear_figure=True)
    st.caption(
        "Le point rouge correspond au milieu entre vos deux energies les plus fortes. "
        "Le radar colore represente votre profil global."
    )


# =========================================================
# PDF
# =========================================================

def build_disc_pdf(
    email: str,
    profile: str,
    scores: dict,
    strengths_text: str,
    axes_text: str,
    success_notes: str,
    difficult_notes: str,
) -> bytes:
    """Construit un PDF simple en latin-1 (sans caracteres exotiques), avec une page radar."""

    email = clean_text(email)
    profile = clean_text(profile)
    strengths_text = clean_text(strengths_text)
    axes_text = clean_text(axes_text)
    success_notes = clean_text(success_notes)
    difficult_notes = clean_text(difficult_notes)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -------- Page texte --------
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, clean_text("Profil DISC - Synthese personnelle"), ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Email : {email}", ln=True)
    if profile:
        pdf.cell(0, 8, f"Profil DISC : {profile}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Scores detailles :", ln=True)
    pdf.set_font("Arial", "", 11)
    scores_line = ", ".join(f"{k} : {v}" for k, v in scores.items())
    scores_line = clean_text(scores_line)
    pdf.multi_cell(0, 6, scores_line)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Vos points forts naturels :", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, strengths_text)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Axes de reflexion pour progresser :", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, axes_text)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Plan d'action - Situation reussie :", ln=True)
    pdf.set_font("Arial", "", 11)
    if success_notes.strip():
        pdf.multi_cell(0, 6, success_notes)
    else:
        pdf.multi_cell(0, 6, "(a completer)")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Plan d'action - Situation difficile :", ln=True)
    pdf.set_font("Arial", "", 11)
    if difficult_notes.strip():
        pdf.multi_cell(0, 6, difficult_notes)
    else:
        pdf.multi_cell(0, 6, "(a completer)")

    # -------- Page radar --------
    if scores:
        fig = create_radar_figure(scores, "vous")
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp_file = tmp.name
                fig.savefig(tmp_file, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)

            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Radar DISC", ln=True)
            pdf.ln(5)
            # centrer approximativement le radar
            pdf.image(tmp_file, x=25, y=30, w=160)
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    # encode latin-1 en ignorant les caracteres restants eventuels
    return pdf.output(dest="S").encode("latin-1", "ignore")


def has_smtp_config() -> bool:
    """Verifie que les secrets SMTP sont bien renseignes."""
    required = ["SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"]
    try:
        for key in required:
            if key not in st.secrets:
                return False
        return True
    except Exception:
        return False


def send_pdf_via_email(to_email: str, pdf_bytes: bytes):
    """Envoie le PDF par mail via SMTP (si secrets configures)."""
    msg = EmailMessage()
    msg["Subject"] = "Votre profil DISC - Synthese"
    msg["From"] = st.secrets["SMTP_FROM"]
    msg["To"] = to_email
    msg.set_content(
        "Bonjour,\n\nVous trouverez en piece jointe votre profil DISC et votre plan d'action personnel.\n\n"
        "A bientot."
    )
    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="profil_DISC.pdf",
    )
    with smtplib.SMTP_SSL(
        st.secrets["SMTP_SERVER"], st.secrets["SMTP_PORT"]
    ) as server:
        server.login(st.secrets["SMTP_USER"], st.secrets["SMTP_PASSWORD"])
        server.send_message(msg)


# =========================================================
# Page principale
# =========================================================

def main():
    st.set_page_config(page_title="Mes resultats & plan d'action", page_icon="🧠")

    st.title("Mes resultats & plan d'action")

    if "user_email" not in st.session_state or not st.session_state["user_email"]:
        st.warning(
            "Pour consulter vos resultats, commencez par renseigner vos informations dans l'onglet **Accueil**."
        )
        return

    email = st.session_state["user_email"].lower().strip()

    sessions = load_disc_sessions()
    whitelist_df = load_whitelist()

    if whitelist_df is not None and email not in whitelist_df["email"].values:
        st.error(
            "Cet email n'est pas reconnu dans la liste des participants. "
            "Verifiez l'adresse ou contactez votre intervenant."
        )
        return

    matches = [s for s in sessions if s["email"] == email]
    if not matches:
        st.error(
            "Aucun resultat DISC trouve pour cet email. "
            "Vous n'avez peut-etre pas encore complete le questionnaire dans l'onglet **Questionnaire DISC**."
        )
        return

    data = matches[-1]
    profile = data.get("profile", "")
    scores = data.get("scores", {})
    ts_raw = data.get("timestamp", "")

    st.success(f"Heureux de vous revoir, {email} !")

    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw.replace("Z", ""))
            st.caption(f"Derniere passation du questionnaire : {dt.strftime('%d/%m/%Y a %H:%M')}")
        except Exception:
            st.caption(f"Derniere passation du questionnaire : {ts_raw}")

    # -------- Profil & radar --------
    st.header("Votre profil DISC")
    render_disc_badges(profile)

    if not scores:
        st.info("Les scores detailles ne sont pas disponibles.")
        return

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    letters_order = [k for k, _ in ordered]
    main_dim = letters_order[0]
    secondary_dim = letters_order[1] if len(letters_order) > 1 else None

    render_radar(scores, "vous")

    st.subheader("Lecture de votre profil")
    if secondary_dim:
        st.markdown(
            f"Vous avez un profil principalement **{DIM_LABELS[main_dim]} ({main_dim})**, "
            f"avec une energie secondaire **{DIM_LABELS[secondary_dim]} ({secondary_dim})**."
        )
    else:
        st.markdown(
            f"Vous avez un profil principalement **{DIM_LABELS[main_dim]} ({main_dim})**."
        )
    st.markdown(
        "En pratique, cela signifie que vous avez tendance a vous appuyer d'abord sur ces energies "
        "pour interagir avec les autres : travailler en groupe, decider, gerer un desaccord, "
        "prendre la parole, etc."
    )
    st.markdown(
        "La section suivante detaille plus precisement vos **points forts naturels** associes a ces energies."
    )

    strengths_letters = letters_order[:2]
    axes_letters = letters_order[-2:]

    # -------- Points forts --------
    st.subheader("Vos points forts naturels")
    for letter in strengths_letters:
        st.markdown(
            f"- **{DIM_LABELS[letter]} ({letter})** : {DIM_STRENGTHS[letter]}"
        )

    # -------- Axes de progression --------
    st.subheader("Axes de reflexion pour progresser")

    st.markdown("**1. Utiliser vos forces sans tomber dans leurs exces**")
    for letter in strengths_letters:
        st.markdown(
            f"- **Energie {DIM_LABELS[letter]} ({letter})** : {DIM_EXCES[letter]}"
        )

    st.markdown("**2. Developper davantage vos energies moins naturelles**")
    for letter in reversed(axes_letters):
        st.markdown(
            f"- **Energie {DIM_LABELS[letter]} ({letter})** : {DIM_DEV[letter]}"
        )

    # -------- Plan d'action / micro-comportements --------
    st.subheader("Plan d'action : travailler vos micro-comportements")

    st.markdown(
        """
L'objectif n'est pas de changer de personnalite, mais de travailler sur de petits comportements concrets dans des situations reelles.

Choisissez deux situations a analyser :
- une **situation reussie**, ou vous avez obtenu un bon resultat ;
- une **situation difficile**, ou vous avez ete en tension ou insatisfait du resultat.
"""
    )

    st.markdown("### 1. Situation ou vous avez obtenu un bon resultat")
    st.markdown(
        """
Reflechissez a cette situation en lien avec votre profil DISC :

- Quelle etait la situation precise ?  
- Quels comportements avez-vous adoptes (ce que vous avez dit, fait, demande, prepare...) ?  
- Quelles energies DISC avez-vous principalement mobilisees (D, I, S, C) ?  
- Quels micro-comportements aimeriez-vous reproduire plus souvent dans d'autres situations ?
"""
    )
    success_notes = st.text_area(
        "Notez ici votre reflexion (pour vous, ces notes ne sont pas sauvegardees sur un serveur) :",
        key="disc_action_success",
        height=180,
    )

    st.markdown("### 2. Situation ou vous avez rencontre des difficultees")
    st.markdown(
        """
Analysez maintenant une situation plus compliquee :

- Qu'est-ce qui a ete difficile concretement ?  
- Quelle energie DISC etait tres presente dans votre reaction ?  
- Quelle autre energie auriez-vous pu activer (par exemple plus de C pour structurer, plus de D pour decider, etc.) ?  
- Quels micro-comportements alternatifs pourriez-vous tester la prochaine fois ?
"""
    )
    difficult_notes = st.text_area(
        "Notez ici vos pistes d'ajustement :",
        key="disc_action_difficult",
        height=180,
    )

    st.info(
        "Vos notes restent visibles tant que cette page est ouverte dans votre navigateur. "
        "Elles servent aussi a generer votre PDF, mais ne sont pas stockees dans une base de donnees."
    )

    # -------- Texte pour le PDF --------
    strengths_text = "\n".join(
        f"- {DIM_LABELS[letter]} ({letter}) : {DIM_STRENGTHS[letter]}"
        for letter in strengths_letters
    )

    axes_text_parts = []
    axes_text_parts.append("Utiliser vos forces sans tomber dans leurs exces :")
    for letter in strengths_letters:
        axes_text_parts.append(f"- {DIM_LABELS[letter]} ({letter}) : {DIM_EXCES[letter]}")
    axes_text_parts.append("")
    axes_text_parts.append("Developper davantage vos energies moins naturelles :")
    for letter in reversed(axes_letters):
        axes_text_parts.append(f"- {DIM_LABELS[letter]} ({letter}) : {DIM_DEV[letter]}")
    axes_text = "\n".join(axes_text_parts)

    pdf_bytes = build_disc_pdf(
        email=email,
        profile=profile,
        scores=scores,
        strengths_text=strengths_text,
        axes_text=axes_text,
        success_notes=success_notes,
        difficult_notes=difficult_notes,
    )

    # -------- Export PDF --------
    st.markdown("### Export de votre profil")

    st.download_button(
        "📄 Telecharger mon profil et plan d'action en PDF",
        data=pdf_bytes,
        file_name="profil_DISC.pdf",
        mime="application/pdf",
    )

    if st.button("✉️ M'envoyer ce PDF par e-mail"):
        if not has_smtp_config():
            st.error(
                "Configuration SMTP manquante dans `secrets.toml` "
                "(champs SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM)."
            )
        else:
            try:
                send_pdf_via_email(email, pdf_bytes)
                st.success("Le PDF vous a ete envoye par e-mail.")
            except Exception as e:
                st.error(f"Erreur lors de l'envoi du mail : {e}")

    st.markdown("**Scores detailles :**")
    st.write(", ".join(f"{k} : {v}" for k, v in scores.items()))

    st.markdown("---")
    st.markdown(
        "Ce debrief s'appuie sur vos reponses au questionnaire DISC (version choix force). "
        "Il ne decrit pas toute votre personnalite, mais une tendance dominante dans votre facon d'agir et de communiquer."
    )


if __name__ == "__main__":
    main()
