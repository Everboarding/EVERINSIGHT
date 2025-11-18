import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Accueil — EverINSIGHT DISC", page_icon="🧠")

st.title("EverINSIGHT — Diagnostic DISC")

st.markdown(
    """
Le modèle **DISC** décrit 4 grandes manières d’agir et de communiquer :

- **D – Dominance** : orienté résultats, aime décider et relever des défis.  
- **I – Influence** : sociable, enthousiaste, aime convaincre et inspirer.  
- **S – Stabilité** : à l’écoute, coopératif, recherche l’harmonie.  
- **C – Conformité** : structuré, rigoureux, orienté qualité et précision.

Ce questionnaire n’est **ni un test d’intelligence, ni un jugement**.  
Il sert à mieux comprendre votre **style naturel**, vos **points forts** et vos **axes de progression** pour travailler plus efficacement en équipe.
"""
)

st.markdown("---")

# ---------- Initialisation session_state ----------
for key in ["user_email", "user_firstname", "user_lastname", "disc_date"]:
    st.session_state.setdefault(key, "")

st.subheader("1. Vos informations")

with st.form("identite_disc"):
    col1, col2 = st.columns(2)
    with col1:
        firstname = st.text_input("Prénom", value=st.session_state.get("user_firstname", ""))
    with col2:
        lastname = st.text_input("Nom", value=st.session_state.get("user_lastname", ""))

    email = st.text_input(
        "Adresse e-mail (celle utilisée pour le cours)",
        value=st.session_state.get("user_email", ""),
        help="Elle sera utilisée pour associer votre profil DISC à vos résultats."
    )

    submitted = st.form_submit_button("Enregistrer mes informations")

if submitted:
    if not email.strip():
        st.error("Merci d’indiquer au minimum votre **adresse e-mail**.")
    else:
        st.session_state["user_firstname"] = firstname.strip()
        st.session_state["user_lastname"] = lastname.strip()
        st.session_state["user_email"] = email.strip().lower()
        # On ne met la date que si elle n’existe pas encore (1ère passation)
        if not st.session_state.get("disc_date"):
            st.session_state["disc_date"] = datetime.today().date().isoformat()

        st.success(
            "Vos informations ont été enregistrées. "
            "Vous pouvez maintenant passer au **Questionnaire DISC** via le menu à gauche."
        )

if st.session_state.get("user_email"):
    st.markdown("### 2. Récapitulatif de vos informations")

    nom_aff = (
        f"{st.session_state.get('user_firstname', '')} {st.session_state.get('user_lastname', '')}"
        ).strip()

    st.write(f"- **Nom / Prénom :** {nom_aff or '—'}")
    st.write(f"- **E-mail :** {st.session_state['user_email']}")

    if st.session_state.get("disc_date"):
        try:
            d = datetime.fromisoformat(st.session_state["disc_date"])
            st.write(f"- **Date d’enregistrement :** {d.strftime('%d/%m/%Y')}")
        except Exception:
            st.write(f"- **Date d’enregistrement :** {st.session_state['disc_date']}")

st.markdown("---")
st.subheader("3. Comment va se dérouler l’exercice ?")

st.markdown(
    """
1. Sur cette page **Accueil**, vous enregistrez vos informations.  
2. Dans l’onglet **Questionnaire DISC**, vous répondez aux 25 situations (choix forcé).  
3. Dans l’onglet **Mes résultats & plan d’action**, vous retrouvez votre profil, vos points forts et vos axes de réflexion.

Vous pourrez revenir sur vos résultats à tout moment pendant la séance avec le **même e-mail**.
"""
)

