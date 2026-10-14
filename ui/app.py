import os
from collections.abc import Mapping
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("PULS_EVENTS_API_URL", os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("PULS_EVENTS_API_TIMEOUT", "30"))


def _response_text(payload: Any) -> str | None:
    """Extract a useful assistant message from common API response shapes."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("answer", "response", "message", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("data", "result", "output"):
            value = payload.get(key)
            extracted = _response_text(value)
            if extracted:
                return extracted
    if isinstance(payload, list):
        for item in payload:
            extracted = _response_text(item)
            if extracted:
                return extracted
    return None


# Titre de l'application
st.title("Puls-Events Chatbot 🎫🎷")
st.markdown("Posez-moi vos questions sur les événements culturels à venir !")

# Initialisation de l'historique du chat dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Champ de saisie pour l'utilisateur
if prompt := st.chat_input("Ex: Quels sont les concerts de jazz ce week-end ?"):
    # 1. Afficher le message de l'utilisateur dans l'interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Ajouter le message à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Interroger l'API FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": prompt},
                    timeout=API_TIMEOUT_SECONDS,
                )
                response.raise_for_status()

                try:
                    payload = response.json()
                except ValueError:
                    payload = None

                bot_reply = _response_text(payload)
                if not bot_reply:
                    bot_reply = "Désolé, l'API a renvoyé une réponse inattendue."

                st.markdown(bot_reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply}
                )
            except requests.exceptions.Timeout:
                st.error("L'API n'a pas répondu à temps. Réessayez dans un instant.")
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "inconnu"
                st.error(f"Erreur de l'API ({status_code}). Réessayez plus tard.")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Impossible de se connecter à l'API ({API_URL}). "
                    "Vérifiez que le serveur FastAPI est bien lancé."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Erreur lors de l'appel à l'API : {exc}")
