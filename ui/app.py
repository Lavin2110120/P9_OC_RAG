"""UI Streamlit du chatbot Puls Events."""
import os
from collections.abc import Mapping
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
API_TIMEOUT_SECONDS = 60


def _response_text(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("response", "answer", "message", "text"):
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


st.title("Puls-Events Chatbot 🎫🎷")

API_URL = os.getenv("PULS_API_URL", DEFAULT_API_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question sur les événements dijonnais..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
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
            bot_reply = _response_text(payload) or "(réponse vide)"
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        except requests.exceptions.Timeout:
            st.error("⏱️ L'API a mis trop de temps à répondre.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Erreur HTTP : {exc}")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Impossible de joindre l'API. Vérifiez qu'elle tourne sur "
                     f"{API_URL}.")
