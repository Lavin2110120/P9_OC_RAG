"""Tests unitaires du module RAG sans appel Mistral ni chargement de modèle."""
from unittest.mock import Mock

import pytest


def test_format_docs_joint_le_contenu_des_documents():
    from api.mistral_qa import _format_docs
    from langchain_core.documents import Document

    assert _format_docs([Document(page_content="A"), Document(page_content="B")]) == "A\n\nB"
    assert _format_docs([]) == ""


def test_main_affiche_la_reponse(monkeypatch, capsys):
    import api.mistral_qa as qa
    generate = Mock(return_value="Réponse déterministe")
    monkeypatch.setattr(qa, "generate_response", generate)

    qa.main("Question ?")

    generate.assert_called_once_with("Question ?", k=5)
    assert capsys.readouterr().out.strip() == "Réponse déterministe"
