"""Testes da API FastAPI do modulo de IA.

Usam TestClient do FastAPI. Criam modelos rapidos em memoria para
garantir que os endpoints /health, /modelos e /prever funcionem.
"""

import numpy as np
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest
import joblib
import tempfile
import os

from src.api import app, preditor as preditor_global
from src.predicao import (
    PERFIL_MANGA_MADURA,
    PERFIL_MANGA_VERDE,
    PERFIL_UVA,
    PreditorIA,
    MAPEAMENTO_ARQUIVOS,
)


def _montar_preditor_com_modelos_fake() -> PreditorIA:
    """Treina modelos fake em tmp_path e retorna PreditorIA carregado."""
    tmpdir = tempfile.mkdtemp()

    # Mais amostras com variancia para o Isolation Forest aprender a fronteira
    X_uva = np.array([
        [11.0, 88.0], [11.5, 90.0], [12.0, 91.0], [12.5, 89.0],
        [11.8, 90.5], [12.2, 88.5], [11.2, 91.5], [12.8, 89.5],
        [13.0, 90.0], [14.0, 88.0], [15.0, 89.0],
    ])
    X_verde = np.array([
        [9.5, 84.0], [10.0, 86.0], [10.5, 85.0], [9.8, 87.0],
        [10.2, 84.5], [9.2, 85.5], [10.8, 86.5], [9.9, 84.8],
        [11.0, 85.0], [12.0, 86.0], [13.0, 85.5],
    ])
    X_madura = np.array([
        [13.5, 85.0], [14.0, 87.0], [14.5, 86.0], [13.8, 88.0],
        [14.2, 85.5], [13.2, 87.5], [14.8, 86.0], [13.9, 88.2],
        [15.0, 86.0], [16.0, 87.0], [17.0, 85.5],
    ])

    modelos = {
        PERFIL_UVA: IsolationForest(contamination=0.1, random_state=42).fit(X_uva),
        PERFIL_MANGA_VERDE: IsolationForest(contamination=0.1, random_state=42).fit(X_verde),
        PERFIL_MANGA_MADURA: IsolationForest(contamination=0.1, random_state=42).fit(X_madura),
    }

    for perfil, modelo in modelos.items():
        caminho = os.path.join(tmpdir, MAPEAMENTO_ARQUIVOS[perfil])
        joblib.dump(modelo, caminho)

    preditor = PreditorIA(pasta_modelos=tmpdir)
    preditor.carregar_modelos()
    return preditor


def test_health_sem_modelos(monkeypatch):
    """Quando nao ha preditor global, /health deve reportar erro."""
    monkeypatch.setattr("src.api.preditor", None)
    client = TestClient(app)
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "erro"
    assert corpo["modelos_carregados"] is False


def test_health_com_modelos(monkeypatch):
    preditor_fake = _montar_preditor_com_modelos_fake()
    monkeypatch.setattr("src.api.preditor", preditor_fake)

    client = TestClient(app)
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["modelos_carregados"] is True
    assert PERFIL_UVA in corpo["modelos"]


def test_listar_modelos(monkeypatch):
    preditor_fake = _montar_preditor_com_modelos_fake()
    monkeypatch.setattr("src.api.preditor", preditor_fake)

    client = TestClient(app)
    resposta = client.get("/modelos")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["total"] == 3
    assert PERFIL_UVA in corpo["modelos_disponiveis"]
    assert PERFIL_MANGA_VERDE in corpo["modelos_disponiveis"]
    assert PERFIL_MANGA_MADURA in corpo["modelos_disponiveis"]


def test_prever_normal(monkeypatch):
    preditor_fake = _montar_preditor_com_modelos_fake()
    monkeypatch.setattr("src.api.preditor", preditor_fake)

    client = TestClient(app)
    payload = {
        "tipo_fruta": "uva",
        "estagio_maturacao": "mesa",
        "temperatura_c": 12.0,
        "umidade_relativa_pct": 90.0,
    }
    resposta = client.post("/prever", json=payload)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert "anomalia" in corpo
    assert "resultado" in corpo
    assert "mensagem" in corpo
    assert corpo["tipo_fruta"] == "uva"


def test_prever_anomalia(monkeypatch):
    preditor_fake = _montar_preditor_com_modelos_fake()
    monkeypatch.setattr("src.api.preditor", preditor_fake)

    client = TestClient(app)
    payload = {
        "tipo_fruta": "uva",
        "estagio_maturacao": "mesa",
        "temperatura_c": 45.0,  # extrema para uva
        "umidade_relativa_pct": 90.0,
    }
    resposta = client.post("/prever", json=payload)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["anomalia"] is True


def test_prever_sem_modelos(monkeypatch):
    monkeypatch.setattr("src.api.preditor", None)
    client = TestClient(app)
    payload = {
        "tipo_fruta": "uva",
        "estagio_maturacao": "mesa",
        "temperatura_c": 12.0,
        "umidade_relativa_pct": 90.0,
    }
    resposta = client.post("/prever", json=payload)
    assert resposta.status_code == 503


def test_prever_dados_invalidos(monkeypatch):
    preditor_fake = _montar_preditor_com_modelos_fake()
    monkeypatch.setattr("src.api.preditor", preditor_fake)

    client = TestClient(app)
    # umidade fora do intervalo 0-100 -> Pydantic deve rejeitar (422)
    payload = {
        "tipo_fruta": "uva",
        "estagio_maturacao": "mesa",
        "temperatura_c": 12.0,
        "umidade_relativa_pct": 150.0,
    }
    resposta = client.post("/prever", json=payload)
    assert resposta.status_code == 422
