"""Testes do modulo de predicao.

Treinam rapidamente um Random Forest Classifier em memoria para validar o PreditorIA
sem depender de arquivos .pkl preexistentes.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.predicao import (
    PERFIL_MANGA_MADURA,
    PERFIL_MANGA_VERDE,
    PERFIL_UVA,
    PreditorIA,
)


def _treinar_modelo_fake(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    modelo = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
    modelo.fit(X, y)
    return modelo


@pytest.fixture
def preditor_com_modelos(tmp_path):
    """Cria um PreditorIA com modelos treinados em memoria e salvos em tmp_path."""
    # Dados para uva: 8 normais, 2 anomalias (10 registros)
    X_uva = np.array([
        [12.0, 90.0, 0.2, 1.0],
        [11.5, 91.0, 0.1, 1.1],
        [12.5, 89.0, 0.3, 0.9],
        [11.8, 90.5, 0.2, 1.0],
        [13.0, 88.0, 0.4, 1.2],
        [12.2, 89.5, 0.1, 0.8],
        [11.2, 91.5, 0.2, 1.1],
        [12.8, 89.0, 0.3, 1.0],
        [40.0, 90.0, 5.0, 0.5],
        [5.0, 90.0, -5.0, 0.5],
    ])
    y_uva = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    # Dados para manga verde: 8 normais, 2 anomalias
    X_verde = np.array([
        [10.0, 85.0, 0.2, 1.0],
        [9.5, 86.0, 0.1, 1.1],
        [10.5, 84.0, 0.3, 0.9],
        [9.8, 87.0, 0.2, 1.0],
        [11.0, 85.0, 0.4, 1.2],
        [10.2, 84.5, 0.1, 0.8],
        [9.2, 85.5, 0.2, 1.1],
        [10.8, 86.5, 0.3, 1.0],
        [20.0, 85.0, 5.0, 0.5],
        [2.0, 85.0, -5.0, 0.5],
    ])
    y_verde = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    # Dados para manga madura: 8 normais, 2 anomalias
    X_madura = np.array([
        [14.0, 87.0, 0.2, 1.0],
        [13.5, 86.0, 0.1, 1.1],
        [14.5, 86.0, 0.3, 0.9],
        [13.8, 88.0, 0.2, 1.0],
        [15.0, 86.0, 0.4, 1.2],
        [14.2, 85.5, 0.1, 0.8],
        [14.8, 86.0, 0.2, 1.1],
        [13.9, 88.2, 0.3, 1.0],
        [25.0, 87.0, 5.0, 0.5],
        [4.0, 87.0, -5.0, 0.5],
    ])
    y_madura = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    modelos = {
        PERFIL_UVA: _treinar_modelo_fake(X_uva, y_uva),
        PERFIL_MANGA_VERDE: _treinar_modelo_fake(X_verde, y_verde),
        PERFIL_MANGA_MADURA: _treinar_modelo_fake(X_madura, y_madura),
    }

    pasta = tmp_path / "modelos"
    pasta.mkdir()

    import joblib

    for perfil, modelo in modelos.items():
        from src.predicao import MAPEAMENTO_ARQUIVOS

        caminho = pasta / MAPEAMENTO_ARQUIVOS[perfil]
        joblib.dump(modelo, str(caminho))

    preditor = PreditorIA(pasta_modelos=str(pasta))
    preditor.carregar_modelos()
    return preditor


def test_carregar_modelos(preditor_com_modelos):
    assert preditor_com_modelos.carregados is True
    assert PERFIL_UVA in preditor_com_modelos.modelos
    assert PERFIL_MANGA_VERDE in preditor_com_modelos.modelos
    assert PERFIL_MANGA_MADURA in preditor_com_modelos.modelos


def test_identificar_perfil_uva(preditor_com_modelos):
    perfil = preditor_com_modelos.identificar_perfil("uva", "mesa")
    assert perfil == PERFIL_UVA


def test_identificar_perfil_manga_verde(preditor_com_modelos):
    perfil = preditor_com_modelos.identificar_perfil("manga", "verde")
    assert perfil == PERFIL_MANGA_VERDE


def test_identificar_perfil_manga_madura(preditor_com_modelos):
    perfil = preditor_com_modelos.identificar_perfil("manga", "madura")
    assert perfil == PERFIL_MANGA_MADURA


def test_prever_normal(preditor_com_modelos):
    resultado = preditor_com_modelos.prever(
        tipo_fruta="uva",
        estagio_maturacao="mesa",
        temperatura_c=12.0,
        umidade_relativa_pct=90.0,
        variacao_temperatura_c=0.2,
        variacao_umidade_pct=1.0,
    )
    assert resultado["resultado"] in ("normal", "anomalia")
    assert isinstance(resultado["anomalia"], bool)
    assert resultado["tipo_fruta"] == "uva"
    assert resultado["perfil"] == PERFIL_UVA


def test_prever_anomalia_extrema(preditor_com_modelos):
    # Temperatura muito fora do padrao aprendido para uva
    resultado = preditor_com_modelos.prever(
        tipo_fruta="uva",
        estagio_maturacao="mesa",
        temperatura_c=40.0,
        umidade_relativa_pct=90.0,
        variacao_temperatura_c=5.0,
        variacao_umidade_pct=0.5,
    )
    # Deve detectar anomalia com temperatura extrema
    assert resultado["anomalia"] is True
    assert resultado["resultado"] == "anomalia"


def test_prever_perfil_desconhecido(preditor_com_modelos):
    resultado = preditor_com_modelos.prever(
        tipo_fruta="abacaxi",
        estagio_maturacao="desconhecido",
        temperatura_c=12.0,
        umidade_relativa_pct=90.0,
        variacao_temperatura_c=0.2,
        variacao_umidade_pct=1.0,
    )
    assert resultado["resultado"] == "erro"
    assert resultado["anomalia"] is None


def test_prever_sem_modelos_carregados():
    preditor = PreditorIA(pasta_modelos="nao_existe")
    preditor.carregados = False
    preditor.modelos = {}

    with pytest.raises(RuntimeError):
        preditor.prever("uva", "mesa", 12.0, 90.0, 0.2, 1.0)


def test_valores_recebidos_estao_na_resposta(preditor_com_modelos):
    resultado = preditor_com_modelos.prever(
        tipo_fruta="uva",
        estagio_maturacao="mesa",
        temperatura_c=12.5,
        umidade_relativa_pct=89.0,
        variacao_temperatura_c=0.3,
        variacao_umidade_pct=0.9,
    )
    valores = resultado["valores_recebidos"]
    assert valores["temperatura_c"] == 12.5
    assert valores["umidade_relativa_pct"] == 89.0
    assert valores["variacao_temperatura_c"] == 0.3
    assert valores["variacao_umidade_pct"] == 0.9
