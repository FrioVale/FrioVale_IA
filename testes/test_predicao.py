"""Testes do modulo de predicao.

Treinam rapidamente um Isolation Forest em memoria para validar o PreditorIA
sem depender de arquivos .pkl preexistentes.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.predicao import (
    PERFIL_MANGA_MADURA,
    PERFIL_MANGA_VERDE,
    PERFIL_UVA,
    PreditorIA,
)


def _treinar_modelo_fake(X: np.ndarray) -> IsolationForest:
    modelo = IsolationForest(contamination=0.1, random_state=42)
    modelo.fit(X)
    return modelo


@pytest.fixture
def preditor_com_modelos(tmp_path):
    """Cria um PreditorIA com modelos treinados em memoria e salvos em tmp_path."""
    # Dados para uva: faixa normal ~11-13C, com alguma variacao ate 15C
    X_uva = np.array([
        [11.0, 88.0], [11.5, 90.0], [12.0, 91.0], [12.5, 89.0],
        [11.8, 90.5], [12.2, 88.5], [11.2, 91.5], [12.8, 89.5],
        [13.0, 90.0], [14.0, 88.0], [15.0, 89.0],
    ])
    # Dados para manga verde: faixa normal ~9-11C, com alguma variacao ate 13C
    X_verde = np.array([
        [9.5, 84.0], [10.0, 86.0], [10.5, 85.0], [9.8, 87.0],
        [10.2, 84.5], [9.2, 85.5], [10.8, 86.5], [9.9, 84.8],
        [11.0, 85.0], [12.0, 86.0], [13.0, 85.5],
    ])
    # Dados para manga madura: faixa normal ~13-15C, com alguma variacao ate 17C
    X_madura = np.array([
        [13.5, 85.0], [14.0, 87.0], [14.5, 86.0], [13.8, 88.0],
        [14.2, 85.5], [13.2, 87.5], [14.8, 86.0], [13.9, 88.2],
        [15.0, 86.0], [16.0, 87.0], [17.0, 85.5],
    ])

    modelos = {
        PERFIL_UVA: _treinar_modelo_fake(X_uva),
        PERFIL_MANGA_VERDE: _treinar_modelo_fake(X_verde),
        PERFIL_MANGA_MADURA: _treinar_modelo_fake(X_madura),
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
    )
    assert resultado["resultado"] == "erro"
    assert resultado["anomalia"] is None


def test_prever_sem_modelos_carregados():
    preditor = PreditorIA(pasta_modelos="nao_existe")
    preditor.carregados = False
    preditor.modelos = {}

    with pytest.raises(RuntimeError):
        preditor.prever("uva", "mesa", 12.0, 90.0)


def test_valores_recebidos_estao_na_resposta(preditor_com_modelos):
    resultado = preditor_com_modelos.prever(
        tipo_fruta="uva",
        estagio_maturacao="mesa",
        temperatura_c=12.5,
        umidade_relativa_pct=89.0,
    )
    valores = resultado["valores_recebidos"]
    assert valores["temperatura_c"] == 12.5
    assert valores["umidade_relativa_pct"] == 89.0
