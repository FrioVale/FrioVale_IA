"""Testes do modulo de processamento de dados.

Nao requerem arquivo CSV externo: criam DataFrames sinteticos em memoria.
"""

import pandas as pd
import pytest

from src.processamento import (
    FEATURES_BASE,
    PERFIL_MANGA_MADURA,
    PERFIL_MANGA_VERDE,
    PERFIL_UVA,
    carregar_dados,
    extrair_rotulos_referencia,
    preparar_features,
    separar_por_perfil,
    validar_dados,
)


def _df_sintetico() -> pd.DataFrame:
    """Cria um DataFrame sintetico valido para testes."""
    dados = {
        "timestamp": [
            "2026-01-01 08:00:00",
            "2026-01-01 09:00:00",
            "2026-01-01 10:00:00",
            "2026-01-01 11:00:00",
            "2026-01-01 12:00:00",
            "2026-01-01 13:00:00",
        ],
        "tipo_fruta": ["uva", "uva", "manga", "manga", "manga", "manga"],
        "estagio_maturacao": [
            "mesa",
            "mesa",
            "verde",
            "verde-madura",
            "parcialmente madura",
            "madura",
        ],
        "temperatura_c": [12.0, 11.5, 10.0, 9.5, 14.0, 13.5],
        "umidade_relativa_pct": [90.0, 91.0, 85.0, 86.0, 88.0, 87.0],
        "variacao_temperatura_c": [0.2, 0.1, 0.3, 0.2, 0.4, 0.3],
        "variacao_umidade_pct": [1.0, 1.1, 0.8, 0.7, 0.9, 0.8],
        "rotulo_referencia": ["normal", "normal", "normal", "normal", "anomalia", "normal"],
        "cenario_simulado": [
            "transporte normal",
            "transporte normal",
            "transporte normal",
            "transporte normal",
            "temperatura alta",
            "transporte normal",
        ],
    }
    return pd.DataFrame(dados)


def test_validar_dados_valido():
    df = _df_sintetico()
    valido, erro = validar_dados(df)
    assert valido is True
    assert erro is None


def test_validar_dados_coluna_faltante():
    df = _df_sintetico().drop(columns=["temperatura_c"])
    valido, erro = validar_dados(df)
    assert valido is False
    assert "temperatura_c" in erro


def test_separar_por_perfil():
    df = _df_sintetico()
    valido, _ = validar_dados(df)
    assert valido is True

    perfis = separar_por_perfil(df)

    assert PERFIL_UVA in perfis
    assert PERFIL_MANGA_VERDE in perfis
    assert PERFIL_MANGA_MADURA in perfis

    assert len(perfis[PERFIL_UVA]) == 2
    assert len(perfis[PERFIL_MANGA_VERDE]) == 2
    assert len(perfis[PERFIL_MANGA_MADURA]) == 2


def test_preparar_features():
    df = _df_sintetico()
    perfis = separar_por_perfil(df)
    df_uva = perfis[PERFIL_UVA]

    X, features = preparar_features(df_uva)

    assert features == FEATURES_BASE
    assert X.shape[1] == 2
    assert X.shape[0] == len(df_uva)


def test_extrair_rotulos_referencia():
    df = _df_sintetico()
    perfis = separar_por_perfil(df)
    df_uva = perfis[PERFIL_UVA]

    rotulos = extrair_rotulos_referencia(df_uva)
    assert len(rotulos) == len(df_uva)
    assert set(rotulos).issubset({"normal", "anomalia"})


def test_carregar_dados_arquivo_inexistente(tmp_path):
    caminho = tmp_path / "nao_existe.csv"
    with pytest.raises(FileNotFoundError):
        carregar_dados(str(caminho))
