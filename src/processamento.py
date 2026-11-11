"""Processamento de dados para o módulo de IA do FrioVale.

Responsabilidades:
- Carregar a base sintética CSV.
- Verificar qualidade (valores ausentes, inconsistências).
- Separar registros por perfil de fruta e maturação.
- Preparar as variáveis (features) para treinamento e predição.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Colunas obrigatórias que o CSV deve conter
COLUNAS_OBRIGATORIAS: List[str] = [
    "timestamp",
    "tipo_fruta",
    "estagio_maturacao",
    "temperatura_c",
    "umidade_relativa_pct",
    "variacao_temperatura_c",
    "variacao_umidade_pct",
    "rotulo_referencia",
    "cenario_simulado",
]

# Colunas usadas como features de treino/predicao (Início com apenas as
# confiaveis, conforme solicitado: temperatura e umidade).
FEATURES_BASE: List[str] = ["temperatura_c", "umidade_relativa_pct"]

# Mapeamento de estagios de maturacao para os perfis de manga.
# Os valores sao normalizados para minusculo na validacao.
ESTAGIOS_MANGA_VERDE: List[str] = ["verde", "verde-madura"]
ESTAGIOS_MANGA_MADURA: List[str] = ["parcialmente madura", "madura", "madura "]

# Nome dos perfis utilizados nos modelos
PERFIL_UVA: str = "uva"
PERFIL_MANGA_VERDE: str = "manga_verde"
PERFIL_MANGA_MADURA: str = "manga_madura"


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega o arquivo CSV da base sintética.

    Parametros
    ----------
    caminho_csv : str
        Caminho para o arquivo base_sintetica_frio_vale.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados brutos.
    """
    df = pd.read_csv(caminho_csv)
    return df


def validar_dados(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Verifica qualidade basica dos dados.

    Verifica:
    - Presenca das colunas obrigatorias.
    - Valores ausentes nas colunas de feature.
    - Tipos numericos nas colunas de temperatura e umidade.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame carregado do CSV.

    Retorna
    -------
    Tuple[bool, Optional[str]]
        (True, None) se valido, ou (False, mensagem_erro) se invalido.
    """
    colunas_faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if colunas_faltantes:
        return False, f"Colunas obrigatorias ausentes: {colunas_faltantes}"

    # Verificar valores ausentes nas features base
    ausentes = df[FEATURES_BASE].isnull().sum()
    colunas_com_ausentes = ausentes[ausentes > 0]
    if not colunas_com_ausentes.empty:
        return False, (
            f"Valores ausentes encontrados nas features: "
            f"{colunas_com_ausentes.to_dict()}"
        )

    # Garantir que temperatura e umidade sejam numericas
    for col in FEATURES_BASE:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as exc:  # noqa: BLE001
                return False, f"Coluna '{col}' nao e numerica: {exc}"
            if df[col].isnull().any():
                return False, f"Coluna '{col}' contem valores nao numericos apos conversao."

    return True, None


def _normalizar_texto(valor: str) -> str:
    """Normaliza texto para comparacao de estagio de maturacao."""
    if pd.isna(valor):
        return ""
    return str(valor).strip().lower()


def separar_por_perfil(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Separa os dados em tres perfis: uva, manga verde e manga madura.

    Nao mistura perfis indiscriminadamente. Cada perfil gera seu proprio modelo.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame ja validado.

    Retorna
    -------
    Dict[str, pd.DataFrame]
        Chaves: 'uva', 'manga_verde', 'manga_madura'.
        Valores: DataFrames filtrados (podem estar vazios se nao houver registros).
    """
    df = df.copy()
    df["tipo_fruta_norm"] = df["tipo_fruta"].apply(_normalizar_texto)
    df["estagio_norm"] = df["estagio_maturacao"].apply(_normalizar_texto)

    mascara_uva = df["tipo_fruta_norm"] == "uva"
    mascara_manga = df["tipo_fruta_norm"] == "manga"

    df_uva = df[mascara_uva].copy()

    df_manga = df[mascara_manga].copy()
    mascara_verde = df_manga["estagio_norm"].isin(
        [e.lower() for e in ESTAGIOS_MANGA_VERDE]
    )
    mascara_madura = df_manga["estagio_norm"].isin(
        [e.lower() for e in ESTAGIOS_MANGA_MADURA]
    )

    df_manga_verde = df_manga[mascara_verde].copy()
    df_manga_madura = df_manga[mascara_madura].copy()

    # Caso exista manga com estagio nao mapeado, atribuimos ao mais proximo
    # com base na faixa de temperatura (heuristica simples de seguranca).
    nao_mapeado = df_manga[~mascara_verde & ~mascara_madura]
    if not nao_mapeado.empty:
        # Tentar inferir pela temperatura media: manga verde costuma ser mais fria
        media_temp = nao_mapeado["temperatura_c"].mean()
        if media_temp <= 12:
            df_manga_verde = pd.concat([df_manga_verde, nao_mapeado], ignore_index=True)
        else:
            df_manga_madura = pd.concat([df_manga_madura, nao_mapeado], ignore_index=True)

    return {
        PERFIL_UVA: df_uva,
        PERFIL_MANGA_VERDE: df_manga_verde,
        PERFIL_MANGA_MADURA: df_manga_madura,
    }


def preparar_features(df_perfil: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Prepara a matriz de features (X) a partir de um DataFrame de perfil.

    Parametros
    ----------
    df_perfil : pd.DataFrame
        DataFrame de um perfil especifico (uva, manga_verde ou manga_madura).

    Retorna
    -------
    Tuple[np.ndarray, List[str]]
        Matriz numerica X e lista com os nomes das features utilizadas.
    """
    X = df_perfil[FEATURES_BASE].to_numpy(dtype=float)
    return X, FEATURES_BASE


def extrair_rotulos_referencia(df_perfil: pd.DataFrame) -> np.ndarray:
    """Extrai o rotulo de referencia (apenas para avaliacao, nunca para treino).

    Parametros
    ----------
    df_perfil : pd.DataFrame

    Retorna
    -------
    np.ndarray
        Array com os rotulos ('normal' ou 'anomalia').
    """
    if "rotulo_referencia" not in df_perfil.columns:
        return np.array([])
    return df_perfil["rotulo_referencia"].to_numpy()
