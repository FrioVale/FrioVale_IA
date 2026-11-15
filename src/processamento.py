"""Processamento de dados para o módulo de IA do FrioVale.

Responsabilidades:
- Carregar a base sintética CSV.
- Verificar qualidade e consistência dos dados.
- Separar registros por perfil de fruta e maturação.
- Preparar as variáveis (features) e os rótulos para treinamento e predição.

O rótulo de referência é utilizado como variável-alvo no aprendizado supervisionado.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings

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

# Features utilizadas no treinamento e na predição
FEATURES_BASE: List[str] = [
    "temperatura_c",
    "umidade_relativa_pct",
    "variacao_temperatura_c",
    "variacao_umidade_pct",
]

# Estágios de maturação reconhecidos para manga
ESTAGIOS_MANGA_VERDE: List[str] = [
    "verde",
    "verde-madura",
]

ESTAGIOS_MANGA_MADURA: List[str] = [
    "parcialmente madura",
    "madura",
]

# Perfis utilizados nos modelos
PERFIL_UVA: str = "uva"
PERFIL_MANGA_VERDE: str = "manga_verde"
PERFIL_MANGA_MADURA: str = "manga_madura"

# Rótulos válidos para aprendizado supervisionado
ROTULO_NORMAL: str = "normal"
ROTULO_ANOMALIA: str = "anomalia"

ROTULOS_VALIDOS: set = {ROTULO_NORMAL, ROTULO_ANOMALIA}


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega o arquivo CSV da base sintética."""
    return pd.read_csv(caminho_csv)


def _normalizar_texto(valor: object) -> str:
    """Normaliza textos para comparação (remove espaços e converte para minúsculo)."""
    if pd.isna(valor):
        return ""

    return str(valor).strip().lower()


def validar_dados(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Verifica a qualidade e a consistência básica dos dados.

    Verifica:
    - Presença das colunas obrigatórias.
    - Existência de registros.
    - Valores ausentes nas features e no rótulo.
    - Valores numéricos e finitos nas quatro features.
    - Validade dos rótulos de referência (apenas 'normal' e 'anomalia').
    - Presença de tipo de fruta e estágio de maturação.

    Não modifica o DataFrame original.

    Retorna:
        (True, None) se os dados forem válidos.
        (False, mensagem) caso exista algum problema.
    """

    if df.empty:
        return False, "A base de dados está vazia."

    # Verificar colunas obrigatórias
    colunas_faltantes = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS
        if coluna not in df.columns
    ]

    if colunas_faltantes:
        return False, (
            f"Colunas obrigatórias ausentes: {colunas_faltantes}"
        )

    # Trabalhar em uma cópia para não modificar o DataFrame original
    df_validacao = df.copy()

    # Verificar valores ausentes nas features
    ausentes = df_validacao[FEATURES_BASE].isnull().sum()
    colunas_com_ausentes = ausentes[ausentes > 0]

    if not colunas_com_ausentes.empty:
        return False, (
            "Valores ausentes encontrados nas features: "
            f"{colunas_com_ausentes.to_dict()}"
        )

    # Verificar valores ausentes no rótulo de referência
    if df_validacao["rotulo_referencia"].isnull().any():
        return False, "Existem registros sem rótulo de referência."

    # Verificar valores ausentes nos campos de identificação de perfil
    if df_validacao["tipo_fruta"].isnull().any():
        return False, "Existem registros sem identificação da fruta."

    if df_validacao["estagio_maturacao"].isnull().any():
        return False, "Existem registros sem estágio de maturação."

    # Converter features para valores numéricos
    for coluna in FEATURES_BASE:
        df_validacao[coluna] = pd.to_numeric(
            df_validacao[coluna],
            errors="coerce",
        )

        if df_validacao[coluna].isnull().any():
            return False, (
                f"A coluna '{coluna}' contém valores não numéricos."
            )

        # Verificar valores infinitos
        if not np.isfinite(df_validacao[coluna]).all():
            return False, (
                f"A coluna '{coluna}' contém valores infinitos."
            )

    # Normalizar e validar rótulos de referência
    rotulos_normalizados = (
        df_validacao["rotulo_referencia"]
        .apply(_normalizar_texto)
    )

    rotulos_encontrados = set(rotulos_normalizados)

    rotulos_invalidos = rotulos_encontrados - ROTULOS_VALIDOS

    if rotulos_invalidos:
        return False, (
            "Rótulos de referência inválidos: "
            f"{sorted(rotulos_invalidos)}. "
            "Utilize 'normal' ou 'anomalia'."
        )

    return True, None


def separar_por_perfil(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Separa os dados em três perfis de frutas.

    Perfis:
    - Uva.
    - Manga verde ou verde-madura.
    - Manga parcialmente madura ou madura.

    Registros com fruta ou estágio de maturação não reconhecidos
    não são classificados automaticamente e são ignorados com aviso.

    Não tenta inferir o estágio pela temperatura.
    """

    df = df.copy()

    df["tipo_fruta_norm"] = (
        df["tipo_fruta"].apply(_normalizar_texto)
    )

    df["estagio_norm"] = (
        df["estagio_maturacao"].apply(_normalizar_texto)
    )

    # Separar uvas
    mascara_uva = df["tipo_fruta_norm"] == "uva"
    df_uva = df[mascara_uva].copy()

    # Separar mangas
    mascara_manga = df["tipo_fruta_norm"] == "manga"
    df_manga = df[mascara_manga].copy()

    # Classificar mangas pelo estágio informado
    estagios_verde_lower = [e.lower() for e in ESTAGIOS_MANGA_VERDE]
    estagios_madura_lower = [e.lower() for e in ESTAGIOS_MANGA_MADURA]

    mascara_verde = df_manga["estagio_norm"].isin(estagios_verde_lower)
    mascara_madura = df_manga["estagio_norm"].isin(estagios_madura_lower)

    df_manga_verde = df_manga[mascara_verde].copy()
    df_manga_madura = df_manga[mascara_madura].copy()

    # Identificar mangas com estágio não reconhecido
    nao_mapeado = df_manga[
        ~mascara_verde & ~mascara_madura
    ]

    if not nao_mapeado.empty:
        estagios_desconhecidos = (
            nao_mapeado["estagio_maturacao"]
            .drop_duplicates()
            .tolist()
        )

        warnings.warn(
            "Existem mangas com estágio de maturação não reconhecido. "
            "Esses registros foram ignorados: "
            f"{estagios_desconhecidos}",
            stacklevel=2,
        )

    # Identificar frutas não contempladas pelos modelos
    frutas_desconhecidas = df[
        ~df["tipo_fruta_norm"].isin(["uva", "manga"])
    ]

    if not frutas_desconhecidas.empty:
        tipos_desconhecidos = (
            frutas_desconhecidas["tipo_fruta"]
            .drop_duplicates()
            .tolist()
        )

        warnings.warn(
            "Existem frutas sem perfil de modelo cadastrado. "
            "Esses registros foram ignorados: "
            f"{tipos_desconhecidos}",
            stacklevel=2,
        )

    return {
        PERFIL_UVA: df_uva,
        PERFIL_MANGA_VERDE: df_manga_verde,
        PERFIL_MANGA_MADURA: df_manga_madura,
    }


def preparar_features(
    df_perfil: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Prepara a matriz de features (X) para treinamento ou predição.

    Utiliza exatamente as quatro variáveis definidas em FEATURES_BASE,
    na mesma ordem, para garantir compatibilidade entre treinamento e predição.

    Não utiliza rótulos de referência como features.
    """

    if df_perfil.empty:
        raise ValueError(
            "Não existem registros para preparar as features."
        )

    colunas_ausentes = [
        coluna
        for coluna in FEATURES_BASE
        if coluna not in df_perfil.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"Features ausentes: {colunas_ausentes}"
        )

    X = df_perfil[FEATURES_BASE].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "As features contêm valores inválidos ou infinitos."
        )

    return X, FEATURES_BASE.copy()


def extrair_rotulos_referencia(
    df_perfil: pd.DataFrame,
) -> np.ndarray:
    """Extrai os rótulos de referência para uso como variável-alvo (y).

    Retorna:
        Array contendo 'normal' ou 'anomalia' (strings normalizadas).
    """

    if "rotulo_referencia" not in df_perfil.columns:
        return np.array([])

    return (
        df_perfil["rotulo_referencia"]
        .apply(_normalizar_texto)
        .to_numpy()
    )


def converter_rotulos_para_numericos(
    rotulos: np.ndarray,
) -> np.ndarray:
    """Converte rótulos textuais para classes numéricas.

    Mapeamento:
        'normal'  -> 0
        'anomalia' -> 1

    Retorna:
        Array de inteiros (0 ou 1).
    """

    mapeamento = {
        ROTULO_NORMAL: 0,
        ROTULO_ANOMALIA: 1,
    }

    return np.array(
        [mapeamento[_normalizar_texto(r)] for r in rotulos],
        dtype=int,
    )
