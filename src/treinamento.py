"""Treinamento dos modelos Isolation Forest para o FrioVale.

Treina um modelo separado para cada perfil de fruta:
- uva
- manga verde-madura
- manga parcialmente madura ou madura

O Isolation Forest eh nao supervisionado: usa apenas registros classificados
como normais para aprender o padrao. O rotulo_referencia eh usado SOMENTE para
avaliacao apos o treino.
"""

from __future__ import annotations

import os
import sys

# Garante que a pasta raiz do projeto esteja no caminho de importacao
# quando o script e executado diretamente (python src/treinamento.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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

# Contaminacao esperada (proporcao de anomalias na base). Ajustavel.
CONTAMINACAO: float = 0.1

# Nomes dos arquivos de modelo
ARQUIVO_MODELO_UVA: str = "modelo_uva.pkl"
ARQUIVO_MODELO_MANGA_VERDE: str = "modelo_manga_verde.pkl"
ARQUIVO_MODELO_MANGA_MADURA: str = "modelo_manga_madura.pkl"

MAPEAMENTO_ARQUIVOS: Dict[str, str] = {
    PERFIL_UVA: ARQUIVO_MODELO_UVA,
    PERFIL_MANGA_VERDE: ARQUIVO_MODELO_MANGA_VERDE,
    PERFIL_MANGA_MADURA: ARQUIVO_MODELO_MANGA_MADURA,
}


def treinar_isolation_forest(X: np.ndarray, contamination: float = CONTAMINACAO) -> IsolationForest:
    """Treina um Isolation Forest com os dados fornecidos.

    Parametros
    ----------
    X : np.ndarray
        Matriz de features (apenas registros normais recomendados).
    contamination : float
        Proporcao esperada de outliers.

    Retorna
    -------
    IsolationForest
        Modelo treinado.
    """
    modelo = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    modelo.fit(X)
    return modelo


def avaliar_modelo(
    modelo: IsolationForest, X: np.ndarray, y_true: np.ndarray
) -> Optional[dict]:
    """Avalia o modelo com os rotulos de referencia (apenas para relatorio).

    Parametros
    ----------
    modelo : IsolationForest
        Modelo ja treinado.
    X : np.ndarray
        Features de avaliacao.
    y_true : np.ndarray
        Rotulos verdadeiros ('normal' / 'anomalia').

    Retorna
    -------
    Optional[dict]
        Dicionario com precisao, recall, f1-score e matriz de confusao,
        ou None se nao houver rotulos.
    """
    if y_true.size == 0:
        return None

    # IsolationForest retorna 1 (inlier/normal) e -1 (outlier/anomalia)
    y_pred = modelo.predict(X)
    y_pred_labels = np.where(y_pred == 1, "normal", "anomalia")

    # Mapear para binario: anomalia = 1, normal = 0
    y_true_bin = (y_true == "anomalia").astype(int)
    y_pred_bin = (y_pred_labels == "anomalia").astype(int)

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    return {
        "precisao": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "matriz_confusao": cm.tolist(),
        "total_registros": int(len(y_true)),
        "anomalias_reais": int(y_true_bin.sum()),
        "anomalias_detectadas": int(y_pred_bin.sum()),
    }


def treinar_todos(
    caminho_csv: str, pasta_modelos: str = "modelos"
) -> Dict[str, dict]:
    """Executa o treinamento completo para todos os perfis.

    Parametros
    ----------
    caminho_csv : str
        Caminho para o CSV da base sintetica.
    pasta_modelos : str
        Pasta onde os arquivos .pkl serao salvos.

    Retorna
    -------
    Dict[str, dict]
        Resultado por perfil com metricas de avaliacao (quando disponiveis).
    """
    os.makedirs(pasta_modelos, exist_ok=True)

    df = carregar_dados(caminho_csv)
    valido, erro = validar_dados(df)
    if not valido:
        raise ValueError(f"Dados invalidos: {erro}")

    perfis = separar_por_perfil(df)
    resultados: Dict[str, dict] = {}

    for perfil, df_perfil in perfis.items():
        if df_perfil.empty:
            resultados[perfil] = {
                "status": "vazio",
                "mensagem": "Nenhum registro para este perfil.",
            }
            continue

        X, features = preparar_features(df_perfil)
        y_true = extrair_rotulos_referencia(df_perfil)

        # Treino usando apenas os dados do perfil (Isolation Forest eh nao
        # supervisionado; na usamos y_true para ajustar o modelo).
        modelo = treinar_isolation_forest(X)

        # Avaliacao opcional com rotulos de referencia (apenas para relatorio)
        avaliacao = avaliar_modelo(modelo, X, y_true)

        # Salvar modelo
        nome_arquivo = MAPEAMENTO_ARQUIVOS[perfil]
        caminho_modelo = os.path.join(pasta_modelos, nome_arquivo)
        joblib.dump(modelo, caminho_modelo)

        resultados[perfil] = {
            "status": "treinado",
            "registros": int(len(df_perfil)),
            "features": features,
            "modelo_salvo": caminho_modelo,
            "avaliacao": avaliacao,
        }

    return resultados


def executar_treinamento() -> None:
    """Ponto de entrada para treinar todos os modelos a partir do CSV padrao."""
    caminho_csv = os.path.join("dados", "base_sintetica_frio_vale.csv")
    if not os.path.exists(caminho_csv):
        print(f"ERRO: arquivo nao encontrado em {caminho_csv}")
        print("Coloque o arquivo base_sintetica_frio_vale.csv na pasta dados/.")
        return

    print("=== Iniciando treinamento dos modelos de IA - FrioVale ===")
    resultados = treinar_todos(caminho_csv)

    for perfil, info in resultados.items():
        print(f"\n--- Perfil: {perfil} ---")
        print(f"Status: {info['status']}")
        if info["status"] == "treinado":
            print(f"Registros: {info['registros']}")
            print(f"Features: {info['features']}")
            print(f"Modelo salvo: {info['modelo_salvo']}")
            aval = info.get("avaliacao")
            if aval:
                print(
                    f"Precisao: {aval['precisao']:.3f} | "
                    f"Recall: {aval['recall']:.3f} | "
                    f"F1: {aval['f1_score']:.3f}"
                )
                print(f"Matriz de confusao: {aval['matriz_confusao']}")
        else:
            print(f"Mensagem: {info['mensagem']}")

    print("\n=== Treinamento concluido ===")


if __name__ == "__main__":
    executar_treinamento()
