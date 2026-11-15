"""Treinamento dos modelos Random Forest Classifier para o FrioVale (versão 2).

Treina um modelo separado para cada perfil:
- Uva
- Manga verde ou verde-madura
- Manga parcialmente madura ou madura

O Random Forest Classifier é um algoritmo de aprendizado supervisionado:
- Utiliza rotulo_referencia como variável-alvo (y).
- normal = 0, anomalia = 1.
- Realiza divisão entre treino e teste com stratify quando possível.
- Avalia o modelo exclusivamente com dados de teste (não utilizados no treino).

Os modelos treinados são salvos na pasta modelos_v2/, mantendo os modelos
versão 1 intactos.
"""

from __future__ import annotations

import os
import sys

# Garante a importação dos módulos ao executar diretamente
# python src/treinamento.py
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.processamento import (
    PERFIL_UVA,
    PERFIL_MANGA_VERDE,
    PERFIL_MANGA_MADURA,
    ROTULO_ANOMALIA,
    ROTULO_NORMAL,
    carregar_dados,
    converter_rotulos_para_numericos,
    extrair_rotulos_referencia,
    preparar_features,
    separar_por_perfil,
    validar_dados,
)


# ==========================================
# PARÂMETROS DO TREINAMENTO
# ==========================================

# Percentual reservado para avaliação.
TAMANHO_TESTE: float = 0.20

# Quantidade mínima para realizar divisão treino/teste com stratify.
MINIMO_REGISTROS_AVALIACAO: int = 5

# Quantidade mínima para treinar um modelo.
MINIMO_REGISTROS_TREINO: int = 2

# Reprodutibilidade dos resultados.
RANDOM_STATE: int = 42

# Arquivos dos modelos (versão 2).
ARQUIVO_MODELO_UVA: str = "modelo_uva_v2.pkl"
ARQUIVO_MODELO_MANGA_VERDE: str = "modelo_manga_verde_v2.pkl"
ARQUIVO_MODELO_MANGA_MADURA: str = "modelo_manga_madura_v2.pkl"

MAPEAMENTO_ARQUIVOS: Dict[str, str] = {
    PERFIL_UVA: ARQUIVO_MODELO_UVA,
    PERFIL_MANGA_VERDE: ARQUIVO_MODELO_MANGA_VERDE,
    PERFIL_MANGA_MADURA: ARQUIVO_MODELO_MANGA_MADURA,
}

# Pasta de saída dos modelos (versão 2). Não sobrescreve a pasta modelos/.
PASTA_MODELOS_PADRAO: str = "modelos_v2"

# Regras de referência para detecção de anomalias por limite (inclusive).
# Se o modelo prever diferente dessas regras, é registrada uma divergência.
REGRAS_REFERENCIA: dict = {
    PERFIL_UVA: {
        "temperatura_min": 0.0,
        "temperatura_max": 2.0,
        "umidade_min": 85.0,
        "umidade_max": 95.0,
    },
    PERFIL_MANGA_VERDE: {
        "temperatura_min": 10.0,
        "temperatura_max": 13.0,
        "umidade_min": 85.0,
        "umidade_max": 90.0,
    },
    PERFIL_MANGA_MADURA: {
        "temperatura_min": 10.0,
        "temperatura_max": 13.0,
        "umidade_min": 85.0,
        "umidade_max": 90.0,
    },
}


# ==========================================
# TREINAMENTO DO MODELO
# ==========================================

def treinar_random_forest(
    X: np.ndarray,
    y: np.ndarray,
) -> RandomForestClassifier:
    """Treina um modelo Random Forest Classifier.

    Parametros
    ----------
    X : np.ndarray
        Matriz de features de treinamento (apenas as 4 features numéricas).
    y : np.ndarray
        Vetor de rótulos numéricos (0 = normal, 1 = anomalia).

    Retorna
    -------
    RandomForestClassifier
        Modelo treinado.
    """

    if len(X) < MINIMO_REGISTROS_TREINO:
        raise ValueError(
            "Quantidade insuficiente de registros para treinamento."
        )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    modelo.fit(X, y)

    return modelo


# ==========================================
# AVALIAÇÃO DO MODELO
# ==========================================

def avaliar_modelo(
    modelo: RandomForestClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Optional[dict]:
    """Avalia o modelo utilizando os rótulos de referência apenas no teste.

    Calcula precisão, recall, F1-score, acurácia, matriz de confusão,
    falsos positivos e falsos negativos.

    Parametros
    ----------
    modelo : RandomForestClassifier
        Modelo ja treinado.
    X : np.ndarray
        Features do conjunto de teste.
    y_true : np.ndarray
        Rótulos verdadeiros do conjunto de teste (0 = normal, 1 = anomalia).

    Retorna
    -------
    Optional[dict]
        Dicionario com metricas e contagens. Retorna None se nao houver rotulos.
    """

    if y_true.size == 0:
        return None

    if len(X) != len(y_true):
        raise ValueError(
            "A quantidade de features e rotulos e diferente."
        )

    # Random Forest Classifier:
    # 0 = normal
    # 1 = anomalia
    y_pred = modelo.predict(X)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    acuracia = accuracy_score(y_true, y_pred)

    # Matriz de confusão com ordem fixa: 0 = normal, 1 = anomalia
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # cm = [[TN, FP], [FN, TP]] com labels=[0, 1]
    tn = int(cm[0][0])  # normal previsto como normal
    fp = int(cm[0][1])  # normal previsto como anomalia (falso positivo)
    fn = int(cm[1][0])  # anomalia previsto como normal (falso negativo)
    tp = int(cm[1][1])  # anomalia previsto como anomalia

    return {
        "precisao": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "acuracia": float(acuracia),
        "matriz_confusao": cm.tolist(),
        "total_registros": int(len(y_true)),
        "anomalias_reais": int(y_true.sum()),
        "anomalias_detectadas": int(y_pred.sum()),
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "verdadeiros_positivos": tp,
        "verdadeiros_negativos": tn,
    }


# ==========================================
# DIVISÃO DOS DADOS
# ==========================================

def dividir_dados(
    df_perfil: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Divide os registros em treinamento e teste.

    Utiliza stratify=y quando a quantidade de registros permitir,
    garantindo a representacao de ambas classes nos conjuntos.

    Para bases muito pequenas, retorna todos os registros para treino
    e None para teste, indicando que a avaliacao independente nao e possivel.

    Retorna:
        (df_treino, df_teste) ou (df_treino, None).
    """

    total_registros = len(df_perfil)

    if total_registros < MINIMO_REGISTROS_AVALIACAO:
        return df_perfil.copy(), None

    # Extrair rotulos para stratify
    rotulos = extrair_rotulos_referencia(df_perfil)
    rotulos_numericos = converter_rotulos_para_numericos(rotulos)

    # Verificar se ha pelo menos um exemplo de cada classe para stratify
    classes_unicas = np.unique(rotulos_numericos)
    pode_stratify = len(classes_unicas) >= 2

    # Verificar se cada classe tem exemplos suficientes para divisao
    contagens = pd.Series(rotulos_numericos).value_counts()
    min_classe = contagens.min() if not contagens.empty else 0

    try:
        if pode_stratify and min_classe >= 2:
            df_treino, df_teste = train_test_split(
                df_perfil,
                test_size=TAMANHO_TESTE,
                random_state=RANDOM_STATE,
                stratify=rotulos_numericos,
                shuffle=True,
            )
        else:
            # Divisao simples sem stratify para bases muito pequenas
            df_treino, df_teste = train_test_split(
                df_perfil,
                test_size=TAMANHO_TESTE,
                random_state=RANDOM_STATE,
                shuffle=True,
            )
    except ValueError:
        # Em caso de falha na divisao, usar todos para treino
        return df_perfil.copy(), None

    return df_treino.copy(), df_teste.copy()


# ==========================================
# VALIDAÇÃO DE LIMITES DE REFERÊNCIA
# ==========================================

def verificar_divergencias_referencia(
    df_teste: pd.DataFrame,
    y_pred: np.ndarray,
    perfil: str,
) -> list:
    """Compara as previsões do modelo com as regras de referência por limites.

    Para cada registro de teste, verifica se a temperatura e umidade estão
    dentro das faixas de referência definidas para o perfil. Se o modelo
    prever diferente da regra, registra uma divergência.

    Regras de referência (inclusive):
    - Uva: temp 0-2 °C, umid 85-95% -> fora disso eh anomalia
    - Manga: temp 10-13 °C, umid 85-90% -> fora disso eh anomalia

    Retorna lista de dicionários com as divergências encontradas.
    """

    regras = REGRAS_REFERENCIA.get(perfil)
    if regras is None:
        return []

    divergencias = []

    for idx, (_, linha) in enumerate(df_teste.iterrows()):
        temp = linha["temperatura_c"]
        umid = linha["umidade_relativa_pct"]

        # Regra de referência: está dentro dos limites = normal (0)
        dentro_temp = regras["temperatura_min"] <= temp <= regras["temperatura_max"]
        dentro_umid = regras["umidade_min"] <= umid <= regras["umidade_max"]
        eh_normal_por_regras = bool(dentro_temp and dentro_umid)
        rotulo_regras = 0 if eh_normal_por_regras else 1

        predicao_modelo = int(y_pred[idx])

        if predicao_modelo != rotulo_regras:
            divergencias.append(
                {
                    "indice_teste": int(idx),
                    "temperatura_c": float(temp),
                    "umidade_relativa_pct": float(umid),
                    "rotulo_referencia_regras": rotulo_regras,
                    "predicao_modelo": predicao_modelo,
                    "dentro_limites_temperatura": dentro_temp,
                    "dentro_limites_umidade": dentro_umid,
                }
            )

    return divergencias


# ==========================================
# TREINAMENTO DE TODOS OS PERFIS
# ==========================================

def treinar_todos(
    caminho_csv: str,
    pasta_modelos: str = PASTA_MODELOS_PADRAO,
) -> Dict[str, dict]:
    """Executa o treinamento completo para os tres perfis."""

    os.makedirs(pasta_modelos, exist_ok=True)

    # Carregar os dados.
    df = carregar_dados(caminho_csv)

    # Validar a base (rótulos e valores numéricos).
    valido, erro = validar_dados(df)

    if not valido:
        raise ValueError(f"Dados invalidos: {erro}")

    # Separar por perfil de fruta.
    perfis = separar_por_perfil(df)

    resultados: Dict[str, dict] = {}

    for perfil, df_perfil in perfis.items():

        print(f"\nProcessando perfil: {perfil}")

        # Definir o caminho do modelo antes das verificacoes.
        nome_arquivo = MAPEAMENTO_ARQUIVOS[perfil]

        caminho_modelo = os.path.join(
            pasta_modelos,
            nome_arquivo,
        )

        # Verificar se existem registros.
        if df_perfil.empty:

            resultados[perfil] = {
                "status": "vazio",
                "mensagem": "Nenhum registro para este perfil.",
            }

            continue

        # Verificar quantidade minima para treinamento.
        if len(df_perfil) < MINIMO_REGISTROS_TREINO:

            resultados[perfil] = {
                "status": "insuficiente",
                "registros": int(len(df_perfil)),
                "mensagem": (
                    "Quantidade insuficiente de registros "
                    "para treinar o modelo."
                ),
            }

            continue

        # Dividir os registros em treino e teste.
        df_treino, df_teste = dividir_dados(df_perfil)

        # Preparar features do treinamento (apenas as 4 numericas).
        X_treino, features = preparar_features(df_treino)

        # Extrair rotulos do treinamento (variavel-alvo).
        y_treino_texto = extrair_rotulos_referencia(df_treino)
        y_treino = converter_rotulos_para_numericos(y_treino_texto)

        # Treinar o modelo utilizando rotulos (aprendizado supervisionado).
        modelo = treinar_random_forest(X_treino, y_treino)

        # Inicializar informacoes da avaliacao.
        avaliacao = None
        quantidade_teste = 0
        mensagem_avaliacao = None
        distribuicao_treino = None
        distribuicao_teste = None
        divergencias_referencia = []

        # Calcular distribuicao de classes no treino.
        if len(y_treino) > 0:
            contagem_treino = pd.Series(y_treino).value_counts().to_dict()
            distribuicao_treino = {
                "normal": int(contagem_treino.get(0, 0)),
                "anomalia": int(contagem_treino.get(1, 0)),
            }

        if df_teste is not None and len(df_teste) > 0:

            # Preparar features do conjunto de teste.
            X_teste, _ = preparar_features(df_teste)

            # Extrair rotulos do teste (apenas para avaliacao).
            y_teste_texto = extrair_rotulos_referencia(df_teste)
            y_teste = converter_rotulos_para_numericos(y_teste_texto)

            # Avaliar as predicoes.
            avaliacao = avaliar_modelo(
                modelo,
                X_teste,
                y_teste,
            )

            quantidade_teste = len(df_teste)

            # Calcular distribuicao de classes no teste.
            contagem_teste = pd.Series(y_teste).value_counts().to_dict()
            distribuicao_teste = {
                "normal": int(contagem_teste.get(0, 0)),
                "anomalia": int(contagem_teste.get(1, 0)),
            }

            # Comparar previsões do modelo com regras de referência.
            y_pred_teste = modelo.predict(X_teste)
            divergencias_referencia = verificar_divergencias_referencia(
                df_teste,
                y_pred_teste,
                perfil,
            )

        else:

            mensagem_avaliacao = (
                "Avaliacao independente nao realizada: "
                f"o perfil possui menos de "
                f"{MINIMO_REGISTROS_AVALIACAO} registros. "
                "Todos foram utilizados no treinamento."
            )

        # Salvar o novo modelo na pasta modelos_v2/ (nao sobrescreve modelos_v1).
        joblib.dump(modelo, caminho_modelo)

        print(f"Modelo salvo: {caminho_modelo}")

        # Registrar os resultados.
        resultados[perfil] = {
            "status": "treinado",
            "registros": int(len(df_perfil)),
            "registros_treino": int(len(df_treino)),
            "registros_teste": int(quantidade_teste),
            "features": features,
            "modelo_salvo": caminho_modelo,
            "avaliacao": avaliacao,
            "mensagem_avaliacao": mensagem_avaliacao,
            "distribuicao_treino": distribuicao_treino,
            "distribuicao_teste": distribuicao_teste,
            "divergencias_referencia": divergencias_referencia,
        }

    return resultados


# ==========================================
# EXECUÇÃO DO TREINAMENTO
# ==========================================

def executar_treinamento() -> None:
    """Executa o treinamento utilizando a base CSV versão 2."""

    caminho_csv = os.path.join(
        "dados",
        "base_frio_vale_v2.csv",
    )

    if not os.path.exists(caminho_csv):
        print(f"ERRO: arquivo nao encontrado em {caminho_csv}")

        print(
            "Coloque o arquivo base_frio_vale_v2.csv "
            "na pasta dados/."
        )

        return

    print("=" * 60)
    print("INICIANDO TREINAMENTO DOS MODELOS - FRIOVALE (v2)")
    print("=" * 60)
    print("Algoritmo: Random Forest Classifier (aprendizado supervisionado)")
    print("n_estimators: 200")
    print("random_state: 42")
    print("class_weight: balanced")
    print("Tamanho do teste: 20%")
    print("Features: temperatura_c, umidade_relativa_pct,")
    print("          variacao_temperatura_c, variacao_umidade_pct")
    print("Rotulo alvo: rotulo_referencia (normal=0, anomalia=1)")
    print("Pasta de modelos: modelos_v2/ (nao sobrescreve modelos_v1)")
    print("=" * 60)

    resultados = treinar_todos(caminho_csv)

    for perfil, info in resultados.items():

        print(f"\n--- Perfil: {perfil} ---")
        print(f"Status: {info['status']}")

        if info["status"] == "treinado":

            print(f"Registros totais: {info['registros']}")
            print(
                f"Registros de treino: "
                f"{info['registros_treino']}"
            )
            print(
                f"Registros de teste: "
                f"{info['registros_teste']}"
            )

            print(f"Features: {info['features']}")

            dist_treino = info.get("distribuicao_treino")
            if dist_treino:
                print(
                    f"Distribuicao treino (normal/anomalia): "
                    f"{dist_treino['normal']}/{dist_treino['anomalia']}"
                )

            dist_teste = info.get("distribuicao_teste")
            if dist_teste:
                print(
                    f"Distribuicao teste (normal/anomalia): "
                    f"{dist_teste['normal']}/{dist_teste['anomalia']}"
                )

            print(f"Modelo salvo: {info['modelo_salvo']}")

            avaliacao = info.get("avaliacao")

            if avaliacao:

                print("\nMetricas de avaliacao (apenas conjunto de teste):")

                print(
                    f"Acuracia: {avaliacao['acuracia']:.3f} | "
                    f"Precisao: {avaliacao['precisao']:.3f} | "
                    f"Recall: {avaliacao['recall']:.3f} | "
                    f"F1-score: {avaliacao['f1_score']:.3f}"
                )

                print(
                    "Matriz de confusao: "
                    f"{avaliacao['matriz_confusao']}"
                )
                print(
                    "(ordem: [[TN, FP], [FN, TP]] com labels=[0,1])"
                )

                print(
                    f"Verdadeiros negativos (TN): {avaliacao['verdadeiros_negativos']}"
                )
                print(
                    f"Falsos positivos (FP): {avaliacao['falsos_positivos']}"
                )
                print(
                    f"Falsos negativos (FN): {avaliacao['falsos_negativos']}"
                )
                print(
                    f"Verdadeiros positivos (TP): {avaliacao['verdadeiros_positivos']}"
                )

                print(
                    f"Anomalias reais no teste: "
                    f"{avaliacao['anomalias_reais']}"
                )

                print(
                    f"Anomalias detectadas no teste: "
                    f"{avaliacao['anomalias_detectadas']}"
                )

            elif info.get("mensagem_avaliacao"):

                print(
                    f"\nAVISO: {info['mensagem_avaliacao']}"
                )

            # Exibir divergências com regras de referência.
            div = info.get("divergencias_referencia", [])
            if div:
                print(
                    f"\nDIVERGÊNCIAS vs regras de referência: {len(div)}"
                )
                for d in div:
                    print(
                        f"  idx={d['indice_teste']}: "
                        f"temp={d['temperatura_c']}, umid={d['umidade_relativa_pct']}, "
                        f"regras={d['rotulo_referencia_regras']}, "
                        f"modelo={d['predicao_modelo']}, "
                        f"temp_dentro={d['dentro_limites_temperatura']}, "
                        f"umid_dentro={d['dentro_limites_umidade']}"
                    )
            else:
                print("\nNenhuma divergência encontrada vs regras de referência.")

        else:

            print(f"Mensagem: {info['mensagem']}")

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUIDO (v2)")
    print("=" * 60)


if __name__ == "__main__":
    executar_treinamento()
