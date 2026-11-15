import joblib
import os
import sys

# Modelos treinados (versão 2 - pasta modelos_v2/)
MODELOS = {
    "uva": "modelos_v2/modelo_uva_v2.pkl",
    "manga_verde": "modelos_v2/modelo_manga_verde_v2.pkl",
    "manga_madura": "modelos_v2/modelo_manga_madura_v2.pkl",
}

# As features devem estar na mesma ordem do treinamento
FEATURES = [
    "temperatura_c",
    "umidade_relativa_pct",
    "variacao_temperatura_c",
    "variacao_umidade_pct",
]

# Faixas de referência (inclusive)
FAIXAS_REFERENCIA = {
    "uva": {
        "temperatura_min": 0.0,
        "temperatura_max": 2.0,
        "umidade_min": 85.0,
        "umidade_max": 95.0,
    },
    "manga_verde": {
        "temperatura_min": 10.0,
        "temperatura_max": 13.0,
        "umidade_min": 85.0,
        "umidade_max": 90.0,
    },
    "manga_madura": {
        "temperatura_min": 10.0,
        "temperatura_max": 13.0,
        "umidade_min": 85.0,
        "umidade_max": 90.0,
    },
}

INTERVALO_SEGUNDOS = 15


def resultado_esperado_por_regras(fruta, temperatura_c, umidade_relativa_pct):
    """Retorna o resultado esperado pelas regras de referência."""
    faixas = FAIXAS_REFERENCIA[fruta]
    dentro_temp = (
        faixas["temperatura_min"] <= temperatura_c <= faixas["temperatura_max"]
    )
    dentro_umid = (
        faixas["umidade_min"] <= umidade_relativa_pct <= faixas["umidade_max"]
    )
    if dentro_temp and dentro_umid:
        return "NORMAL"
    return "ANOMALIA"


def gerar_sequencia_estavel(fruta, num_leituras=10):
    """Cenário 1: Temperatura e umidade estáveis dentro das faixas."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_medio = (faixas["temperatura_min"] + faixas["temperatura_max"]) / 2
    umid_medio = (faixas["umidade_min"] + faixas["umidade_max"]) / 2

    leituras = []
    for i in range(num_leituras):
        leituras.append(
            {
                "temperatura_c": round(temp_medio, 2),
                "umidade_relativa_pct": round(umid_medio, 2),
                "variacao_temperatura_c": 0.0 if i == 0 else 0.0,
                "variacao_umidade_pct": 0.0 if i == 0 else 0.0,
            }
        )
    return leituras


def gerar_sequencia_temp_aumentando(fruta, num_leituras=10):
    """Cenário 2: Temperatura aumentando gradualmente até ultrapassar o limite."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_inicial = faixas["temperatura_min"]
    temp_final = faixas["temperatura_max"] + 5.0  # sai da faixa
    passo = (temp_final - temp_inicial) / (num_leituras - 1)

    umid_medio = (faixas["umidade_min"] + faixas["umidade_max"]) / 2

    leituras = []
    temp_anterior = None
    for i in range(num_leituras):
        temp = round(temp_inicial + passo * i, 2)
        variacao_temp = 0.0 if temp_anterior is None else round(temp - temp_anterior, 2)
        leituras.append(
            {
                "temperatura_c": temp,
                "umidade_relativa_pct": round(umid_medio, 2),
                "variacao_temperatura_c": variacao_temp,
                "variacao_umidade_pct": 0.0,
            }
        )
        temp_anterior = temp
    return leituras


def gerar_sequencia_temp_diminuindo(fruta, num_leituras=10):
    """Cenário 3: Temperatura diminuindo gradualmente até ultrapassar o limite."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_inicial = faixas["temperatura_max"]
    temp_final = faixas["temperatura_min"] - 5.0  # sai da faixa
    passo = (temp_final - temp_inicial) / (num_leituras - 1)

    umid_medio = (faixas["umidade_min"] + faixas["umidade_max"]) / 2

    leituras = []
    temp_anterior = None
    for i in range(num_leituras):
        temp = round(temp_inicial + passo * i, 2)
        variacao_temp = 0.0 if temp_anterior is None else round(temp - temp_anterior, 2)
        leituras.append(
            {
                "temperatura_c": temp,
                "umidade_relativa_pct": round(umid_medio, 2),
                "variacao_temperatura_c": variacao_temp,
                "variacao_umidade_pct": 0.0,
            }
        )
        temp_anterior = temp
    return leituras


def gerar_sequencia_umidade_aumentando(fruta, num_leituras=10):
    """Cenário 4: Umidade aumentando gradualmente até ultrapassar o limite."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_medio = (faixas["temperatura_min"] + faixas["temperatura_max"]) / 2
    umid_inicial = faixas["umidade_min"]
    umid_final = faixas["umidade_max"] + 10.0  # sai da faixa
    passo = (umid_final - umid_inicial) / (num_leituras - 1)

    leituras = []
    umid_anterior = None
    for i in range(num_leituras):
        umid = round(umid_inicial + passo * i, 2)
        variacao_umid = (
            0.0 if umid_anterior is None else round(umid - umid_anterior, 2)
        )
        leituras.append(
            {
                "temperatura_c": round(temp_medio, 2),
                "umidade_relativa_pct": umid,
                "variacao_temperatura_c": 0.0,
                "variacao_umidade_pct": variacao_umid,
            }
        )
        umid_anterior = umid
    return leituras


def gerar_sequencia_umidade_diminuindo(fruta, num_leituras=10):
    """Cenário 5: Umidade diminuindo gradualmente até ultrapassar o limite."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_medio = (faixas["temperatura_min"] + faixas["temperatura_max"]) / 2
    umid_inicial = faixas["umidade_max"]
    umid_final = faixas["umidade_min"] - 10.0  # sai da faixa
    passo = (umid_final - umid_inicial) / (num_leituras - 1)

    leituras = []
    umid_anterior = None
    for i in range(num_leituras):
        umid = round(umid_inicial + passo * i, 2)
        variacao_umid = (
            0.0 if umid_anterior is None else round(umid - umid_anterior, 2)
        )
        leituras.append(
            {
                "temperatura_c": round(temp_medio, 2),
                "umidade_relativa_pct": umid,
                "variacao_temperatura_c": 0.0,
                "variacao_umidade_pct": variacao_umid,
            }
        )
        umid_anterior = umid
    return leituras


def gerar_sequencia_retorno_normal(fruta, num_leituras=12):
    """Cenário 6: Temperatura e umidade retornando gradualmente à faixa normal."""
    faixas = FAIXAS_REFERENCIA[fruta]
    temp_fora = faixas["temperatura_max"] + 5.0
    temp_normal = (faixas["temperatura_min"] + faixas["temperatura_max"]) / 2
    umid_fora = faixas["umidade_max"] + 10.0
    umid_normal = (faixas["umidade_min"] + faixas["umidade_max"]) / 2

    metade = num_leituras // 2

    leituras = []
    temp_anterior = temp_fora
    umid_anterior = umid_fora

    # Primeira metade: saindo da faixa normal (começa fora)
    for i in range(metade):
        temp = round(temp_fora + (temp_normal - temp_fora) * (i / (metade - 1)), 2) if metade > 1 else temp_normal
        umid = round(umid_fora + (umid_normal - umid_fora) * (i / (metade - 1)), 2) if metade > 1 else umid_normal
        variacao_temp = round(temp - temp_anterior, 2)
        variacao_umid = round(umid - umid_anterior, 2)
        leituras.append(
            {
                "temperatura_c": temp,
                "umidade_relativa_pct": umid,
                "variacao_temperatura_c": variacao_temp,
                "variacao_umidade_pct": variacao_umid,
            }
        )
        temp_anterior = temp
        umid_anterior = umid

    # Segunda metade: já dentro da faixa normal
    for i in range(metade, num_leituras):
        leituras.append(
            {
                "temperatura_c": round(temp_normal, 2),
                "umidade_relativa_pct": round(umid_normal, 2),
                "variacao_temperatura_c": 0.0,
                "variacao_umidade_pct": 0.0,
            }
        )

    return leituras


def executar_sequencia(
    modelo, fruta, nome_cenario, leituras, numero_sequencia
):
    """Executa uma sequência de leituras no modelo e retorna estatísticas."""
    print(f"\n{'=' * 70}")
    print(f"CENÁRIO {numero_sequencia}: {nome_cenario} - Perfil: {fruta}")
    print(f"{'=' * 70}")

    total = len(leituras)
    acertos = 0
    erros = 0
    falsos_positivos = 0
    falsos_negativos = 0

    primeira_fora_faixa = None
    primeira_detec_anomalia = None
    atraso_leituras = None

    leitura_anterior_temp = None
    leitura_anterior_umid = None

    for idx, leitura in enumerate(leituras):
        temp = leitura["temperatura_c"]
        umid = leitura["umidade_relativa_pct"]

        # Calcular variações em relação à leitura anterior
        if leitura_anterior_temp is not None:
            variacao_temp = round(temp - leitura_anterior_temp, 2)
        else:
            variacao_temp = 0.0

        if leitura_anterior_umid is not None:
            variacao_umid = round(umid - leitura_anterior_umid, 2)
        else:
            variacao_umid = 0.0

        # Atualizar leituras anteriores para a próxima iteração
        leitura_anterior_temp = temp
        leitura_anterior_umid = umid

        # Montar features na ordem correta do treinamento
        entrada = [[temp, umid, variacao_temp, variacao_umid]]

        # Previsão da IA
        previsao = int(modelo.predict(entrada)[0])
        resultado_ia = "ANOMALIA" if previsao == 1 else "NORMAL"

        # Resultado esperado pelas regras
        esperado = resultado_esperado_por_regras(fruta, temp, umid)

        # Verificar acerto/erro
        acertou = resultado_ia == esperado
        eh_fp = False
        eh_fn = False
        if not acertou:
            if esperado == "NORMAL" and previsao == 1:
                eh_fp = True
            elif esperado == "ANOMALIA" and previsao == 0:
                eh_fn = True

        if acertou:
            acertos += 1
        else:
            erros += 1
            if eh_fp:
                falsos_positivos += 1
            if eh_fn:
                falsos_negativos += 1

        # Detectar primeira leitura fora da faixa (pelas regras)
        if esperado == "ANOMALIA" and primeira_fora_faixa is None:
            primeira_fora_faixa = idx + 1  # 1-based

        # Detectar primeira leitura em que a IA detectou anomalia
        if previsao == 1 and primeira_detec_anomalia is None:
            primeira_detec_anomalia = idx + 1  # 1-based

        # Calcular atraso se ambas foram encontradas
        if primeira_fora_faixa is not None and primeira_detec_anomalia is not None:
            atraso_leituras = primeira_detec_anomalia - primeira_fora_faixa

        # Exibir detalhes da leitura
        tempo_simulado = idx * INTERVALO_SEGUNDOS
        print(f"\nLeitura {idx + 1} (t +{tempo_simulado}s):")
        print(f"  Temperatura: {temp} °C")
        print(f"  Umidade: {umid} %")
        print(f"  Variação temperatura: {variacao_temp} °C")
        print(f"  Variação umidade: {variacao_umid} %")
        print(f"  Esperado pelas regras: {esperado}")
        print(f"  Previsão da IA: {resultado_ia}")
        print(f"  Resultado: {'ACERTOU' if acertou else 'ERROU'}")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada)[0]
            try:
                indice_anomalia = list(modelo.classes_).index(1)
                confianca = probabilidades[indice_anomalia] * 100
                print(f"  Probabilidade de anomalia: {confianca:.2f}%")
            except ValueError:
                pass

    # Calcular atraso final
    if (
        primeira_fora_faixa is not None
        and primeira_detec_anomalia is not None
    ):
        atraso_leituras = primeira_detec_anomalia - primeira_fora_faixa
    elif primeira_fora_faixa is not None and primeira_detec_anomalia is None:
        atraso_leituras = None  # nunca detectou

    estatisticas = {
        "total": total,
        "acertos": acertos,
        "erros": erros,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "primeira_fora_faixa": primeira_fora_faixa,
        "primeira_detec_anomalia": primeira_detec_anomalia,
        "atraso_leituras": atraso_leituras,
    }

    return estatisticas


def testar_sequencias():
    print("=" * 70)
    print("TESTE DE SEQUENCIAS - MODELOS V2 - FRIOVALE")
    print("=" * 70)
    print(f"Intervalo simulado entre leituras: {INTERVALO_SEGUNDOS} segundos")
    print("")

    # Carregar os três modelos da pasta modelos_v2/
    modelos_carregados = {}
    for fruta, caminho in MODELOS.items():
        try:
            modelos_carregados[fruta] = joblib.load(caminho)
            print(f"Modelo carregado: {fruta} ({caminho})")
        except FileNotFoundError:
            print(f"ERRO: arquivo nao encontrado: {caminho}")
            return
        except Exception as e:
            print(f"ERRO ao carregar {fruta}: {e}")
            return

    # Definir cenários e suas funções geradoras
    cenarios = [
        (1, "Temperatura e umidade estáveis dentro da faixa", gerar_sequencia_estavel),
        (2, "Temperatura aumentando gradualmente até ultrapassar o limite", gerar_sequencia_temp_aumentando),
        (3, "Temperatura diminuindo gradualmente até ultrapassar o limite", gerar_sequencia_temp_diminuindo),
        (4, "Umidade aumentando gradualmente até ultrapassar o limite", gerar_sequencia_umidade_aumentando),
        (5, "Umidade diminuindo gradualmente até ultrapassar o limite", gerar_sequencia_umidade_diminuindo),
        (6, "Temperatura e umidade retornando gradualmente à faixa normal", gerar_sequencia_retorno_normal),
    ]

    # Estatísticas agregadas por perfil
    estatisticas_perfil = {
        "uva": {},
        "manga_verde": {},
        "manga_madura": {},
    }

    for fruta in MODELOS.keys():
        modelo = modelos_carregados[fruta]
        print(f"\n{'#' * 70}")
        print(f"# PERFIL: {fruta.upper()}")
        print(f"{'#' * 70}")

        for num_cenario, nome_cenario, funcao_geradora in cenarios:
            leituras = funcao_geradora(fruta)
            stats = executar_sequencia(
                modelo, fruta, nome_cenario, leituras, num_cenario
            )
            estatisticas_perfil[fruta][num_cenario] = stats

    # Relatório final agregado
    print("\n" + "=" * 70)
    print("RELATORIO FINAL AGREGADO - POR PERFIL E CENÁRIO")
    print("=" * 70)

    for fruta in MODELOS.keys():
        print(f"\n{'=' * 70}")
        print(f"PERFIL: {fruta.upper()}")
        print(f"{'=' * 70}")

        for num_cenario, nome_cenario, _ in cenarios:
            stats = estatisticas_perfil[fruta][num_cenario]
            total = stats["total"]
            acertos = stats["acertos"]
            erros = stats["erros"]
            fp = stats["falsos_positivos"]
            fn = stats["falsos_negativos"]
            pff = stats["primeira_fora_faixa"]
            pda = stats["primeira_detec_anomalia"]
            atraso = stats["atraso_leituras"]

            taxa_acerto = (acertos / total * 100) if total > 0 else 0.0

            print(f"\n  Cenário {num_cenario}: {nome_cenario}")
            print(f"    Total de leituras: {total}")
            print(f"    Acertos: {acertos} | Erros: {erros}")
            print(f"    Falsos positivos: {fp} | Falsos negativos: {fn}")
            print(f"    Taxa de acerto: {taxa_acerto:.2f}%")
            print(f"    Primeira leitura fora da faixa (regras): {pff if pff else 'N/A'}")
            print(
                f"    Primeira detecção de anomalia (IA): {pda if pda else 'N/A'}"
            )
            if atraso is not None:
                print(f"    Atraso de detecção: {atraso} leitura(ções)")
            else:
                print(f"    Atraso de detecção: N/A (anomalia não detectada ou sem leitura fora da faixa)")


if __name__ == "__main__":
    testar_sequencias()
