"""API REST do modulo de IA do FrioVale.

Endpoints:
- GET /health  -> status da API e modelos carregados
- GET /modelos -> lista de modelos disponíveis
- POST /prever -> analisa uma nova leitura e retorna se eh anomalia

Integração futura com o backend principal (Spring Boot) ocorre por chamadas HTTP
a esses endpoints. Nao ha MQTT neste modulo.
"""

from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat

from src.predicao import (
    PASTA_MODELOS_PADRAO,
    carregar_preditor,
)

app = FastAPI(
    title="FrioVale IA - Detecção de Anomalias",
    description="API para detecção de anomalias nas condições de transporte de frutas.",
    version="1.0.0",
)

# Instancia global do preditor (carregada na inicializacao)
preditor = None


class LeituraEntrada(BaseModel):
    """Dados de entrada para predicao de anomalia."""

    tipo_fruta: str = Field(
        ..., description="Tipo da fruta (ex: uva, manga)"
    )
    estagio_maturacao: str = Field(
        ..., description="Estagio de maturacao (ex: mesa, verde, madura)"
    )
    temperatura_c: confloat(ge=-50, le=80) = Field(
        ..., description="Temperatura em graus Celsius"
    )
    umidade_relativa_pct: confloat(ge=0, le=100) = Field(
        ..., description="Umidade relativa em porcentagem"
    )


class PrevisaoSaida(BaseModel):
    """Resposta da API com o resultado da predicao."""

    anomalia: bool
    resultado: str
    mensagem: str
    tipo_fruta: str
    perfil: str
    valores_recebidos: Dict


@app.on_event("startup")
def carregar_modelos_na_inicializacao() -> None:
    """Carrega os modelos quando a API inicia."""
    global preditor
    preditor = carregar_preditor(pasta_modelos=PASTA_MODELOS_PADRAO)
    carregados = preditor.carregados
    print(f"API iniciada. Modelos carregados: {carregados}")


@app.get("/health", tags=["Sistema"])
def health() -> Dict:
    """Verifica se a API esta online e se os modelos foram carregados."""
    if preditor is None:
        return {"status": "erro", "modelos_carregados": False}
    return {
        "status": "ok",
        "modelos_carregados": preditor.carregados,
        "modelos": list(preditor.modelos.keys()) if preditor.modelos else [],
    }


@app.get("/modelos", tags=["Sistema"])
def listar_modelos() -> Dict:
    """Lista os modelos disponiveis no modulo de IA."""
    if preditor is None or not preditor.modelos:
        return {"modelos_disponiveis": [], "mensagem": "Nenhum modelo carregado."}
    return {
        "modelos_disponiveis": list(preditor.modelos.keys()),
        "total": len(preditor.modelos),
    }


@app.post("/prever", tags=["Predicao"])
def prever(leitura: LeituraEntrada) -> Dict:
    """Recebe uma nova leitura e retorna se foi identificada anomalia.

    Nao diagnostica qualidade da fruta. Apenas sinaliza comportamento
    fora do padrao aprendido pelo Isolation Forest.
    """
    if preditor is None or not preditor.carregados:
        raise HTTPException(
            status_code=503,
            detail="Modelos de IA nao estao disponiveis. Verifique o treinamento.",
        )

    try:
        resultado = preditor.prever(
            tipo_fruta=leitura.tipo_fruta,
            estagio_maturacao=leitura.estagio_maturacao,
            temperatura_c=leitura.temperatura_c,
            umidade_relativa_pct=leitura.umidade_relativa_pct,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na predicao: {str(exc)}")

    if resultado.get("resultado") == "erro":
        raise HTTPException(status_code=400, detail=resultado.get("mensagem", "Erro na predicao."))

    return resultado
