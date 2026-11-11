"""Predicao de anomalias para o modulo de IA do FrioVale.

Carrega os modelos treinados (.pkl) e realiza a analise de uma nova leitura,
identificando o perfil da fruta e selecionando o modelo correspondente.

O Isolation Forest retorna:
- 1: comportamento dentro do padrao aprendido.
- -1: possivel anomalia.

Nao diagnostica qualidade da fruta; apenas sinaliza comportamento incomum.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

from src.processamento import (
    ESTAGIOS_MANGA_MADURA,
    ESTAGIOS_MANGA_VERDE,
    FEATURES_BASE,
    PERFIL_MANGA_MADURA,
    PERFIL_MANGA_VERDE,
    PERFIL_UVA,
    _normalizar_texto,
    preparar_features,
)

# Caminho padrao da pasta de modelos (ajustavel via construtor)
PASTA_MODELOS_PADRAO: str = "modelos"

ARQUIVO_MODELO_UVA: str = "modelo_uva.pkl"
ARQUIVO_MODELO_MANGA_VERDE: str = "modelo_manga_verde.pkl"
ARQUIVO_MODELO_MANGA_MADURA: str = "modelo_manga_madura.pkl"

MAPEAMENTO_ARQUIVOS: Dict[str, str] = {
    PERFIL_UVA: ARQUIVO_MODELO_UVA,
    PERFIL_MANGA_VERDE: ARQUIVO_MODELO_MANGA_VERDE,
    PERFIL_MANGA_MADURA: ARQUIVO_MODELO_MANGA_MADURA,
}

# Mensagens padronizadas de retorno
MENSAGEM_NORMAL: str = "Comportamento dentro do padrao aprendido."
MENSAGEM_ANOMALIA: str = (
    "Comportamento fora do padrao aprendido. Verifique as condições de transporte."
)


class PreditorIA:
    """Classe responsavel por carregar modelos e predizer anomalias."""

    def __init__(self, pasta_modelos: str = PASTA_MODELOS_PADRAO) -> None:
        self.pasta_modelos = pasta_modelos
        self.modelos: Dict[str, object] = {}
        self.carregados: bool = False

    def _caminho_modelo(self, perfil: str) -> str:
        nome = MAPEAMENTO_ARQUIVOS.get(perfil)
        if not nome:
            raise ValueError(f"Perfil desconhecido: {perfil}")
        return os.path.join(self.pasta_modelos, nome)

    def carregar_modelos(self) -> Dict[str, bool]:
        """Carrega todos os modelos disponíveis na pasta de modelos.

        Retorna
        -------
        Dict[str, bool]
            Mapeamento perfil -> True se carregado com sucesso.
        """
        status: Dict[str, bool] = {}
        for perfil in MAPEAMENTO_ARQUIVOS.keys():
            caminho = self._caminho_modelo(perfil)
            if os.path.exists(caminho):
                try:
                    self.modelos[perfil] = joblib.load(caminho)
                    status[perfil] = True
                except Exception:  # noqa: BLE001
                    status[perfil] = False
            else:
                status[perfil] = False
        self.carregados = any(status.values())
        return status

    def identificar_perfil(
        self, tipo_fruta: str, estagio_maturacao: str
    ) -> Optional[str]:
        """Identifica o perfil da fruta com base no tipo e estagio de maturacao.

        Parametros
        ----------
        tipo_fruta : str
            Tipo da fruta ('uva' ou 'manga').
        estagio_maturacao : str
            Estagio de maturacao informado.

        Retorna
        -------
        Optional[str]
            Nome do perfil ou None se nao possivel identificar.
        """
        tipo = _normalizar_texto(tipo_fruta)
        estagio = _normalizar_texto(estagio_maturacao)

        if tipo == "uva":
            return PERFIL_UVA

        if tipo == "manga":
            estagios_verde = [e.lower() for e in ESTAGIOS_MANGA_VERDE]
            estagios_madura = [e.lower() for e in ESTAGIOS_MANGA_MADURA]
            if estagio in estagios_verde:
                return PERFIL_MANGA_VERDE
            if estagio in estagios_madura:
                return PERFIL_MANGA_MADURA
            # Se nao souber o estagio mas for manga, retorna verde como padrao seguro
            # (poderia ser ajustado futuramente com mais informacoes).
            return PERFIL_MANGA_VERDE

        return None

    def prever(
        self,
        tipo_fruta: str,
        estagio_maturacao: str,
        temperatura_c: float,
        umidade_relativa_pct: float,
    ) -> Dict:
        """Realiza a predicao de anomalia para uma nova leitura.

        Parametros
        ----------
        tipo_fruta : str
        estagio_maturacao : str
        temperatura_c : float
        umidade_relativa_pct : float

        Retorna
        -------
        Dict
            Dicionario com anomalia (bool), resultado (str), mensagem (str),
            tipo_fruta (str), perfil (str) e valores recebidos.
        """
        if not self.carregados or not self.modelos:
            raise RuntimeError("Nenhum modelo carregado. Chame carregar_modelos() primeiro.")

        perfil = self.identificar_perfil(tipo_fruta, estagio_maturacao)
        if perfil is None:
            return {
                "anomalia": None,
                "resultado": "erro",
                "mensagem": "Nao foi possivel identificar o perfil da fruta.",
                "tipo_fruta": tipo_fruta,
                "perfil": None,
                "valores_recebidos": {
                    "temperatura_c": temperatura_c,
                    "umidade_relativa_pct": umidade_relativa_pct,
                },
            }

        modelo = self.modelos.get(perfil)
        if modelo is None:
            return {
                "anomalia": None,
                "resultado": "erro",
                "mensagem": f"Modelo nao disponivel para o perfil: {perfil}",
                "tipo_fruta": tipo_fruta,
                "perfil": perfil,
                "valores_recebidos": {
                    "temperatura_c": temperatura_c,
                    "umidade_relativa_pct": umidade_relativa_pct,
                },
            }

        X = np.array([[temperatura_c, umidade_relativa_pct]], dtype=float)
        predicao = modelo.predict(X)[0]

        eh_anomalia = bool(predicao == -1)
        resultado = "anomalia" if eh_anomalia else "normal"
        mensagem = MENSAGEM_ANOMALIA if eh_anomalia else MENSAGEM_NORMAL

        return {
            "anomalia": eh_anomalia,
            "resultado": resultado,
            "mensagem": mensagem,
            "tipo_fruta": tipo_fruta,
            "perfil": perfil,
            "valores_recebidos": {
                "temperatura_c": temperatura_c,
                "umidade_relativa_pct": umidade_relativa_pct,
            },
        }


def carregar_preditor(pasta_modelos: str = PASTA_MODELOS_PADRAO) -> PreditorIA:
    """Factory que cria e carrega um PreditorIA pronto para uso."""
    preditor = PreditorIA(pasta_modelos=pasta_modelos)
    preditor.carregar_modelos()
    return preditor
