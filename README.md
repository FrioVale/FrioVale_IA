# Módulo de Inteligência Artificial — FrioVale

## Objetivo

Detectar **anomalias nas condições ambientais** durante o transporte de frutas (uvas e mangas),
identificando comportamentos fora do padrão aprendido, como:

- Aumento excessivo da temperatura.
- Redução excessiva da temperatura.
- Variações incomuns de temperatura.
- Umidade fora do comportamento esperado.
- Combinações incomuns entre temperatura e umidade.

A IA **não diagnostica** se a fruta está estragada e **não detecta abertura de portas**
(esses recursos dependem de sensores que ainda não estão disponíveis).

## Tecnologias utilizadas

- Python 3.9+
- pandas — manipulação e exploração dos dados
- NumPy — operações numéricas
- scikit-learn — modelo Isolation Forest e métricas de avaliação
- Joblib — salvar e carregar modelos treinados (`.pkl`)
- FastAPI — API REST para integração com o backend principal
- Uvicorn — servidor para executar a API
- pytest — testes automatizados

## Estrutura de pastas

```
IA/
├── dados/                 # Coloque aqui o arquivo base_sintetica_frio_vale.csv
├── modelos/               # Aqui serão salvos os modelos .pkl após o treinamento
├── src/
│   ├── processamento.py   # Carregamento, validação e preparação dos dados
│   ├── treinamento.py     # Treino e avaliação dos modelos por perfil de fruta
│   ├── predicao.py        # Carregamento de modelos e predição de novas leituras
│   └── api.py             # Endpoints FastAPI: /health, /modelos, /prever
├── testes/
│   ├── test_processamento.py
│   ├── test_predicao.py
│   └── test_api.py
├── main.py                # Executa a API com uvicorn
├── requirements.txt       # Dependências do projeto
└── README.md              # Este arquivo
```

## Como instalar as dependências

No VS Code, abra a pasta `IA/` e no terminal integrado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Onde colocar o arquivo CSV

Copie o arquivo `base_sintetica_frio_vale.csv` para a pasta:

```
IA/dados/base_sintetica_frio_vale.csv
```

O arquivo deve conter, no mínimo, as colunas:
`timestamp`, `tipo_fruta`, `estagio_maturacao`, `temperatura_c`,
`umidade_relativa_pct`, `variacao_temperatura_c`, `variacao_umidade_pct`,
`rotulo_referencia`, `cenario_simulado`.

## Como executar o treinamento

Com o ambiente virtual ativado e o CSV na pasta `dados/`:

```powershell
cd src
python treinamento.py
```

O script irá:
1. Carregar o CSV.
2. Verificar valores ausentes e inconsistências.
3. Separar os dados por perfil: **uva**, **manga verde-madura**, **manga parcialmente madura ou madura**.
4. Treinar um Isolation Forest para cada perfil (usando apenas leituras normais).
5. Avaliar com precisão, recall, F1-score e matriz de confusão.
6. Salvar os modelos na pasta `modelos/`:
   - `modelo_uva.pkl`
   - `modelo_manga_verde.pkl`
   - `modelo_manga_madura.pkl`

> Limitação: a avaliação é feita com dados sintéticos, portanto os resultados são indicativos
> e não representam comportamento real de sensores em campo.

## Como executar a API

Com o ambiente virtual ativado:

```powershell
python main.py
```

A API estará disponível em `http://localhost:8000`.

## Swagger (documentação interativa)

Após iniciar a API, acesse:

```
http://localhost:8000/docs
```

Lá você pode testar os endpoints diretamente pelo navegador, especialmente o `POST /prever`.

## Endpoints da API

### GET /health
Verifica se a API está online e se os modelos foram carregados.

### GET /modelos
Lista os modelos disponíveis (uva, manga verde, manga madura).

### POST /prever
Recebe uma nova leitura e retorna se houve anomalia.

Exemplo de corpo da requisição:

```json
{
  "tipo_fruta": "uva",
  "estagio_maturacao": "mesa",
  "temperatura_c": -0.5,
  "umidade_relativa_pct": 92.0
}
```

Exemplo de resposta:

```json
{
  "anomalia": false,
  "resultado": "normal",
  "mensagem": "Comportamento dentro do padrão aprendido.",
  "tipo_fruta": "uva"
}
```

- `anomalia: true` → possível comportamento incomum (não significa que a fruta está estragada).
- `anomalia: false` → dentro do padrão aprendido.

## Como interpretar os resultados

O Isolation Forest retorna:
- `1` → comportamento dentro do padrão aprendido.
- `-1` → possível anomalia.

A API traduz isso para `anomalia: false` ou `anomalia: true` com uma mensagem simples.
Sempre que `anomalia: true`, o backend principal do FrioVale pode registrar o evento
e exibir um aviso no dashboard para verificação humana.

## Preparação para integração futura

O módulo funciona de forma independente via HTTP. No futuro, o backend do FrioVale
(pode ser o Spring Boot existente) deverá:

1. Receber leituras do ESP32 via MQTT.
2. Armazenar as leituras no banco de dados.
3. Enviar `temperatura_c`, `umidade_relativa_pct`, `tipo_fruta` e `estagio_maturacao`
   para `POST http://localhost:8000/prever`.
4. Receber a classificação da IA e salvar o resultado.

Não há MQTT neste módulo. A integração será exclusivamente por chamada REST.

## Executar testes automatizados

Com o ambiente virtual ativado, na pasta raiz do módulo `IA/`:

```powershell
pytest testes/ -v
```

## Executar no VS Code

1. Abra a pasta `IA/` no VS Code (**Arquivo > Abrir Pasta**).
2. Instale a extensão **Python** (Microsoft).
3. Crie e ative o ambiente virtual no terminal integrado.
4. Rode `pip install -r requirements.txt`.
5. Para iniciar a API: `python main.py`.
6. Para rodar os testes: `pytest testes/ -v`.
7. Para gerar os modelos: `cd src && python treinamento.py`.
