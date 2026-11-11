"""Ponto de entrada para executar a API FastAPI com Uvicorn.

Uso:
    python main.py

A API ficara disponivel em http://localhost:8000
Documentacao interativa (Swagger) em http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
