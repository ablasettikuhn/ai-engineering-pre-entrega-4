"""
config.py
---------
Configuración única del proyecto. Todos los scripts leen de aquí, así que es
imposible indexar con un modelo de embeddings y consultar con otro, o subir
vectores de una dimensión a un índice creado con otra.

Los valores se leen del archivo .env (ver .env.example).
"""

import logging
import os
import re
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# langchain-community avisa que dejará de mantenerse. Solo usamos su BM25Retriever
# (la clase que pide la consigna), así que el aviso se silencia para no ensuciar la consola.
warnings.filterwarnings("ignore", message=r".*langchain-community.*")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"
PDF_DIR = DATA_DIR / "pdf"
RESULTADOS_DIR = RAIZ / "resultados"
FUENTES = RAIZ / "fuentes.json"
GOLDEN_SET = RAIZ / "golden_set.json"
PREGUNTAS_TRAMPA = RAIZ / "preguntas_trampa.json"
MANIFEST = RESULTADOS_DIR / "manifest_ingesta.json"
README = RAIZ / "README.md"

# ---------------------------------------------------------------------------
# Pinecone (Serverless)
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
INDEX_NAME = os.getenv("INDEX_NAME", "papers-ciencias-deporte").strip()
# El namespace separa tipos de datos dentro del mismo índice (ver README).
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "papers-cientificos").strip()
# El plan gratuito (Starter) solo permite índices Serverless en AWS us-east-1.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()
PINECONE_METRIC = "cosine"

if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", INDEX_NAME):
    raise ValueError("INDEX_NAME solo admite minúsculas, números y guiones (máximo 45 caracteres).")

# ---------------------------------------------------------------------------
# Embeddings (los mismos para indexar y para consultar)
# ---------------------------------------------------------------------------
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "gemini-embedding-001").strip()
# 1536 es la dimensión que sugiere la consigna. gemini-embedding-001 permite elegirla
# (768, 1536 o 3072), así que el índice queda igual que con text-embedding-3-small.
EMBEDDINGS_DIM = int(os.getenv("EMBEDDINGS_DIM", "1536"))
# Plan gratuito de Gemini: la ingesta envía lotes que no superan este presupuesto por minuto.
EMBEDDINGS_TOKENS_POR_MINUTO = int(os.getenv("EMBEDDINGS_TOKENS_POR_MINUTO", "25000"))

# ---------------------------------------------------------------------------
# Chunking (medido en tokens, encoding cl100k_base de tiktoken)
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not 500 <= CHUNK_SIZE <= 800:
    raise ValueError("CHUNK_SIZE debe estar entre 500 y 800 tokens (punto medio que recomienda la consigna).")
if not 0 < CHUNK_OVERLAP < CHUNK_SIZE // 2:
    raise ValueError("CHUNK_OVERLAP debe ser mayor que 0 y menor que la mitad de CHUNK_SIZE.")

# ---------------------------------------------------------------------------
# Recuperación híbrida
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))  # documentos que devuelve el sistema (Top-5 de la consigna)
K_CANDIDATOS = int(os.getenv("K_CANDIDATOS", "10"))  # candidatos que aporta CADA recuperador al ensemble
PESO_BM25 = float(os.getenv("PESO_BM25", "0.4"))
PESO_VECTORIAL = round(1 - PESO_BM25, 4)

# Umbral mínimo de similitud coseno (búsqueda vectorial) para considerar que la biblioteca
# tiene evidencia sobre la pregunta. Si la similitud máxima queda por debajo, el sistema
# responde "No lo sé" sin llamar al LLM. Se calibra con `python evaluate.py` (ver README).
UMBRAL_SIMILITUD = float(os.getenv("UMBRAL_SIMILITUD", "0.70"))

if K_CANDIDATOS < TOP_K:
    raise ValueError("K_CANDIDATOS debe ser mayor o igual que TOP_K.")
if not 0 <= PESO_BM25 <= 1:
    raise ValueError("PESO_BM25 debe estar entre 0 y 1.")
if not 0 <= UMBRAL_SIMILITUD < 1:
    raise ValueError("UMBRAL_SIMILITUD debe estar entre 0 y 1.")

# ---------------------------------------------------------------------------
# LLM (solo para `consultar.py --responder`)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()


def clave_gemini() -> str:
    clave = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not clave:
        raise RuntimeError("Falta GOOGLE_API_KEY en el .env: se necesita para los embeddings de Gemini.")
    return clave


def clave_pinecone() -> str:
    if not PINECONE_API_KEY:
        raise RuntimeError("Falta PINECONE_API_KEY en el .env (se crea en https://app.pinecone.io).")
    return PINECONE_API_KEY


def obtener_embeddings():
    """Modelo de embeddings que usan la ingesta y la consulta.

    embed_documents usa task_type RETRIEVAL_DOCUMENT y embed_query usa RETRIEVAL_QUERY:
    es el MISMO modelo, optimizado para cada lado de la búsqueda.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=clave_gemini(),
        output_dimensionality=EMBEDDINGS_DIM,
    )


def firma_embeddings() -> dict:
    """Identifica el espacio vectorial: si cambia, hay que re-indexar."""
    return {"embeddings_model": EMBEDDINGS_MODEL, "embeddings_dim": EMBEDDINGS_DIM}


def configurar_logging(archivo: str | None = None, acumular: bool = False) -> None:
    """Log en consola y, opcionalmente, en resultados/<archivo> (evidencia para la corrección).

    acumular=True agrega al final del archivo en vez de reemplazarlo: así el log de la
    ingesta conserva la primera indexación Y la segunda ejecución que no re-indexa.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if archivo:
        RESULTADOS_DIR.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(RESULTADOS_DIR / archivo, mode="a" if acumular else "w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for ruidoso in ("httpx", "httpcore", "urllib3", "google_genai", "pinecone", "pypdf"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
