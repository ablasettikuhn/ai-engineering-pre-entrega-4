"""
rag_system.py
-------------
Paso 3: el Recuperador Híbrido.

    RAGSystem
     └── EnsembleRetriever (LangChain)  -> fusiona con Reciprocal Rank Fusion (RRF)
          ├── BM25Retriever             -> búsqueda LÉXICA: coincidencia exacta de palabras
          │                                (siglas como sRPE, HRV, GNSS; nombres como Foster, Hooper)
          └── PineconeRetriever         -> búsqueda SEMÁNTICA: similitud de vectores en Pinecone
                                           (entiende la pregunta en español aunque los papers estén en inglés)

Por qué combinar los dos: los embeddings capturan el significado, pero pueden
"diluir" un término técnico exacto; BM25 encuentra el término exacto, pero no
entiende sinónimos ni otro idioma. El ensemble aprovecha lo mejor de cada uno.

El corpus de BM25 se arma leyendo los registros de Pinecone (el texto original está
en la metadata), así que BM25 y la búsqueda vectorial ven exactamente los mismos
chunks, con los mismos IDs, sin una base de datos adicional.
"""

import logging
import re
import unicodedata
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

import config
import setup_pinecone

logger = logging.getLogger("rag")

MODOS = ("hibrido", "vectorial", "bm25")
NO_LO_SE = "No lo sé con los papers disponibles."
LOTE_FETCH = 100

# ---------------------------------------------------------------------------
# Tokenizador para BM25
# ---------------------------------------------------------------------------
# Palabras vacías en español (las preguntas) e inglés (los papers): no aportan a la búsqueda.
STOPWORDS = set(
    """
    a al algo algun alguna algunas alguno algunos ante antes aqui asi cada como con contra cual cuales
    cuando de del desde donde durante e el ella ellas ellos en entre era es esa esas ese eso esos esta
    estan estas este esto estos fue fueron ha han hay la las le les lo los mas me mi mis mucho muy
    ni no nos o otra otras otro otros para pero poco por porque puede pueden que quien se segun ser si
    sin sirve sirven sobre son su sus tambien tan tanto te tiene tienen todo todos tu un una unas uno
    unos y ya cuanto cuanta cuantos cuantas hace hacer usar usando utilizar
    about after all also an and any are as at be been being between both but by can could did do does
    during each for from had has have he her his how if in into is it its may more most no not of on
    only or other our out over she should so some such than that the their them then there these they
    this those through to too under up use used using very was we were what when where which while who
    will with within would you
    """.split()
)


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin tildes, sin palabras vacías. Conserva siglas y términos con guion.

    "session-RPE" -> ["session-rpe", "session", "rpe"]: así coincide aunque el paper
    escriba "session RPE" o "session-RPE".
    """
    texto = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    tokens = []
    for palabra in re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", texto):
        partes = re.split(r"[-/]", palabra)
        candidatos = [palabra] + partes if len(partes) > 1 else [palabra]
        tokens.extend(t for t in candidatos if t not in STOPWORDS and (len(t) > 1 or t.isdigit()))
    return tokens


# ---------------------------------------------------------------------------
# De registro de Pinecone a Document de LangChain
# ---------------------------------------------------------------------------
def registro_a_documento(id_: str, metadata: dict | None, score: float | None = None) -> Document:
    metadata = dict(metadata or {})
    texto = metadata.pop("texto", "")
    metadata["chunk_id"] = id_
    if score is not None:
        metadata["score_vectorial"] = round(float(score), 4)
    return Document(page_content=texto, metadata=metadata, id=id_)


def _ids_de_pagina(pagina) -> list[str]:
    """`index.list()` devuelve páginas de IDs (objetos con .vectors en el SDK 10; listas en versiones viejas)."""
    if isinstance(pagina, (list, tuple)):
        return [getattr(x, "id", x) for x in pagina]
    return [item.id for item in pagina.vectors]


def cargar_corpus(indice, namespace: str = config.PINECONE_NAMESPACE) -> list[Document]:
    """Lee TODOS los chunks del namespace (ID + metadata con el texto) para construir BM25."""
    ids = [i for pagina in indice.list(namespace=namespace) for i in _ids_de_pagina(pagina)]
    documentos = []
    for j in range(0, len(ids), LOTE_FETCH):
        respuesta = indice.fetch(ids=ids[j : j + LOTE_FETCH], namespace=namespace)
        documentos.extend(registro_a_documento(id_, v.metadata) for id_, v in respuesta.vectors.items())
    documentos.sort(key=lambda d: d.metadata["chunk_id"])
    return documentos


# ---------------------------------------------------------------------------
# Recuperadores
# ---------------------------------------------------------------------------
class PineconeRetriever(BaseRetriever):
    """Búsqueda vectorial con el SDK nativo de Pinecone, expuesta como retriever de LangChain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    indice: Any
    embeddings: Embeddings
    namespace: str = config.PINECONE_NAMESPACE
    k: int = config.TOP_K
    filtro: dict | None = None
    _ultimos_ids: list[str] = PrivateAttr(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        respuesta = self.indice.query(
            vector=self.embeddings.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            filter=self.filtro,
            include_metadata=True,
        )
        documentos = [registro_a_documento(m.id, m.metadata, m.score) for m in respuesta.matches]
        self._ultimos_ids = [d.metadata["chunk_id"] for d in documentos]
        return documentos


class BM25ConRegistro(BM25Retriever):
    """BM25Retriever de LangChain que además recuerda qué chunks devolvió (para mostrar el origen)."""

    _ultimos_ids: list[str] = PrivateAttr(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        documentos = super()._get_relevant_documents(query, run_manager=run_manager)
        self._ultimos_ids = [d.metadata["chunk_id"] for d in documentos]
        return documentos


class EmbeddingsConCache(Embeddings):
    """Evita pagar dos veces el embedding de la misma pregunta (útil en la evaluación)."""

    def __init__(self, base: Embeddings):
        self.base, self._cache = base, {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.base.embed_query(text)
        return self._cache[text]


# ---------------------------------------------------------------------------
# RAGSystem
# ---------------------------------------------------------------------------
class RAGSystem:
    """Recibe una consulta y devuelve los top-k chunks combinando búsqueda léxica y semántica."""

    def __init__(
        self,
        indice=None,
        embeddings: Embeddings | None = None,
        namespace: str = config.PINECONE_NAMESPACE,
        k: int = config.TOP_K,
        k_candidatos: int = config.K_CANDIDATOS,
        peso_bm25: float = config.PESO_BM25,
        umbral: float = config.UMBRAL_SIMILITUD,
    ):
        self.indice = indice or setup_pinecone.abrir_indice()
        self.embeddings = EmbeddingsConCache(embeddings or config.obtener_embeddings())
        self.namespace, self.k, self.k_candidatos = namespace, k, k_candidatos
        self.pesos = [peso_bm25, round(1 - peso_bm25, 4)]
        self.umbral = umbral

        self.corpus = cargar_corpus(self.indice, namespace)
        if not self.corpus:
            raise RuntimeError(f"El namespace '{namespace}' está vacío. Ejecuta primero: python ingesta.py")
        self.categorias = sorted({c for d in self.corpus for c in d.metadata.get("categorias", [])})
        self.documentos = sorted({d.metadata["doc_id"] for d in self.corpus})
        logger.info("Corpus cargado desde Pinecone: %d chunks de %d papers (namespace '%s')",
                    len(self.corpus), len(self.documentos), namespace)

        # El EnsembleRetriever principal (sin filtro de categoría).
        self.bm25, self.vectorial = self._crear_recuperadores(categoria=None)
        self.ensemble = self._crear_ensemble(self.bm25, self.vectorial)

    # -- construcción ----------------------------------------------------------
    def _crear_recuperadores(self, categoria: str | None) -> tuple[BM25ConRegistro, PineconeRetriever]:
        corpus = self.corpus
        filtro = None
        if categoria:
            if categoria not in self.categorias:
                raise ValueError(f"Categoría desconocida '{categoria}'. Disponibles: {', '.join(self.categorias)}")
            corpus = [d for d in self.corpus if categoria in d.metadata.get("categorias", [])]
            filtro = {"categorias": {"$in": [categoria]}}  # filtro por metadata en Pinecone
        bm25 = BM25ConRegistro.from_documents(corpus, preprocess_func=tokenizar, k=self.k_candidatos)
        vectorial = PineconeRetriever(
            indice=self.indice, embeddings=self.embeddings, namespace=self.namespace,
            k=self.k_candidatos, filtro=filtro,
        )
        return bm25, vectorial

    def _crear_ensemble(self, bm25, vectorial) -> EnsembleRetriever:
        # id_key: dos resultados con el mismo chunk_id son el MISMO chunk (se fusionan sus puntajes).
        return EnsembleRetriever(retrievers=[bm25, vectorial], weights=self.pesos, id_key="chunk_id")

    # -- búsqueda --------------------------------------------------------------
    def buscar(self, consulta: str, modo: str = "hibrido", categoria: str | None = None) -> list[Document]:
        """Devuelve los top-k chunks. modo: 'hibrido' (por defecto), 'vectorial' o 'bm25'."""
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}")
        if categoria:
            bm25, vectorial = self._crear_recuperadores(categoria)
            ensemble = self._crear_ensemble(bm25, vectorial)
        else:
            bm25, vectorial, ensemble = self.bm25, self.vectorial, self.ensemble

        if modo == "bm25":
            documentos = bm25.invoke(consulta)
        elif modo == "vectorial":
            documentos = vectorial.invoke(consulta)
        else:
            documentos = ensemble.invoke(consulta)

        resultado = []
        for posicion, doc in enumerate(documentos[: self.k], start=1):
            cid = doc.metadata["chunk_id"]
            if modo == "hibrido":
                origen = [n for n, r in (("bm25", bm25), ("vectorial", vectorial)) if cid in r._ultimos_ids]
            else:
                origen = [modo]
            metadata = {**doc.metadata, "posicion": posicion, "encontrado_por": "+".join(origen)}
            resultado.append(Document(page_content=doc.page_content, metadata=metadata, id=cid))
        return resultado

    # -- ¿hay evidencia? (umbral mínimo de similitud) -------------------------
    def similitud_maxima(self, consulta: str, categoria: str | None = None) -> float:
        """Similitud coseno del chunk más parecido a la consulta (búsqueda vectorial, top-1).

        BM25 siempre devuelve algo (alguna palabra coincide), así que la decisión de
        "la biblioteca no tiene evidencia" se toma con la similitud semántica.
        """
        filtro = {"categorias": {"$in": [categoria]}} if categoria else None
        respuesta = self.indice.query(
            vector=self.embeddings.embed_query(consulta), top_k=1,
            namespace=self.namespace, filter=filtro, include_metadata=False,
        )
        return round(float(respuesta.matches[0].score), 4) if respuesta.matches else 0.0

    def hay_evidencia(self, consulta: str, categoria: str | None = None) -> tuple[bool, float]:
        similitud = self.similitud_maxima(consulta, categoria)
        return similitud >= self.umbral, similitud

    # -- generación (opcional) -----------------------------------------------
    def responder(self, pregunta: str, categoria: str | None = None) -> dict:
        """RAG completo: recupera los top-k chunks y le pide a Gemini una respuesta citada.

        Si la similitud máxima no alcanza el umbral, responde "No lo sé" SIN llamar al LLM:
        así no puede completar con conocimiento propio lo que los papers no dicen.
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        evidencia, similitud = self.hay_evidencia(pregunta, categoria)
        if not evidencia:
            return {"respuesta": NO_LO_SE, "fuentes": [], "similitud_maxima": similitud}

        documentos = self.buscar(pregunta, categoria=categoria)
        contexto = "\n\n".join(
            f"[{d.metadata['posicion']}] ({d.metadata['autores'].split(',')[0]} et al., {d.metadata['anio']}, "
            f"p. {d.metadata['pagina']})\n{d.page_content}"
            for d in documentos
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Eres un asistente para el staff de rendimiento de un club deportivo. Respondes en español "
                 "usando SOLO la información de los fragmentos de papers científicos que se te dan. "
                 "Cita cada dato con el número del fragmento entre corchetes, por ejemplo [2]. "
                 f"Si los fragmentos no contienen la respuesta, responde exactamente: \"{NO_LO_SE}\" "
                 "No inventes cifras ni referencias."),
                ("human", "Fragmentos:\n{contexto}\n\nPregunta: {pregunta}"),
            ]
        )
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.clave_gemini(),
                                     temperature=0, max_retries=3)
        respuesta = (prompt | llm | StrOutputParser()).invoke({"contexto": contexto, "pregunta": pregunta})
        return {"respuesta": respuesta, "fuentes": documentos, "similitud_maxima": similitud}
