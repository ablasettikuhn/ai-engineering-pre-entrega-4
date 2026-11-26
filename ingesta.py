"""
ingesta.py
----------
Paso 2: pipeline de ingesta  data/*.md  ->  chunks  ->  embeddings  ->  Pinecone.

    1. Lee los papers de /data (texto + marcas de página) y sus metadatos de fuentes.json.
    2. Los divide con RecursiveCharacterTextSplitter, midiendo el tamaño en TOKENS
       (700 tokens con 100 de solapamiento por defecto: el punto medio de 500-800).
    3. Calcula los embeddings con Gemini (gemini-embedding-001, 1536 dimensiones).
    4. Hace upsert en el índice Serverless, dentro del namespace del proyecto.

Ingesta inteligente: cada registro de Pinecone guarda, en su metadata, el TEXTO
ORIGINAL del fragmento junto con la fuente, la página y las categorías. Una consulta
devuelve todo lo necesario para citar, sin otra base de datos.

Idempotencia: los IDs son deterministas ("haddad-2017-c004"), y una huella del
dataset (archivos + chunking + embeddings) se guarda en resultados/manifest_ingesta.json.
Si nada cambió y Pinecone tiene los mismos vectores, NO se re-indexa (ahorra cuota).

Uso:
    python ingesta.py            # indexa solo si hace falta
    python ingesta.py --forzar   # vacía el namespace y re-indexa todo
"""

import argparse
import bisect
import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import setup_pinecone

logger = logging.getLogger("ingesta")

MARCA_PAGINA = re.compile(r"<!-- pagina (\d+) -->\n?")
EXTENSIONES = {".md", ".txt"}
LOTE_UPSERT = 100  # Pinecone recomienda hasta 100 vectores por request


# ---------------------------------------------------------------------------
# 1) Lectura de documentos
# ---------------------------------------------------------------------------
def archivos_dataset() -> list[Path]:
    if not config.DATA_DIR.exists():
        return []
    return sorted(p for p in config.DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)


def cargar_fuentes() -> dict[str, dict]:
    """fuentes.json es la fuente de verdad de los metadatos (título, autores, categorías...)."""
    if not config.FUENTES.exists():
        return {}
    return {f["id"]: f for f in json.loads(config.FUENTES.read_text(encoding="utf-8"))}


def _leer_cabecera(texto: str) -> tuple[dict, str]:
    """Separa la cabecera '--- clave: valor ---' del cuerpo del documento."""
    if not texto.startswith("---\n"):
        return {}, texto
    fin = texto.index("\n---\n", 4)
    metadatos = {}
    for linea in texto[4:fin].splitlines():
        clave, _, valor = linea.partition(":")
        metadatos[clave.strip()] = valor.strip()
    return metadatos, texto[fin + 5 :]


def leer_documento(ruta: Path, fuentes: dict[str, dict]) -> dict:
    """Devuelve el texto limpio (sin marcas de página), dónde empieza cada página y los metadatos."""
    cabecera, cuerpo = _leer_cabecera(ruta.read_text(encoding="utf-8"))
    cuerpo = unicodedata.normalize("NFKC", cuerpo)  # ligaduras "ﬁ" -> "fi" (importante para BM25)

    partes, inicios, numeros, cursor = [], [], [], 0
    trozos = MARCA_PAGINA.split(cuerpo)  # [antes, n1, texto1, n2, texto2, ...]
    if trozos[0].strip():
        partes.append(trozos[0])
        inicios.append(0)
        numeros.append(1)
        cursor += len(trozos[0])
    for numero, contenido in zip(trozos[1::2], trozos[2::2]):
        inicios.append(cursor)
        numeros.append(int(numero))
        partes.append(contenido)
        cursor += len(contenido)

    doc_id = cabecera.get("id", ruta.stem)
    fuente = fuentes.get(doc_id, {})
    categorias = fuente.get("categorias") or [c.strip() for c in cabecera.get("categorias", "").split(",") if c.strip()]
    anio = fuente.get("anio") or cabecera.get("anio", "0")
    doi = fuente.get("doi") or cabecera.get("doi", "")
    metadatos = {
        "doc_id": doc_id,
        "fuente": ruta.name,
        "titulo": fuente.get("titulo") or cabecera.get("titulo", ruta.stem),
        "autores": fuente.get("autores") or cabecera.get("autores", "desconocido"),
        "anio": int(anio) if str(anio).isdigit() else 0,
        "revista": fuente.get("revista") or cabecera.get("revista", ""),
        "doi": doi,
        "url": f"https://doi.org/{doi}" if doi else cabecera.get("url", ""),
        "licencia": fuente.get("licencia") or cabecera.get("licencia", ""),
        "categorias": categorias or ["sin_categoria"],
    }
    return {"texto": "".join(partes), "inicios": inicios, "numeros": numeros, "metadatos": metadatos}


# ---------------------------------------------------------------------------
# 2) Chunking con número de página
# ---------------------------------------------------------------------------
def crear_splitter() -> RecursiveCharacterTextSplitter:
    # from_tiktoken_encoder mide el tamaño en TOKENS (no en caracteres).
    # Corta primero por párrafos, luego por líneas, oraciones y palabras.
    # keep_separator="end" deja el punto al final de la oración (no al inicio del chunk siguiente).
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )


def _pagina_en(posicion: int, inicios: list[int], numeros: list[int]) -> int:
    i = bisect.bisect_right(inicios, posicion) - 1
    return numeros[max(i, 0)] if numeros else 0


def fragmentar(documento: dict, splitter, encoder) -> list[dict]:
    """Divide un documento en chunks. Cada chunk sabe en qué página empieza y termina.

    La página se calcula buscando la posición exacta del chunk en el texto original
    (el splitter devuelve trozos literales del texto). Se busca desde la posición del
    chunk anterior, porque el solapamiento hace que se repitan frases.
    """
    texto, meta = documento["texto"], documento["metadatos"]
    chunks, cursor = [], 0
    for n, trozo in enumerate(splitter.split_text(texto)):
        inicio = texto.find(trozo, cursor)
        if inicio == -1:  # no debería pasar; por seguridad se busca desde el principio
            inicio = max(texto.find(trozo), 0)
        fin = inicio + len(trozo) - 1
        cursor = inicio + 1
        chunks.append(
            {
                "id": f"{meta['doc_id']}-c{n:03d}",
                "texto": trozo,
                "metadata": {
                    **meta,
                    "chunk": n,
                    "pagina": _pagina_en(inicio, documento["inicios"], documento["numeros"]),
                    "pagina_fin": _pagina_en(fin, documento["inicios"], documento["numeros"]),
                    "tokens": len(encoder.encode(trozo)),
                },
            }
        )
    return chunks


def construir_registro(chunk: dict, vector: list[float]) -> dict:
    """Esquema del registro en Pinecone: id + vector + metadata (con el texto original)."""
    return {"id": chunk["id"], "values": vector, "metadata": {"texto": chunk["texto"], **chunk["metadata"]}}


# ---------------------------------------------------------------------------
# 3) Embeddings y upsert
# ---------------------------------------------------------------------------
def _lotes_por_tokens(chunks: list[dict]) -> list[list[dict]]:
    """Agrupa los chunks para no superar el límite de tokens por minuto del plan gratuito."""
    lotes, actual, tokens = [], [], 0
    for c in chunks:
        t = c["metadata"]["tokens"]
        if actual and tokens + t > config.EMBEDDINGS_TOKENS_POR_MINUTO:
            lotes.append(actual)
            actual, tokens = [], 0
        actual.append(c)
        tokens += t
    if actual:
        lotes.append(actual)
    return lotes


def _embeber_con_reintentos(embeddings, textos: list[str], intentos: int = 3) -> list[list[float]]:
    for intento in range(1, intentos + 1):
        try:
            return embeddings.embed_documents(textos)
        except Exception as exc:  # 429 = límite de cuota del plan gratuito de Gemini
            cuota = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if cuota and intento < intentos:
                logger.warning("Límite de cuota de embeddings (429). Espero 60 s y reintento (%d/%d)", intento, intentos)
                time.sleep(60)
            else:
                raise
    raise RuntimeError("inalcanzable")


def _esperar_conteo(indice, esperado: int, segundos: int = 60) -> int:
    """Pinecone Serverless es eventualmente consistente: el conteo tarda unos segundos en actualizarse."""
    total = setup_pinecone.vectores_en_namespace(indice)
    limite = time.monotonic() + segundos
    while total != esperado and time.monotonic() < limite:
        time.sleep(3)
        total = setup_pinecone.vectores_en_namespace(indice)
    return total


def huella_dataset() -> str:
    """Cambia si cambia cualquier archivo, el catálogo, el chunking o el modelo de embeddings."""
    h = hashlib.sha256()
    for ruta in archivos_dataset() + ([config.FUENTES] if config.FUENTES.exists() else []):
        h.update(ruta.name.encode())
        h.update(ruta.read_bytes())
    parametros = {
        "chunk": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "indice": config.INDEX_NAME,
        "namespace": config.PINECONE_NAMESPACE,
        **config.firma_embeddings(),
    }
    h.update(json.dumps(parametros, sort_keys=True).encode())
    return h.hexdigest()


def leer_manifest() -> dict | None:
    if config.MANIFEST.exists():
        return json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    return None


def ingestar(forzar: bool = False, indice=None, embeddings=None) -> int:
    """Indexa /data en Pinecone si hace falta. Devuelve la cantidad de vectores del namespace."""
    if not archivos_dataset():
        raise RuntimeError(f"No hay archivos .md/.txt en {config.DATA_DIR}. Ejecuta antes descargar_papers.py.")

    indice = indice or setup_pinecone.asegurar_indice()
    ns = config.PINECONE_NAMESPACE
    huella = huella_dataset()
    manifest = leer_manifest()
    en_namespace = setup_pinecone.vectores_en_namespace(indice, ns)

    if not forzar and manifest and manifest.get("huella") == huella and en_namespace == manifest.get("chunks"):
        logger.info("El namespace '%s' ya está al día (%d vectores): no se re-indexa.", ns, en_namespace)
        return en_namespace

    motivo = "re-indexación forzada" if forzar else ("primera indexación" if not en_namespace else "el dataset o la configuración cambió")
    logger.info("Indexando en '%s' / namespace '%s' (%s)...", config.INDEX_NAME, ns, motivo)
    config.MANIFEST.unlink(missing_ok=True)  # si la ingesta se corta, la próxima ejecución re-indexa
    if en_namespace:
        logger.info("Vaciando el namespace '%s' (%d vectores viejos)...", ns, en_namespace)
        indice.delete(delete_all=True, namespace=ns)
        _esperar_conteo(indice, 0)

    # 1-2) Lectura y chunking
    fuentes = cargar_fuentes()
    splitter, encoder = crear_splitter(), tiktoken.get_encoding("cl100k_base")
    chunks = []
    for ruta in archivos_dataset():
        documento = leer_documento(ruta, fuentes)
        propios = fragmentar(documento, splitter, encoder)
        chunks.extend(propios)
        tokens = [c["metadata"]["tokens"] for c in propios]
        logger.info(
            "  %-24s -> %3d chunks (tokens: media %d, máx %d) | categorías: %s",
            ruta.name, len(propios), sum(tokens) // max(len(tokens), 1), max(tokens, default=0),
            ", ".join(documento["metadatos"]["categorias"]),
        )
    logger.info("Total: %d chunks de %d documentos", len(chunks), len(archivos_dataset()))

    # 3-4) Embeddings + upsert, por lotes que respetan la cuota de Gemini
    embeddings = embeddings or config.obtener_embeddings()
    lotes = _lotes_por_tokens(chunks)
    inicio = time.perf_counter()
    for i, lote in enumerate(lotes, start=1):
        if i > 1:
            logger.info("Pausa de 60 s para respetar el límite de tokens por minuto de los embeddings...")
            time.sleep(60)
        vectores = _embeber_con_reintentos(embeddings, [c["texto"] for c in lote])
        if any(len(v) != config.EMBEDDINGS_DIM for v in vectores):
            raise setup_pinecone.DimensionIncompatibleError(
                f"El modelo devolvió vectores de {len(vectores[0])} dimensiones y el índice espera {config.EMBEDDINGS_DIM}."
            )
        registros = [construir_registro(c, v) for c, v in zip(lote, vectores)]
        for j in range(0, len(registros), LOTE_UPSERT):
            indice.upsert(vectors=registros[j : j + LOTE_UPSERT], namespace=ns, show_progress=False)
        logger.info("Lote %d/%d: %d chunks embebidos y subidos a Pinecone", i, len(lotes), len(lote))

    total = _esperar_conteo(indice, len(chunks))
    if total != len(chunks):
        logger.warning("Pinecone todavía informa %d de %d vectores (consistencia eventual). Vuelve a "
                       "ejecutar python setup_pinecone.py en unos segundos para confirmarlo.", total, len(chunks))

    ejemplo = construir_registro(chunks[0], [0.0])["metadata"]
    ejemplo["texto"] = ejemplo["texto"][:120] + "..."
    logger.info("Ejemplo de metadata de un registro (%s):\n%s", chunks[0]["id"], json.dumps(ejemplo, ensure_ascii=False, indent=2))

    config.MANIFEST.parent.mkdir(exist_ok=True)
    config.MANIFEST.write_text(
        json.dumps(
            {
                "huella": huella,
                "indice": config.INDEX_NAME,
                "namespace": ns,
                "chunks": len(chunks),
                **config.firma_embeddings(),
                "chunk_size_tokens": config.CHUNK_SIZE,
                "chunk_overlap_tokens": config.CHUNK_OVERLAP,
                "documentos": [p.name for p in archivos_dataset()],
                "fecha": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Ingesta completa: %d vectores en el namespace '%s' (%.1f s)", total, ns, time.perf_counter() - inicio)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa /data en Pinecone Serverless.")
    parser.add_argument("--forzar", action="store_true", help="vacía el namespace y re-indexa todo")
    args = parser.parse_args()
    config.configurar_logging("log_ingesta.txt", acumular=True)
    logger.info("===== Ejecución: python ingesta.py%s =====", " --forzar" if args.forzar else "")
    ingestar(forzar=args.forzar)


if __name__ == "__main__":
    main()
