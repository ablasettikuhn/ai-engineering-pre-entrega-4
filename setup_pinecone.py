"""
setup_pinecone.py
-----------------
Paso 1: prepara la infraestructura en la nube.

    - Si el índice INDEX_NAME no existe, lo crea en modo Serverless
      (AWS us-east-1, métrica coseno, dimensión EMBEDDINGS_DIM).
    - Si ya existe, verifica que su dimensión y su métrica coincidan con las del
      modelo de embeddings. Así se evita el error clásico de subir vectores de
      1536 dimensiones a un índice creado con 768 (o al revés).
    - Muestra el estado del índice: host, cantidad de vectores por namespace.

La ingesta y la consulta llaman a `abrir_indice()`, que hace la misma verificación.

Uso:
    python setup_pinecone.py
"""

import logging
import warnings

from pinecone import Pinecone, ServerlessSpec

import config

logger = logging.getLogger("setup")


class DimensionIncompatibleError(RuntimeError):
    """El índice existe pero fue creado para otro tamaño de vector u otra métrica."""


def cliente() -> Pinecone:
    return Pinecone(api_key=config.clave_pinecone())


def _dimension_y_metrica(descripcion) -> tuple[int | None, str | None]:
    # En el SDK 10, `dimension` sigue disponible pero se marca como obsoleta
    # (los índices nuevos describen sus campos en un `schema`). Se lee sin avisos.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dimension = getattr(descripcion, "dimension", None)
        metrica = getattr(descripcion, "metric", None)
    return dimension, (str(metrica).lower() if metrica else None)


def asegurar_indice(pc: Pinecone | None = None, crear: bool = True):
    """Devuelve el índice listo para usar. Lo crea si no existe (y `crear` es True)."""
    pc = pc or cliente()

    if pc.has_index(config.INDEX_NAME):
        logger.info("El índice '%s' ya existe.", config.INDEX_NAME)
    elif crear:
        logger.info(
            "El índice '%s' no existe: creándolo (Serverless %s/%s, dimensión %d, métrica %s)...",
            config.INDEX_NAME, config.PINECONE_CLOUD, config.PINECONE_REGION,
            config.EMBEDDINGS_DIM, config.PINECONE_METRIC,
        )
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBEDDINGS_DIM,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            deletion_protection="disabled",
            tags={"proyecto": "pre-entrega-4", "embeddings": config.EMBEDDINGS_MODEL},
        )  # bloquea hasta que el índice está listo
        logger.info("Índice creado y listo.")
    else:
        raise RuntimeError(f"El índice '{config.INDEX_NAME}' no existe. Ejecuta: python setup_pinecone.py")

    descripcion = pc.describe_index(config.INDEX_NAME)
    dimension, metrica = _dimension_y_metrica(descripcion)
    if dimension is not None and dimension != config.EMBEDDINGS_DIM:
        raise DimensionIncompatibleError(
            f"El índice '{config.INDEX_NAME}' tiene dimensión {dimension}, pero los embeddings "
            f"({config.EMBEDDINGS_MODEL}) tienen {config.EMBEDDINGS_DIM}. Usa otro INDEX_NAME "
            f"o cambia EMBEDDINGS_DIM a {dimension} en el .env."
        )
    if metrica and metrica != config.PINECONE_METRIC:
        raise DimensionIncompatibleError(
            f"El índice '{config.INDEX_NAME}' usa la métrica '{metrica}' y este proyecto espera "
            f"'{config.PINECONE_METRIC}'. Usa otro INDEX_NAME."
        )
    return pc.Index(host=descripcion.host)


def abrir_indice():
    """Para la consulta y la evaluación: abre el índice sin crearlo."""
    return asegurar_indice(crear=False)


def vectores_en_namespace(indice, namespace: str = config.PINECONE_NAMESPACE) -> int:
    resumen = indice.describe_index_stats().namespaces.get(namespace)
    return resumen.vector_count if resumen else 0


def mostrar_estado(indice) -> None:
    stats = indice.describe_index_stats()
    logger.info("Estado del índice '%s':", config.INDEX_NAME)
    logger.info("  dimensión: %s | métrica: %s | vectores totales: %d",
                stats.dimension, stats.metric or config.PINECONE_METRIC, stats.total_vector_count)
    if not stats.namespaces:
        logger.info("  (sin namespaces todavía: ejecuta python ingesta.py)")
    for nombre, resumen in sorted(stats.namespaces.items()):
        marca = "  <- namespace de este proyecto" if nombre == config.PINECONE_NAMESPACE else ""
        logger.info("  namespace '%s': %d vectores%s", nombre, resumen.vector_count, marca)


def main() -> None:
    config.configurar_logging("log_setup.txt")
    indice = asegurar_indice()
    mostrar_estado(indice)


if __name__ == "__main__":
    main()
