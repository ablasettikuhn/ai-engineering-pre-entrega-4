"""
descargar_papers.py
-------------------
Paso 0: arma el dataset de /data a partir de papers científicos reales.

Para cada paper de `fuentes.json`:
    enlace -> descarga el PDF -> extrae el texto por página (pypdf) -> limpia -> data/<id>.md

Cada .md lleva una cabecera con los metadatos y marcas `<!-- pagina N -->`, que la
ingesta usa para guardar la página de cada fragmento en Pinecone.

Los .md ya están en el repositorio (los papers tienen licencia CC BY, que permite
redistribuirlos citando la fuente), así que este script solo hace falta para
regenerarlos o para agregar papers nuevos al catálogo.

Uso:
    python descargar_papers.py            # genera solo los .md que faltan
    python descargar_papers.py --forzar   # vuelve a generar todos
"""

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from statistics import median

import httpx
from pypdf import PdfReader

from config import DATA_DIR, FUENTES, PDF_DIR, configurar_logging

logger = logging.getLogger("descarga")

# Algunos sitios rechazan clientes sin un User-Agent de navegador.
CABECERAS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
}

TITULOS_REFERENCIAS = re.compile(r"^\s*(references|bibliography|referencias)\s*$", re.IGNORECASE)
# Marcas técnicas de los PDF de PLOS (no son texto del artículo).
LINEAS_BASURA = re.compile(r"^(a1111111111\s*)+$")


def normalizar_texto(texto: str) -> str:
    """NFKC convierte las ligaduras tipográficas del PDF ("ﬁ", "ﬂ") en letras normales.
    Sin esto, "inﬂuence" y "influence" serían palabras distintas para BM25."""
    return unicodedata.normalize("NFKC", texto)


# ---------------------------------------------------------------------------
# 1) Descarga
# ---------------------------------------------------------------------------
def descargar_pdf(fuente: dict) -> bool:
    destino = PDF_DIR / f"{fuente['id']}.pdf"
    if destino.exists():
        logger.info("[%s] PDF ya descargado: %s", fuente["id"], destino.name)
        return True

    logger.info("[%s] Descargando %s", fuente["id"], fuente["url_pdf"])
    try:
        r = httpx.get(fuente["url_pdf"], headers=CABECERAS, follow_redirects=True, timeout=60)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("[%s] No se pudo descargar: %s", fuente["id"], exc)
        _explicar_descarga_manual(fuente, destino)
        return False

    if not r.content.startswith(b"%PDF"):
        logger.error("[%s] El enlace no devolvió un PDF (¿página de verificación del sitio?)", fuente["id"])
        _explicar_descarga_manual(fuente, destino)
        return False

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    logger.info("[%s] PDF guardado (%d KB)", fuente["id"], len(r.content) // 1024)
    return True


def _explicar_descarga_manual(fuente: dict, destino) -> None:
    logger.error(
        "[%s] Descárgalo desde el navegador (https://doi.org/%s), guárdalo como %s "
        "y vuelve a ejecutar este script.",
        fuente["id"], fuente["doi"], destino,
    )


# ---------------------------------------------------------------------------
# 2) Extracción y limpieza del texto
# ---------------------------------------------------------------------------
def _clave_linea(linea: str) -> str:
    """Normaliza una línea para detectar encabezados/pies repetidos (ignora números)."""
    return re.sub(r"\d+", "#", linea.strip().lower())


def extraer_paginas(ruta_pdf) -> list[str]:
    paginas = [normalizar_texto(p.extract_text() or "") for p in PdfReader(ruta_pdf).pages]

    # Encabezados y pies de página: líneas del borde de la página (3 primeras y 3 últimas)
    # que se repiten en muchas páginas (ej. "Frontiers in Physiology | www.frontiersin.org 3").
    def bordes(texto: str) -> set[str]:
        lineas = [l for l in texto.splitlines() if l.strip()]
        return {_clave_linea(l) for l in lineas[:3] + lineas[-3:]}

    conteo = Counter(clave for texto in paginas for clave in bordes(texto))
    umbral = max(3, int(len(paginas) * 0.4))
    repetidas = {clave for clave, n in conteo.items() if n >= umbral and len(clave) < 120}

    lineas_por_pagina = [
        [
            l.strip()
            for l in texto.splitlines()
            if l.strip() and _clave_linea(l) not in repetidas and not LINEAS_BASURA.match(l.strip())
        ]
        for texto in paginas
    ]
    return [_unir_parrafos(lineas) for lineas in _cortar_referencias(lineas_por_pagina)]


def _cortar_referencias(paginas: list[list[str]]) -> list[list[str]]:
    """Elimina la bibliografía (desde el título 'References'): es ruido para la búsqueda."""
    for i in range(len(paginas) - 1, len(paginas) // 2 - 1, -1):
        for j, linea in enumerate(paginas[i]):
            if TITULOS_REFERENCIAS.match(linea):
                return paginas[:i] + [paginas[i][:j]]
    return paginas


def _unir_parrafos(lineas: list[str]) -> str:
    """Une las líneas cortadas por el ancho de columna y repara palabras con guion."""
    if not lineas:
        return ""
    largo_tipico = median(len(l) for l in lineas)
    parrafos, actual = [], ""
    for linea in lineas:
        if actual.endswith("-") and linea[:1].islower():
            actual = actual[:-1] + linea  # "valid-" + "ity" -> "validity"
        else:
            actual = f"{actual} {linea}".strip()
        # Fin de párrafo: línea corta que termina en puntuación final.
        if linea.endswith((".", ":", "?")) and len(linea) < 0.75 * largo_tipico:
            parrafos.append(actual)
            actual = ""
    if actual:
        parrafos.append(actual)
    return "\n\n".join(parrafos)


# ---------------------------------------------------------------------------
# 3) Escritura del .md con metadatos
# ---------------------------------------------------------------------------
def escribir_markdown(fuente: dict, paginas: list[str]) -> None:
    cabecera = [
        "---",
        f"id: {fuente['id']}",
        f"titulo: {fuente['titulo']}",
        f"autores: {fuente['autores']}",
        f"anio: {fuente['anio']}",
        f"revista: {fuente['revista']}",
        f"doi: {fuente['doi']}",
        f"url: https://doi.org/{fuente['doi']}",
        f"licencia: {fuente['licencia']}",
        f"categorias: {', '.join(fuente['categorias'])}",
        "---",
        "",
    ]
    cuerpo = [f"<!-- pagina {n} -->\n{texto}\n" for n, texto in enumerate(paginas, start=1) if texto.strip()]
    destino = DATA_DIR / f"{fuente['id']}.md"
    destino.write_text("\n".join(cabecera) + "\n".join(cuerpo), encoding="utf-8")
    palabras = sum(len(t.split()) for t in paginas)
    logger.info("[%s] Guardado %s (%d páginas, ~%d palabras)", fuente["id"], destino.name, len(cuerpo), palabras)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera data/*.md a partir de papers científicos.")
    parser.add_argument("--forzar", action="store_true", help="regenera los .md aunque ya existan")
    args = parser.parse_args()

    configurar_logging("log_descarga.txt")
    DATA_DIR.mkdir(exist_ok=True)
    fuentes = json.loads(FUENTES.read_text(encoding="utf-8"))

    faltantes = []
    for fuente in fuentes:
        if (DATA_DIR / f"{fuente['id']}.md").exists() and not args.forzar:
            logger.info("[%s] Ya existe data/%s.md: se omite", fuente["id"], fuente["id"])
            continue
        if descargar_pdf(fuente):
            escribir_markdown(fuente, extraer_paginas(PDF_DIR / f"{fuente['id']}.pdf"))
        else:
            faltantes.append(fuente["id"])

    listos = len(fuentes) - len(faltantes)
    logger.info("Dataset: %d de %d papers listos en /data", listos, len(fuentes))
    if faltantes:
        logger.warning("Faltan: %s (ver las instrucciones de descarga manual más arriba)", ", ".join(faltantes))


if __name__ == "__main__":
    main()
