"""
consultar.py
------------
Uso real del sistema: un preparador físico hace una pregunta y recibe los 5
fragmentos más relevantes (paper, página, categorías y qué búsqueda los encontró).
Antes de mostrarlos, indica si la biblioteca tiene evidencia sobre el tema
(similitud máxima frente a UMBRAL_SIMILITUD).

Uso:
    python consultar.py "¿Cómo se calcula la carga con el método session-RPE?"
    python consultar.py "..." --modo vectorial          # compara: solo vectores
    python consultar.py "..." --modo bm25               # compara: solo BM25
    python consultar.py "..." --categoria frecuencia_cardiaca
    python consultar.py "..." --responder               # además genera una respuesta citada con Gemini
    python consultar.py "..." --guardar                 # guarda la salida en resultados/ejemplo_consulta.txt
    python consultar.py                                 # modo interactivo (escribe 'salir' para terminar)
"""

import argparse
import textwrap

import config
from rag_system import MODOS, RAGSystem


def consultar(rag: RAGSystem, pregunta: str, modo: str, categoria: str | None, responder: bool) -> str:
    lineas = [f"Pregunta: {pregunta}",
              f"Modo: {modo} | top-{rag.k}" + (f" | categoría: {categoria}" if categoria else "")]
    evidencia, similitud = rag.hay_evidencia(pregunta, categoria)
    estado = "hay evidencia" if evidencia else "SIN EVIDENCIA SUFICIENTE en la biblioteca"
    lineas.append(f"Similitud máxima: {similitud:.3f} (umbral {rag.umbral:.2f}) -> {estado}")
    lineas.append("-" * 78)

    if responder:
        salida = rag.responder(pregunta, categoria=categoria)
        lineas.append(textwrap.fill(salida["respuesta"], width=78, replace_whitespace=False))
        lineas.append("-" * 78)
        documentos = salida["fuentes"]
    else:
        documentos = rag.buscar(pregunta, modo=modo, categoria=categoria)
        if not evidencia:
            lineas.append("Aviso: estos fragmentos son los más parecidos, pero probablemente no responden la pregunta.")

    for d in documentos:
        m = d.metadata
        paginas = f"p. {m['pagina']}" + (f"-{m['pagina_fin']}" if m["pagina_fin"] != m["pagina"] else "")
        autor = m["autores"].split(",")[0]
        lineas.append(f"[{m['posicion']}] {m['doc_id']} | {autor} et al. ({m['anio']}) | {paginas} | "
                      f"encontrado por: {m['encontrado_por']}")
        lineas.append(f"    categorías: {', '.join(m.get('categorias', []))} | doi: {m['doi']}")
        lineas.append(textwrap.indent(textwrap.shorten(d.page_content, width=300, placeholder=" [...]"), "    "))
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el buscador de evidencia (RAG híbrido en Pinecone).")
    parser.add_argument("pregunta", nargs="?", help="pregunta (si no se indica, modo interactivo)")
    parser.add_argument("--modo", choices=MODOS, default="hibrido")
    parser.add_argument("--categoria", help="filtra por una categoría de la metadata")
    parser.add_argument("--responder", action="store_true", help="genera una respuesta citada con Gemini")
    parser.add_argument("--guardar", action="store_true", help="agrega la salida a resultados/ejemplo_consulta.txt")
    args = parser.parse_args()

    config.configurar_logging()
    rag = RAGSystem()
    print(f"Categorías disponibles: {', '.join(rag.categorias)}")

    preguntas = [args.pregunta] if args.pregunta else None
    while True:
        pregunta = preguntas.pop() if preguntas else (None if args.pregunta else input("\nPregunta ('salir' para terminar): ").strip())
        if not pregunta or pregunta.lower() in {"salir", "exit", "q"}:
            break
        texto = consultar(rag, pregunta, args.modo, args.categoria, args.responder)
        print("\n" + texto + "\n")
        if args.guardar:
            config.RESULTADOS_DIR.mkdir(exist_ok=True)
            with open(config.RESULTADOS_DIR / "ejemplo_consulta.txt", "a", encoding="utf-8") as f:
                f.write(f"$ python consultar.py \"{pregunta}\" --modo {args.modo}"
                        + (f" --categoria {args.categoria}" if args.categoria else "")
                        + (" --responder" if args.responder else "") + "\n" + texto + "\n\n")


if __name__ == "__main__":
    main()
