"""
evaluate.py
-----------
Paso 4: evaluación del recuperador con un "Golden Set" (golden_set.json).

Cada pregunta del golden set indica el paper que DEBERÍA aparecer
(`documento_id_esperado`). Para cada pregunta se piden los Top-5 chunks y se calcula:

    Recall@5    ¿Está el documento correcto entre los 5 recuperados? (1 o 0)
                Con un solo documento esperado por pregunta, es la "tasa de acierto".
    Precision@5 ¿Qué porcentaje de los 5 chunks recuperados son realmente útiles?
                Un chunk es útil si viene del documento esperado (o de otro paper
                marcado como relevante en `documentos_relevantes`).
    MRR         (extra) 1 / posición del primer chunk del documento esperado.
                Premia que el documento correcto aparezca ARRIBA, no solo dentro del Top-5.

Se evalúan los tres modos (vectorial, BM25 e híbrido) para ver qué aporta la
búsqueda híbrida.

Control de preguntas trampa (preguntas_trampa.json): preguntas del mismo dominio
que la biblioteca NO responde. Se verifica que el umbral mínimo de similitud
(UMBRAL_SIMILITUD) las separe de las 5 preguntas del golden set: las del golden set
deben quedar por encima y las trampa por debajo. Si no, se sugiere un umbral.

El reporte se imprime en consola (con PASA / FALLA) y se guarda en resultados/.

Uso:
    python evaluate.py
    python evaluate.py --actualizar-readme   # además copia la tabla de resultados al README.md
"""

import argparse
import json
import logging
import re
from datetime import datetime

import config
from rag_system import MODOS, RAGSystem

logger = logging.getLogger("evaluacion")

MARCA_INICIO, MARCA_FIN = "<!-- METRICAS:INICIO -->", "<!-- METRICAS:FIN -->"


def cargar_golden_set() -> list[dict]:
    preguntas = json.loads(config.GOLDEN_SET.read_text(encoding="utf-8"))
    for p in preguntas:
        faltan = {"pregunta", "documento_id_esperado"} - p.keys()
        if faltan:
            raise ValueError(f"La entrada {p} del golden set no tiene: {', '.join(sorted(faltan))}")
    return preguntas


def evaluar_pregunta(ids_recuperados: list[str], esperado: str, relevantes: set[str], k: int) -> dict:
    top = ids_recuperados[:k]
    recall = 1.0 if esperado in top else 0.0
    precision = sum(1 for d in top if d in relevantes) / k
    rr = next((1 / pos for pos, d in enumerate(top, start=1) if d == esperado), 0.0)
    return {"recall": recall, "precision": precision, "rr": rr}


def cargar_trampas() -> list[dict]:
    if not config.PREGUNTAS_TRAMPA.exists():
        return []
    return json.loads(config.PREGUNTAS_TRAMPA.read_text(encoding="utf-8"))


def control_umbral(rag: RAGSystem, preguntas: list[dict], trampas: list[dict]) -> dict:
    """Similitud máxima de cada pregunta frente al umbral: las del golden set deben pasar, las trampa no."""
    filas = []
    for p in preguntas:
        s = rag.similitud_maxima(p["pregunta"])
        filas.append({"id": p.get("id", ""), "tipo": "golden", "pregunta": p["pregunta"], "similitud_maxima": s,
                      "esperado": "con evidencia", "ok": s >= rag.umbral})
    for t in trampas:
        s = rag.similitud_maxima(t["pregunta"])
        filas.append({"id": t.get("id", ""), "tipo": "trampa", "pregunta": t["pregunta"], "similitud_maxima": s,
                      "esperado": "sin evidencia", "ok": s < rag.umbral})
    min_golden = min((f["similitud_maxima"] for f in filas if f["tipo"] == "golden"), default=None)
    max_trampa = max((f["similitud_maxima"] for f in filas if f["tipo"] == "trampa"), default=None)
    separables = min_golden is not None and max_trampa is not None and min_golden > max_trampa
    return {
        "umbral": rag.umbral,
        "filas": filas,
        "min_similitud_golden": min_golden,
        "max_similitud_trampa": max_trampa,
        "umbral_sugerido": round((min_golden + max_trampa) / 2, 3) if separables else None,
        "pasa": bool(trampas) and all(f["ok"] for f in filas),
    }


def promedio(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def evaluar(rag: RAGSystem, preguntas: list[dict], trampas: list[dict] | None = None) -> dict:
    k = rag.k
    desconocidos = {p["documento_id_esperado"] for p in preguntas} - set(rag.documentos)
    if desconocidos:
        logger.warning("El golden set espera documentos que NO están en Pinecone: %s", ", ".join(sorted(desconocidos)))

    detalle = {modo: [] for modo in MODOS}
    for p in preguntas:
        esperado = p["documento_id_esperado"]
        relevantes = {esperado, *p.get("documentos_relevantes", [])}
        for modo in MODOS:
            documentos = rag.buscar(p["pregunta"], modo=modo)
            ids = [d.metadata["doc_id"] for d in documentos]
            detalle[modo].append(
                {
                    "id": p.get("id", ""),
                    "pregunta": p["pregunta"],
                    "tipo": p.get("tipo", ""),
                    "esperado": esperado,
                    "recuperados": [
                        {"chunk_id": d.metadata["chunk_id"], "doc_id": d.metadata["doc_id"],
                         "pagina": d.metadata["pagina"], "encontrado_por": d.metadata["encontrado_por"]}
                        for d in documentos
                    ],
                    **evaluar_pregunta(ids, esperado, relevantes, k),
                }
            )

    resumen = {
        modo: {
            f"recall@{k}": round(promedio([r["recall"] for r in filas]), 3),
            f"precision@{k}": round(promedio([r["precision"] for r in filas]), 3),
            "mrr": round(promedio([r["rr"] for r in filas]), 3),
        }
        for modo, filas in detalle.items()
    }
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "indice": config.INDEX_NAME,
        "namespace": rag.namespace,
        "k": k,
        "k_candidatos": rag.k_candidatos,
        "pesos_ensemble": {"bm25": rag.pesos[0], "vectorial": rag.pesos[1]},
        "chunks_en_corpus": len(rag.corpus),
        "papers_en_corpus": len(rag.documentos),
        "preguntas": len(preguntas),
        "resumen": resumen,
        "detalle": detalle,
        "control_umbral": control_umbral(rag, preguntas, trampas or []),
        "verificaciones": {
            "papers_del_golden_set_en_pinecone": not desconocidos,
        },
    }


def imprimir_reporte(reporte: dict) -> str:
    k = reporte["k"]
    lineas = [
        "=" * 78,
        f" EVALUACIÓN DEL RECUPERADOR  |  Precision@{k} y Recall@{k} (+ MRR)",
        f" Índice '{reporte['indice']}' / namespace '{reporte['namespace']}' | "
        f"{reporte['chunks_en_corpus']} chunks de {reporte['papers_en_corpus']} papers | {reporte['preguntas']} preguntas",
        f" Ensemble: BM25 {reporte['pesos_ensemble']['bm25']} + vectorial {reporte['pesos_ensemble']['vectorial']} "
        f"(cada uno aporta {reporte['k_candidatos']} candidatos; se devuelven {k})",
        "=" * 78,
        "",
        "DETALLE (modo híbrido)",
    ]
    for fila in reporte["detalle"]["hibrido"]:
        lineas.append(f"\n{fila['id']}. {fila['pregunta']}")
        lineas.append(f"    tipo: {fila['tipo']} | esperado: {fila['esperado']}")
        for pos, r in enumerate(fila["recuperados"], start=1):
            marca = "OK " if r["doc_id"] == fila["esperado"] else "   "
            lineas.append(f"    {marca}{pos}. {r['doc_id']:<20} p.{r['pagina']:<3} ({r['encontrado_por']})")
        lineas.append(f"    Recall@{k} = {fila['recall']:.0f} | Precision@{k} = {fila['precision']:.2f} | RR = {fila['rr']:.2f}")

    lineas += ["", f"ACIERTOS POR PREGUNTA (Recall@{k}: 1 = el paper esperado está en el Top-{k})",
               f"    {'pregunta':<9}" + "".join(f"{m:>12}" for m in MODOS)]
    for i, fila in enumerate(reporte["detalle"]["hibrido"]):
        valores = "".join(f"{reporte['detalle'][m][i]['recall']:>12.0f}" for m in MODOS)
        lineas.append(f"    {fila['id']:<9}{valores}")

    c = reporte["control_umbral"]
    lineas += ["", f"CONTROL DE PREGUNTAS TRAMPA (umbral mínimo de similitud = {c['umbral']:.2f})",
               f"    {'id':<5}{'tipo':<8}{'similitud':>10}  {'esperado':<15}resultado"]
    for f in c["filas"]:
        lineas.append(f"    {f['id']:<5}{f['tipo']:<8}{f['similitud_maxima']:>10.3f}  {f['esperado']:<15}"
                      f"{'PASA' if f['ok'] else 'FALLA'}")
    if c["min_similitud_golden"] is not None and c["max_similitud_trampa"] is not None:
        lineas.append(f"    Mínima del golden set: {c['min_similitud_golden']:.3f} | "
                      f"máxima de las trampa: {c['max_similitud_trampa']:.3f}")
    if c["umbral_sugerido"] is not None:
        lineas.append(f"    Umbral que separa ambos grupos (punto medio): {c['umbral_sugerido']:.3f}")
    else:
        lineas.append("    Ningún umbral separa ambos grupos: alguna trampa se parece tanto como una pregunta real.")

    lineas += ["", "VERIFICACIONES AUTOMÁTICAS",
               f"    [{'PASA' if reporte['verificaciones']['papers_del_golden_set_en_pinecone'] else 'FALLA'}] "
               "Los papers esperados del golden set están en Pinecone",
               f"    [{'PASA' if c['pasa'] else 'FALLA'}] El umbral separa las preguntas reales de las trampa"]
    if not c["pasa"] and c["umbral_sugerido"] is not None:
        lineas.append(f"           -> pon UMBRAL_SIMILITUD={c['umbral_sugerido']:.3f} en el .env y vuelve a ejecutar")

    lineas += ["", "RESUMEN (promedio de las preguntas)",
               f"    {'modo':<12}{'Recall@' + str(k):>12}{'Precision@' + str(k):>14}{'MRR':>8}"]
    for modo, m in reporte["resumen"].items():
        lineas.append(f"    {modo:<12}{m[f'recall@{k}']:>12.2f}{m[f'precision@{k}']:>14.2f}{m['mrr']:>8.2f}")
    lineas.append("=" * 78)
    texto = "\n".join(lineas)
    print(texto)
    return texto


def tabla_markdown(reporte: dict) -> str:
    k = reporte["k"]
    filas = [
        f"Resultados de `python evaluate.py` ({reporte['fecha'][:10]}): {reporte['preguntas']} preguntas, "
        f"{reporte['chunks_en_corpus']} chunks de {reporte['papers_en_corpus']} papers, "
        f"ensemble BM25 {reporte['pesos_ensemble']['bm25']} / vectorial {reporte['pesos_ensemble']['vectorial']}.",
        "",
        f"| Modo | Recall@{k} | Precision@{k} | MRR |",
        "|---|---|---|---|",
    ]
    for modo, m in reporte["resumen"].items():
        filas.append(f"| {modo} | {m[f'recall@{k}']:.2f} | {m[f'precision@{k}']:.2f} | {m['mrr']:.2f} |")
    filas += ["", f"| Pregunta | Paper esperado | " + " | ".join(f"Recall@{k} {m}" for m in MODOS) + " |",
              "|---|---|" + "---|" * len(MODOS)]
    for i, fila in enumerate(reporte["detalle"]["hibrido"]):
        valores = " | ".join(f"{reporte['detalle'][m][i]['recall']:.0f}" for m in MODOS)
        filas.append(f"| {fila['id']} | {fila['esperado']} | {valores} |")
    c = reporte["control_umbral"]
    filas += ["", f"**Control de preguntas trampa** (umbral mínimo de similitud = {c['umbral']:.2f}; "
              f"resultado: {'PASA' if c['pasa'] else 'FALLA'}):", "",
              "| ID | Tipo | Similitud máxima | Esperado | Resultado |", "|---|---|---|---|---|"]
    for f in c["filas"]:
        filas.append(f"| {f['id']} | {f['tipo']} | {f['similitud_maxima']:.3f} | {f['esperado']} | "
                     f"{'PASA' if f['ok'] else 'FALLA'} |")
    filas.append("")
    filas.append("El detalle completo (los 5 chunks recuperados por pregunta y modo) está en `resultados/reporte_evaluacion.json`.")
    return "\n".join(filas)


def actualizar_readme(reporte: dict) -> None:
    readme = config.README.read_text(encoding="utf-8")
    patron = re.compile(re.escape(MARCA_INICIO) + r".*?" + re.escape(MARCA_FIN), re.DOTALL)
    if not patron.search(readme):
        logger.warning("No encontré las marcas %s ... %s en el README: no se actualizó.", MARCA_INICIO, MARCA_FIN)
        return
    nuevo = patron.sub(lambda _: f"{MARCA_INICIO}\n{tabla_markdown(reporte)}\n{MARCA_FIN}", readme)
    config.README.write_text(nuevo, encoding="utf-8")
    logger.info("README.md actualizado con la tabla de métricas.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula Precision@k y Recall@k con el golden set.")
    parser.add_argument("--actualizar-readme", action="store_true", help="copia la tabla de resultados al README.md")
    args = parser.parse_args()

    config.configurar_logging("log_evaluacion.txt")
    rag = RAGSystem()
    reporte = evaluar(rag, cargar_golden_set(), cargar_trampas())
    texto = imprimir_reporte(reporte)

    config.RESULTADOS_DIR.mkdir(exist_ok=True)
    (config.RESULTADOS_DIR / "reporte_evaluacion.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.RESULTADOS_DIR / "reporte_evaluacion.txt").write_text(texto + "\n", encoding="utf-8")
    logger.info("Reporte guardado en resultados/reporte_evaluacion.json y .txt")
    if args.actualizar_readme:
        actualizar_readme(reporte)


if __name__ == "__main__":
    main()
