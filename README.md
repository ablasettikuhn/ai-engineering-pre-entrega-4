# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

**Comisión #102135 · CoderHouse · Alexandra Blasetti Kuhn**

Módulo de Recuperación Escalable: un pipeline de ingesta a **Pinecone Serverless**, un **recuperador híbrido** (BM25 + búsqueda vectorial combinados con `EnsembleRetriever`) y un script de **evaluación** con Precision@5 y Recall@5 sobre un Golden Set.

---

## La aplicación real: buscador de evidencia para el staff de rendimiento

**El problema.** El staff de rendimiento de un club (preparadores físicos, fisiólogos, readaptadores) toma decisiones diarias sobre carga, fatiga y riesgo de lesión. La evidencia está en papers científicos en inglés, y las preguntas surgen en español y con prisa: *"¿el índice de Hooper sirve más que la HRV después de un partido?"*, *"¿qué tan válidas son las unidades GNSS de 10 Hz?"*.

Consultar un asistente de IA general no resuelve el problema. Responde con su conocimiento general, sin mostrar de qué estudio sale cada dato, y puede "alucinar": dar cifras o afirmaciones convincentes que ningún estudio respalda. Además, no distingue qué evidencia es la más sólida para una disciplina específica, como el monitoreo de la carga en deportes de equipo. En un staff de rendimiento, una cifra inventada sobre fatiga o riesgo de lesión puede terminar en una mala decisión sobre un jugador.

**La solución.** Un buscador que responde **solo con la biblioteca de papers que elige el staff**, y devuelve los 5 fragmentos más relevantes con autor, año, página y DOI para verificarlos. Opcionalmente, Gemini redacta una respuesta citando esos fragmentos. Si la biblioteca no tiene evidencia sobre la pregunta, el sistema responde "No lo sé" en lugar de inventar.

**Pensado para crecer.** Hoy la biblioteca tiene 8 papers. Mañana pueden ser cientos, y también otros tipos de documentos: protocolos internos del club, informes médicos o guías de las federaciones. Por eso los datos viven en un índice en la nube (Pinecone Serverless), que escala sin cambiar el código, y cada tipo de documento puede ir en su propio namespace.

**Por qué en la nube y por qué híbrido:**

| Necesidad real | Cómo la resuelve este repo |
|---|---|
| La biblioteca crece (hoy 8 papers, mañana cientos, y otros tipos de documentos) | Índice **Pinecone Serverless**: escala sin administrar servidores; cada tipo de dato va en su propio **namespace** |
| Las preguntas usan siglas y nombres propios (sRPE, HRV, HRR, GNSS, Foster, Hooper) | **BM25** encuentra el término exacto, que un embedding puede "diluir" |
| Las preguntas están en español y los papers en inglés | La **búsqueda vectorial** (Gemini, multilingüe) entiende el significado en ambos idiomas |
| A veces solo interesa un tema | Cada chunk lleva **categorías** en la metadata y se puede filtrar (`--categoria frecuencia_cardiaca`) |
| Hay que citar bien | Cada chunk guarda en Pinecone su **texto original, paper y página**: no hace falta otra base de datos |

**Relación con la Pre-entrega 3.** La Pre-entrega 3 era un RAG local (ChromaDB) con 3 papers y búsqueda solo vectorial. Esta entrega lleva el mismo dominio a la nube, amplía la biblioteca a 8 papers, agrega búsqueda léxica y mide la calidad de la recuperación con métricas.

---

## Cumplimiento de la consigna y evidencia

Cada requisito, el archivo que lo implementa, el comando que lo ejecuta y la evidencia que queda en el repositorio (carpeta `resultados/`, generada por los propios scripts):

| Requisito de la consigna | Implementación | Comando | Evidencia |
|---|---|---|---|
| `.env` con `PINECONE_API_KEY`, clave del LLM e `INDEX_NAME` | `.env.example` (Gemini reemplaza a OpenAI: `GOOGLE_API_KEY`); `config.py` lee y valida cada variable | `copy .env.example .env` | Cada variable que lee `config.py` está en `.env.example` |
| Script de inicialización que verifica si el índice existe y lo crea en modo Serverless | `setup_pinecone.py` → `asegurar_indice()` | `python setup_pinecone.py` | `resultados/log_setup.txt` (dimensión, métrica, vectores por namespace) |
| Evitar el mismatch de dimensiones | `setup_pinecone.py` compara dimensión y métrica; `ingesta.py` valida cada vector | `python setup_pinecone.py` | Mensaje `DimensionIncompatibleError` si no coinciden (ver [Decisiones técnicas](#decisiones-técnicas)) |
| Dataset de documentos técnicos (PDF, Markdown o JSON) | 8 papers científicos reales con licencia CC BY: `fuentes.json` → `data/*.md` | `python descargar_papers.py` | `data/*.md` y `resultados/log_descarga.txt` |
| Chunking con `RecursiveCharacterTextSplitter` (~500-800 tokens) | `ingesta.py` → `crear_splitter()`: 700 tokens, 100 de solapamiento | `python ingesta.py` | `resultados/log_ingesta.txt` (chunks y tokens por paper) |
| Embeddings en Pinecone con el contenido y la fuente en la metadata | `ingesta.py` → `construir_registro()` | `python ingesta.py` | Registro de ejemplo impreso en `resultados/log_ingesta.txt` |
| Metadatos avanzados (fuente, página, categorías) | campos `fuente`, `pagina`, `pagina_fin`, `categorias` ([esquema](#esquema-de-un-registro-en-pinecone)) | `python consultar.py "..." --categoria frecuencia_cardiaca` | `resultados/ejemplo_consulta.txt` |
| Uso de namespaces | `PINECONE_NAMESPACE=papers-cientificos` en ingesta, consulta y evaluación | `python setup_pinecone.py` | `resultados/log_setup.txt` (vectores por namespace) |
| Ingesta idempotente (sin duplicar ni re-indexar) | IDs deterministas + huella en `resultados/manifest_ingesta.json` | `python ingesta.py` (dos veces) | `resultados/log_ingesta.txt` (2.ª ejecución: "no se re-indexa") |
| Clase `RAGSystem` con `EnsembleRetriever` (BM25 + Pinecone) que devuelve el Top-5 | `rag_system.py` | `python consultar.py "..."` | `resultados/ejemplo_consulta.txt` (origen de cada resultado: bm25, vectorial o ambos) |
| `evaluate.py` con benchmark de 5 preguntas, Recall@5 y Precision@5 | `evaluate.py` + `golden_set.json` | `python evaluate.py --actualizar-readme` | `resultados/reporte_evaluacion.txt` y `.json`; tabla en [Resultados](#resultados-de-la-evaluación) |
| Reporte en consola y resumen en el README | `evaluate.py` imprime el reporte y escribe la tabla en este README | `python evaluate.py --actualizar-readme` | Sección [Resultados](#resultados-de-la-evaluación) |
| Instrucciones para replicar el índice | Este README | — | Sección [Cómo replicar el índice](#cómo-replicar-el-índice) |

---

## Arquitectura

```mermaid
flowchart LR
    A[fuentes.json<br/>8 papers CC BY] -->|descargar_papers.py| B[data/*.md<br/>texto + marcas de página]
    B -->|ingesta.py<br/>chunks de 700 tokens| C[Gemini embeddings<br/>1536 dim]
    C -->|upsert| D[(Pinecone Serverless<br/>índice papers-ciencias-deporte<br/>namespace papers-cientificos)]
    D -->|query vectorial| E[PineconeRetriever]
    D -->|list + fetch<br/>texto de la metadata| F[BM25Retriever]
    E --> G[EnsembleRetriever<br/>Reciprocal Rank Fusion]
    F --> G
    G --> H[RAGSystem.buscar<br/>Top-5]
    H --> I[consultar.py]
    H --> J[evaluate.py<br/>Precision@5 / Recall@5]
```

| Archivo | Función |
|---|---|
| `config.py` | Única fuente de configuración (lee el `.env`). Valida chunking, pesos y nombre del índice. |
| `descargar_papers.py` | Paso 0: descarga los PDFs, extrae el texto por página y genera `data/<id>.md`. |
| `setup_pinecone.py` | Paso 1: crea o verifica el índice Serverless y controla dimensión y métrica. |
| `ingesta.py` | Paso 2: chunking, embeddings y upsert con metadata. Es idempotente. |
| `rag_system.py` | Paso 3: `PineconeRetriever`, `BM25Retriever`, `EnsembleRetriever` y la clase `RAGSystem`. |
| `consultar.py` | Uso real: pregunta → Top-5 con paper y página (y `--responder` con Gemini). |
| `evaluate.py` | Paso 4: métricas con el Golden Set en 3 modos (vectorial, BM25, híbrido). |
| `golden_set.json` | 5 preguntas con su documento esperado. |
| `preguntas_trampa.json` | 2 preguntas del dominio que la biblioteca no responde (control del umbral de similitud). |
| `resultados/` | Logs y reportes generados por los scripts (evidencia). |

---

## Esquema de un registro en Pinecone

Cada chunk es un registro con `id`, `values` (el embedding de 1536 dimensiones) y esta metadata. El **texto original** viaja en la metadata, así una consulta devuelve todo lo necesario para mostrar y citar el fragmento sin consultar otra base de datos.

| Campo | Tipo | Ejemplo | Para qué sirve |
|---|---|---|---|
| `texto` | string | "The Apex units (10 and 18 Hz) were…" | Texto original del chunk (lo muestra la consulta y lo usa BM25) |
| `doc_id` | string | `beato-2018` | Identifica el paper (lo usa la evaluación) |
| `fuente` | string | `beato-2018.md` | Archivo de origen |
| `titulo`, `autores`, `anio`, `revista` | string / int | `2018` | Cita bibliográfica |
| `doi`, `url`, `licencia` | string | `10.3389/fphys.2018.01288` | Trazabilidad y licencia |
| `categorias` | lista de strings | `["validez_dispositivos", "carga_externa"]` | Filtro por tema (`$in`) |
| `pagina`, `pagina_fin` | int | `4`, `5` | Página donde empieza y termina el chunk |
| `chunk`, `tokens` | int | `7`, `684` | Posición del chunk en el paper y tamaño |

El `id` es determinista (`beato-2018-c007`): re-ejecutar la ingesta sobrescribe los mismos registros en vez de duplicarlos, y permite listar o borrar un paper entero por prefijo.

---

## Decisiones técnicas

- **Gemini en lugar de OpenAI, con la dimensión de la consigna.** `gemini-embedding-001` permite elegir la dimensión de salida; se usa **1536**, la misma que `text-embedding-3-small`. Así el índice queda exactamente como sugiere la consigna y Gemini tiene plan gratuito.
- **Protección contra el "mismatch de dimensiones".** `setup_pinecone.py` compara la dimensión y la métrica del índice existente con las del modelo antes de subir nada, y la ingesta vuelve a verificar el largo de cada vector. Si cambias `EMBEDDINGS_DIM` a 768 con el índice de 1536 ya creado, el script se detiene con un mensaje que explica qué hacer.
- **SDK nativo de Pinecone.** La consigna permite `PineconeVectorStore` o el SDK nativo. Se eligió el SDK nativo (versión 10) porque `langchain-pinecone` exige `pinecone<8`, una versión que usa una API anterior. `PineconeRetriever` (en `rag_system.py`) es un retriever de LangChain de pocas líneas sobre `index.query()`, y deja ver exactamente qué se consulta.
- **Namespace.** Los papers viven en el namespace `papers-cientificos`. Si el club agregara, por ejemplo, sus protocolos internos, irían a otro namespace del mismo índice: cada búsqueda recorre solo los datos de su tipo, sin ruido y más rápido.
- **Chunking de 700 tokens con 100 de solapamiento.** Es el punto medio del rango 500-800. Un párrafo de "Métodos" o "Resultados" de un paper entra completo, con su contexto, sin mezclar varios temas en un mismo vector. El tamaño se mide en tokens (`from_tiktoken_encoder`), no en caracteres.
- **Página exacta de cada chunk.** El splitter devuelve trozos literales del texto, así que la ingesta busca la posición de cada chunk en el documento y la traduce a la página con las marcas `<!-- pagina N -->`.
- **BM25 con tokenizador propio.** El tokenizador por defecto de BM25 solo separa por espacios. El de este repo pasa a minúsculas, quita tildes, elimina palabras vacías en español e inglés y conserva términos compuestos (`session-RPE` → `session-rpe`, `session`, `rpe`). La extracción del PDF normaliza las ligaduras ("ﬁ" → "fi") para que BM25 no pierda palabras.
- **El corpus de BM25 se lee desde Pinecone.** `RAGSystem` lista los IDs del namespace y trae su metadata (`list` + `fetch`). BM25 y la búsqueda vectorial ven exactamente los mismos chunks con los mismos IDs, y no hace falta tener la carpeta `data/` para consultar.
- **Fusión con `EnsembleRetriever`.** Cada recuperador aporta 10 candidatos. El ensemble los combina con Reciprocal Rank Fusion (pesos BM25 0.4 / vectorial 0.6) y `RAGSystem` devuelve los 5 primeros. Se da más peso a la búsqueda vectorial porque las preguntas están en español y los papers en inglés. `id_key="chunk_id"` asegura que un chunk encontrado por los dos recuperadores se cuente una sola vez, sumando sus puntajes. Los pesos se cambian en el `.env`.
- **Origen de cada resultado.** La consulta y la evaluación muestran si cada chunk lo encontró BM25, la búsqueda vectorial o ambas. Así se ve en la práctica qué aporta cada una.
- **Umbral mínimo de similitud.** BM25 siempre devuelve algo (alguna palabra coincide), así que la decisión "la biblioteca no tiene evidencia sobre esto" se toma con la similitud coseno del chunk más parecido. Si queda por debajo de `UMBRAL_SIMILITUD`, `consultar.py` lo avisa y `--responder` contesta "No lo sé con los papers disponibles." sin llamar a Gemini, que podría completar con conocimiento propio. El umbral no se elige a ojo: se calibra con las preguntas reales y las trampa (ver [Evaluación](#evaluación)).
- **Versiones fijas.** `requirements.txt` fija cada dependencia con `==`, para que cualquiera reproduzca el mismo entorno con el que se obtuvieron los resultados.
- **Cuota del plan gratuito.** La ingesta envía los embeddings en lotes de hasta 25.000 tokens por minuto y reintenta ante un error 429. La evaluación guarda en memoria el embedding de cada pregunta para no calcularlo dos veces.

---

## Cómo replicar el índice

Probado en Windows con Command Prompt y Python 3.14 (compatible con Python 3.12 o superior).

### 0. Requisitos

- Una API key de **Pinecone** (plan gratuito Starter): https://app.pinecone.io → *API Keys*.
- Una API key de **Gemini** (plan gratuito): https://aistudio.google.com/apikey.

### 1. Clonar e instalar

```
git clone https://github.com/ablasettikuhn/ai-engineering-pre-entrega-4.git pre-entrega-4
cd pre-entrega-4
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar las claves

```
copy .env.example .env
notepad .env
```

Completa `PINECONE_API_KEY` y `GOOGLE_API_KEY`. El resto de los valores ya tiene un valor por defecto que funciona.

### 3. Ejecutar el pipeline

| Comando | Qué hace | Cómo verificar |
|---|---|---|
| `python descargar_papers.py` | Descarga los PDFs y genera los `.md` que falten en `data/` (los que ya existen se omiten) | Termina con "Dataset: 8 de 8 papers listos en /data" |
| `python setup_pinecone.py` | Crea el índice Serverless (AWS us-east-1, 1536 dim, coseno) o verifica el existente | "Índice creado y listo" o "ya existe" + estado del índice |
| `python ingesta.py` | Chunking, embeddings y upsert al namespace. Tarda unos minutos por las pausas de cuota | Termina con "Ingesta completa: N vectores" y muestra un registro de ejemplo |
| `python ingesta.py` (otra vez) | Detecta que el namespace está al día y no re-indexa | "ya está al día … no se re-indexa". `resultados/log_ingesta.txt` conserva las dos ejecuciones |
| `python setup_pinecone.py` | Muestra cuántos vectores hay en el namespace | El número coincide con el de la ingesta |
| `python evaluate.py --actualizar-readme` | Calcula las métricas, controla las preguntas trampa, guarda el reporte y actualiza la tabla de este README | Imprime Recall@5, Precision@5 y MRR, y las verificaciones con PASA/FALLA. Si el control del umbral da FALLA, el reporte sugiere un valor: ponlo en `UMBRAL_SIMILITUD` del `.env` y vuelve a ejecutar |

### 4. Usar el buscador

```
python consultar.py "¿Qué validez tienen las unidades GNSS de 10 Hz y 18 Hz?"
python consultar.py "¿Qué validez tienen las unidades GNSS de 10 Hz y 18 Hz?" --modo bm25
python consultar.py "¿Cómo cambia la HRV después de un partido?" --categoria frecuencia_cardiaca
python consultar.py "¿Cómo se calcula la carga con el método session-RPE?" --responder
python consultar.py "¿Qué dosis de cafeína mejora el rendimiento en sprints repetidos?" --responder
python consultar.py
```

Agrega `--guardar` a cualquier consulta para sumar su salida a `resultados/ejemplo_consulta.txt` (evidencia).

La pregunta sobre cafeína es una trampa: la biblioteca no la responde, así que el sistema avisa que no hay evidencia suficiente y contesta "No lo sé". El último comando abre el modo interactivo (escribe `salir` para terminar). `--modo vectorial` y `--modo bm25` permiten comparar con la búsqueda híbrida, que es la opción por defecto.

**Para empezar de cero:** `python ingesta.py --forzar` vacía el namespace y re-indexa todo. Si cambias el modelo o la dimensión de los embeddings, usa además otro `INDEX_NAME`.

---

## Evaluación

### Golden Set

`golden_set.json` tiene 5 preguntas en español. Para cada una se sabe de antemano qué paper la responde. Las preguntas combinan los casos en los que se espera que ayude cada tipo de búsqueda:

| ID | Pregunta (resumida) | Paper esperado | Tipo |
|---|---|---|---|
| q1 | Cálculo de la carga con session-RPE de Foster | `haddad-2017` | sigla + nombre propio |
| q2 | Validez de unidades GNSS de 10 y 18 Hz | `beato-2018` | término técnico + cifras |
| q3 | Índice de Hooper frente a HRV tras un partido | `rabbani-2019` | nombre propio + sigla |
| q4 | Predecir lesiones con GPS y machine learning | `rossi-2018` | conceptual |
| q5 | Medidas de frecuencia cardíaca para el estado de entrenamiento | `buchheit-2014` (también relevante: `schneider-2018`) | conceptual, dos papers relevantes |

Los 3 papers restantes (`torres-ronda-2022`, `gomez-carmona-2020` y, en q1-q4, `schneider-2018`) funcionan como distractores: tratan temas cercanos (tracking, acelerometría, frecuencia cardíaca) y podrían confundirse con el paper esperado.

### Métricas

- **Recall@5:** ¿está el paper esperado entre los 5 chunks recuperados? Vale 1 o 0 por pregunta. Con un solo paper esperado por pregunta, equivale a la tasa de acierto.
- **Precision@5:** proporción de los 5 chunks recuperados que provienen de un paper relevante (el esperado o los de `documentos_relevantes`).
- **MRR (extra):** 1 / posición del primer chunk del paper esperado. Premia que el paper correcto aparezca arriba.

`evaluate.py` calcula las tres métricas en los tres modos (vectorial, BM25 e híbrido), para ver qué aporta la combinación.

### Preguntas trampa y umbral de similitud

Un buen recuperador también tiene que reconocer cuándo la biblioteca **no** tiene la respuesta. `preguntas_trampa.json` tiene dos preguntas del mismo dominio que ningún paper responde:

| ID | Pregunta (resumida) | Por qué es trampa |
|---|---|---|
| t1 | Dosis de cafeína (mg/kg) para sprints repetidos | Ningún paper estudia la cafeína; el LLM conoce una cifra de memoria |
| t2 | Gramos de proteína por kilo para un futbolista | Nutrición: fuera de la biblioteca |

`evaluate.py` calcula la similitud máxima de las 5 preguntas reales y de las 2 trampa. El control **PASA** si todas las reales quedan en o por encima de `UMBRAL_SIMILITUD` y las trampa por debajo. Si no, el reporte informa el punto medio entre la menor similitud real y la mayor similitud trampa como umbral sugerido. Así el umbral se calibra con datos, no a ojo.

### Resultados de la evaluación

<!-- METRICAS:INICIO -->
Resultados de `python evaluate.py` (2026-11-26): 5 preguntas, 202 chunks de 8 papers, ensemble BM25 0.4 / vectorial 0.6.

| Modo | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| hibrido | 1.00 | 1.00 | 1.00 |
| vectorial | 1.00 | 1.00 | 1.00 |
| bm25 | 1.00 | 0.96 | 1.00 |

| Pregunta | Paper esperado | Recall@5 hibrido | Recall@5 vectorial | Recall@5 bm25 |
|---|---|---|---|---|
| q1 | haddad-2017 | 1 | 1 | 1 |
| q2 | beato-2018 | 1 | 1 | 1 |
| q3 | rabbani-2019 | 1 | 1 | 1 |
| q4 | rossi-2018 | 1 | 1 | 1 |
| q5 | buchheit-2014 | 1 | 1 | 1 |

**Control de preguntas trampa** (umbral mínimo de similitud = 0.70; resultado: PASA):

| ID | Tipo | Similitud máxima | Esperado | Resultado |
|---|---|---|---|---|
| q1 | golden | 0.770 | con evidencia | PASA |
| q2 | golden | 0.822 | con evidencia | PASA |
| q3 | golden | 0.833 | con evidencia | PASA |
| q4 | golden | 0.823 | con evidencia | PASA |
| q5 | golden | 0.760 | con evidencia | PASA |
| t1 | trampa | 0.645 | sin evidencia | PASA |
| t2 | trampa | 0.614 | sin evidencia | PASA |

El detalle completo (los 5 chunks recuperados por pregunta y modo) está en `resultados/reporte_evaluacion.json`.
<!-- METRICAS:FIN -->

### Interpretación de los resultados

- **Recall@5 = 1.00 en los tres modos.** Las 5 preguntas apuntan a papers con temas bien diferenciados, así que cada búsqueda por sí sola ya encuentra el paper correcto. El benchmark confirma que el pipeline completo funciona de punta a punta (ingesta, metadata, búsqueda vectorial, BM25 y fusión), pero con 5 preguntas no alcanza para mostrar una diferencia entre los modos.
- **BM25 queda apenas por debajo en Precision@5 (0.96).** Uno de sus 25 chunks vino de un paper no relevante: BM25 solo coincide en las palabras que comparten la pregunta en español y el paper en inglés. La búsqueda vectorial, multilingüe, no tuvo ese error, y el modo híbrido tampoco.
- **Qué aporta el modo híbrido.** En este golden set, ningún recuperador falló, así que el híbrido iguala al vectorial. Su valor aparece cuando uno de los dos falla y el otro compensa. `resultados/ejemplo_consulta.txt` muestra, para cada chunk, si lo encontró BM25, la búsqueda vectorial o ambas. El siguiente paso sería un golden set más exigente: preguntas formuladas solo con una sigla o un apellido, o preguntas donde dos papers compiten.
- **Umbral de similitud.** Las preguntas reales quedaron entre 0.760 y 0.833 de similitud máxima, y las trampa entre 0.614 y 0.645. El umbral de 0.70 queda en el medio de esa brecha, con un margen de unas 0.06 hacia cada lado. Es un margen chico: una pregunta trampa más cercana a los temas de la biblioteca podría superarlo, así que conviene recalibrarlo cuando la biblioteca crezca.

### Limitaciones

- **El umbral se calibró con 7 preguntas.** Separa bien estos casos, pero con más preguntas reales y trampa la calibración sería más confiable.
- **5 preguntas es una muestra chica.** Sirve para comparar los modos y detectar errores gruesos, no para dar una cifra definitiva de calidad.
- **La relevancia se juzga por paper.** Un chunk cuenta como útil si viene del paper esperado, aunque ese fragmento puntual no responda la pregunta. Es una aproximación razonable para un benchmark chico.
- **BM25 vive en memoria.** Se reconstruye en cada ejecución desde la metadata de Pinecone. Funciona muy bien con cientos o miles de chunks. Con millones, convendría la búsqueda léxica dentro del propio Pinecone (vectores dispersos o full-text search) para no traer todo el corpus.
- **Idioma.** BM25 solo coincide en las palabras que comparten la pregunta en español y el paper en inglés (siglas, nombres, términos técnicos). Una mejora futura sería traducir la pregunta a palabras clave en inglés antes de pasarla a BM25.

---

## Papers del dataset

Todos tienen licencia Creative Commons Attribution (CC BY), que permite redistribuir el texto citando la fuente. El texto extraído está en `data/`.

| ID | Referencia | Categorías |
|---|---|---|
| `torres-ronda-2022` | Torres-Ronda L, et al. (2022). Tracking Systems in Team Sports: A Narrative Review of Applications of the Data and Sport Specific Analysis. *Sports Medicine - Open*, 8:15. https://doi.org/10.1186/s40798-022-00408-z | carga_externa, tecnologia_tracking |
| `haddad-2017` | Haddad M, et al. (2017). Session-RPE Method for Training Load Monitoring: Validity, Ecological Usefulness, and Influencing Factors. *Frontiers in Neuroscience*, 11:612. https://doi.org/10.3389/fnins.2017.00612 | carga_interna, percepcion_esfuerzo |
| `beato-2018` | Beato M, et al. (2018). The Validity and Between-Unit Variability of GNSS Units (STATSports Apex 10 and 18 Hz) for Measuring Distance and Peak Speed in Team Sports. *Frontiers in Physiology*, 9:1288. https://doi.org/10.3389/fphys.2018.01288 | validez_dispositivos, tecnologia_tracking, carga_externa |
| `buchheit-2014` | Buchheit M (2014). Monitoring training status with HR measures: do all roads lead to Rome? *Frontiers in Physiology*, 5:73. https://doi.org/10.3389/fphys.2014.00073 | frecuencia_cardiaca, estado_entrenamiento |
| `schneider-2018` | Schneider C, et al. (2018). Heart Rate Monitoring in Team Sports—A Conceptual Framework for Contextualizing Heart Rate Measures for Training and Recovery Prescription. *Frontiers in Physiology*, 9:639. https://doi.org/10.3389/fphys.2018.00639 | frecuencia_cardiaca, fatiga_recuperacion |
| `rabbani-2019` | Rabbani A, et al. (2019). Match Fatigue Time-Course Assessment Over Four Days: Usefulness of the Hooper Index and Heart Rate Variability in Professional Soccer Players. *Frontiers in Physiology*, 10:109. https://doi.org/10.3389/fphys.2019.00109 | fatiga_recuperacion, bienestar, frecuencia_cardiaca |
| `gomez-carmona-2020` | Gómez-Carmona CD, et al. (2020). Accelerometry as a method for external workload monitoring in invasion team sports. A systematic review. *PLoS ONE*, 15(8): e0236643. https://doi.org/10.1371/journal.pone.0236643 | carga_externa, acelerometria |
| `rossi-2018` | Rossi A, et al. (2018). Effective injury forecasting in soccer with GPS training data and machine learning. *PLoS ONE*, 13(7): e0201264. https://doi.org/10.1371/journal.pone.0201264 | lesiones, machine_learning, carga_externa |

Los títulos de algunos papers nombran marcas comerciales de dispositivos. Aquí aparecen solo como parte de la cita de investigaciones publicadas, con sus datos reales.

---

## Glosario

### Siglas y nombres propios de los papers

Los papers usan estos términos tal cual, en inglés. Son el tipo de palabra que BM25 encuentra de forma exacta: si la pregunta dice "HRV" o "Hooper", BM25 busca exactamente esa palabra en los chunks, aunque el resto de la pregunta esté en español.

| Término | Significado | Paper principal |
|---|---|---|
| **RPE** | *Rating of Perceived Exertion*: percepción subjetiva del esfuerzo que informa el deportista, por lo general en la escala CR-10 de Borg (0 a 10). | `haddad-2017` |
| **sRPE / session-RPE** | Carga interna de una sesión: RPE × duración de la sesión en minutos. Se expresa en unidades arbitrarias (UA). Ejemplo: RPE 7 × 60 min = 420 UA. | `haddad-2017` |
| **Foster** | Carl Foster, fisiólogo del ejercicio. Él y sus colaboradores propusieron el método session-RPE a fines de los años 90, por eso se lo llama "método de Foster". | `haddad-2017` |
| **HR** | *Heart Rate*: frecuencia cardíaca, en latidos por minuto. | `buchheit-2014`, `schneider-2018` |
| **HRV** | *Heart Rate Variability*: variabilidad de la frecuencia cardíaca, es decir, cuánto varía el tiempo entre latidos consecutivos. Se mide en reposo y refleja la actividad del sistema nervioso parasimpático. Si baja, puede indicar fatiga. | `buchheit-2014`, `rabbani-2019` |
| **HRR** | *Heart Rate Recovery*: cuánto baja la frecuencia cardíaca en los primeros minutos después de un esfuerzo submáximo. Una recuperación más rápida suele indicar mejor estado de forma. | `buchheit-2014` |
| **HRex** | *Exercise Heart Rate*: frecuencia cardíaca durante un ejercicio submáximo estandarizado, por ejemplo en la entrada en calor. | `buchheit-2014`, `schneider-2018` |
| **Índice de Hooper** | Cuestionario de bienestar de Hooper y Mackinnon (1995). El deportista puntúa cuatro ítems: calidad del sueño, estrés, fatiga y dolor muscular. Un total más alto indica peor bienestar. | `rabbani-2019` |
| **GNSS** | *Global Navigation Satellite System*: término general para los sistemas de posicionamiento por satélite (GPS, GLONASS, Galileo, BeiDou). Los dispositivos que llevan los jugadores lo usan para medir distancias y velocidades. | `beato-2018` |
| **GPS** | El sistema de posicionamiento de EE.UU., uno de los GNSS. En deporte se dice "GPS" por costumbre para hablar de cualquier dispositivo de tracking por satélite. | `rossi-2018`, `torres-ronda-2022` |
| **Hz (10 Hz, 18 Hz)** | Frecuencia de muestreo del dispositivo: 10 Hz son 10 posiciones registradas por segundo. Una frecuencia mayor mejora la precisión en acciones cortas, como un sprint. | `beato-2018` |
| **Velocidad pico** | La velocidad máxima alcanzada en una acción o sesión. | `beato-2018` |
| **Acelerometría** | Medición de aceleraciones con sensores inerciales del dispositivo (por lo general ubicado entre las escápulas). Sirve para cuantificar saltos, impactos y cambios de dirección que el GNSS no capta bien. | `gomez-carmona-2020` |
| **Machine learning** | Modelos que aprenden patrones a partir de datos. En `rossi-2018` se usan para predecir lesiones a partir de la carga registrada con GPS. | `rossi-2018` |

### Categorías de la metadata

Cada chunk guarda en Pinecone las categorías de su paper (campo `categorias`). Se escriben sin tildes ni espacios porque son claves para filtrar, por ejemplo:
`python consultar.py "¿Cómo cambia la HRV después de un partido?" --categoria frecuencia_cardiaca`

| Categoría | Qué agrupa | Papers |
|---|---|---|
| `carga_externa` | El trabajo físico que realiza el deportista, medido desde afuera: distancias, velocidades, aceleraciones, impactos. | `torres-ronda-2022`, `beato-2018`, `gomez-carmona-2020`, `rossi-2018` |
| `carga_interna` | La respuesta del organismo a esa carga: frecuencia cardíaca, esfuerzo percibido. | `haddad-2017` |
| `percepcion_esfuerzo` | Métodos basados en lo que el deportista informa sobre su esfuerzo (RPE, sRPE). | `haddad-2017` |
| `tecnologia_tracking` | Sistemas que registran la posición y el movimiento de los jugadores (GNSS, sistemas locales, video). | `torres-ronda-2022`, `beato-2018` |
| `validez_dispositivos` | Estudios que comprueban si un dispositivo mide bien, comparándolo con una referencia. | `beato-2018` |
| `acelerometria` | Cuantificación de la carga con acelerómetros. | `gomez-carmona-2020` |
| `frecuencia_cardiaca` | Uso de la frecuencia cardíaca y sus derivados (HRV, HRR, HRex) para monitorear al deportista. | `buchheit-2014`, `schneider-2018`, `rabbani-2019` |
| `estado_entrenamiento` | Cómo saber si el deportista está mejorando, estancado o fatigado frente al entrenamiento. | `buchheit-2014` |
| `fatiga_recuperacion` | Seguimiento de la fatiga y de la recuperación entre sesiones o después de un partido. | `schneider-2018`, `rabbani-2019` |
| `bienestar` | Cuestionarios de bienestar autoinformado, como el índice de Hooper. | `rabbani-2019` |
| `lesiones` | Riesgo, prevención o predicción de lesiones. | `rossi-2018` |
| `machine_learning` | Modelos predictivos aplicados a datos de entrenamiento. | `rossi-2018` |
