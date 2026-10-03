# Trabajo práctico — QA multiagente sobre tablas en PDFs

**Materia:** Sistemas Multiagente de IA
**Publicación:** miércoles 30 de septiembre de 2026
**Modalidad:** Individual
**Entrega:** Link al repositorio Git + ID de commit (ver [Sección 9](#9-entregables))

---

## 1. Descripción general

Muchos informes (estados contables, anuarios estadísticos, papers científicos, publicaciones gubernamentales) guardan sus datos más útiles en **tablas dentro de PDFs**. Esos datos son difíciles de consultar: hay que encontrar la tabla correcta, entender qué significan sus filas y columnas, y después hacer las cuentas a mano.

En este trabajo vas a construir un **sistema de dos agentes** que resuelve este problema:

1. **Agente de Ingesta** — recibe un PDF, extrae sus tablas, deduce de qué trata cada una (nombre, descripción, encabezados de columna, encabezados de fila) a partir del contexto que la rodea, y las indexa en una base de datos vectorial.
2. **Agente Analista** — recibe una pregunta en lenguaje natural, encuentra las tablas que pueden responderla, **escribe un programa en Python** que calcula la respuesta a partir de esas tablas, lo ejecuta y devuelve el resultado.

Los usuarios interactúan con ambos agentes a través de una aplicación web en **Streamlit**.

---

## 2. Objetivos de aprendizaje

Al completar este trabajo vas a poder:

- Diseñar e implementar flujos de agentes como grafos explícitos con **LangGraph** (estado, nodos, aristas condicionales, ciclos).
- Usar las abstracciones de **LangChain** (chat models, prompts, salida estructurada, tools, retrievers).
- Usar un LLM para enriquecer datos no estructurados (tablas + contexto) con **metadatos semánticos**.
- Construir un pipeline de recuperación sobre una **base de datos vectorial (Pinecone)**.
- Generar, ejecutar y **autocorregir** código escrito por un LLM de forma controlada.
- Instrumentar y depurar un sistema multiagente con trazas de **LangSmith**.
- Consumir modelos a través de un **proxy compatible con OpenAI (LiteLLM)**.

---

## 3. Stack tecnológico obligatorio

| Aspecto              | Tecnología                                              |
|----------------------|---------------------------------------------------------|
| Front end            | **Streamlit**                                           |
| Orquestación de agentes | **LangGraph**                                        |
| Capa de LLM / herramientas | **LangChain**                                     |
| Servidor de inferencia | **Proxy LiteLLM de la universidad** (API compatible con OpenAI) |
| Base de datos vectorial | **Pinecone**                                         |
| Trazas / observabilidad | **LangSmith**                                        |
| Lenguaje             | Python 3.11+                                            |
| Procesamiento de PDF | PyMuPDF (sugerido, no obligatorio)                      |

### Sugerencia: PyMuPDF para procesar PDFs

Recomendamos **[PyMuPDF](https://pymupdf.readthedocs.io/)** (`pip install pymupdf`, se importa como `pymupdf` o `fitz`). Es rápido, no tiene dependencias externas como Java o Ghostscript, y cubre las dos partes de lo que necesita el Agente de Ingesta:

- **Extracción de tablas:** `page.find_tables()` detecta tablas y devuelve objetos con `.header.names`, `.extract()` y `.to_pandas()`.
- **Extracción de contexto:** `page.get_text("blocks")` / `page.get_text("dict")` devuelven bloques de texto con sus bounding boxes. Comparalos con el `.bbox` de cada tabla para encontrar el epígrafe, el título y los párrafos que están justo arriba y abajo de la tabla.

```python
import pymupdf

doc = pymupdf.open("report.pdf")
for page in doc:
    for i, table in enumerate(page.find_tables().tables):
        df = table.to_pandas()
        x0, y0, x1, y1 = table.bbox
        nearby = [
            b[4] for b in page.get_text("blocks")
            if b[3] <= y0 and y0 - b[3] < 80   # bloques de texto justo arriba de la tabla
        ]
        print(page.number + 1, i, df.shape, nearby[-1:])
```

Podés usar otra librería (por ejemplo `pdfplumber`, `camelot`, `docling`, `unstructured`), o combinar alguna con PyMuPDF, siempre que **expliques tu elección en el README**. También podés agregar cualquier librería auxiliar que necesites (`pandas`, `pydantic`, etc.).

### 3.1 Acceso a los servicios

| Servicio | Quién da el acceso | Cómo obtenerlo | Límites |
|---------|---------------------|---------------|--------|
| **Proxy LiteLLM** | Equipo docente | Tu clave personal, la URL del proxy y la lista de modelos se envían por el canal de la materia en los primeros días del trabajo. | Cada clave tiene un presupuesto/límite de uso definido por el equipo docente. |
| **Pinecone** | Vos | Registrate en [pinecone.io](https://www.pinecone.io/). El **plan gratuito Starter alcanza**. Creá un índice serverless cuya dimensión coincida con el modelo de embeddings que uses, y copiá la API key desde la consola. | Aplican los límites del plan gratuito; un índice alcanza para este trabajo. |
| **LangSmith** | Vos | Registrate en [smith.langchain.com](https://smith.langchain.com/). El **plan gratuito Developer alcanza**. Creá una API key en *Settings → API Keys*. | Límite mensual de trazas del plan gratuito; tenelo en cuenta cuando corras evaluaciones. |

No hace falta tarjeta de crédito ni plan pago. Si no podés crear una cuenta, contactá al equipo docente lo antes posible, no cerca de la fecha de entrega.

### 3.2 Proxy LiteLLM

Todas las llamadas a LLMs y a embeddings **deben** pasar por el proxy LiteLLM de la universidad. Comunicate con él usando las clases compatibles con OpenAI de LangChain:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    base_url=os.environ["LITELLM_BASE_URL"],
    api_key=os.environ["LITELLM_API_KEY"],
    model=os.environ["LLM_MODEL"],
)

embeddings = OpenAIEmbeddings(
    base_url=os.environ["LITELLM_BASE_URL"],
    api_key=os.environ["LITELLM_API_KEY"],
    model=os.environ["EMBEDDING_MODEL"],
)
```

Usá solo los modelos que expone el proxy:

| Propósito | Modelo (definido por variable de entorno) |
|---------|-------------------------|
| Chat / generación de código | `LLM_MODEL` = `<nombre del modelo de chat informado en el canal de la materia>` |
| Embeddings | `EMBEDDING_MODEL` = `<nombre del modelo de embeddings informado en el canal de la materia>` (usá su dimensión de salida para tu índice de Pinecone) |

Podés listar los modelos disponibles con `curl -H "Authorization: Bearer $LITELLM_API_KEY" $LITELLM_BASE_URL/models`. **Nunca subas claves al repositorio.**

### 3.3 LangSmith

Las trazas tienen que estar activadas en cada ejecución de los agentes. Configuralas con variables de entorno:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=mas-2026s2-<tu-legajo>
```

Pinecone se configura de la misma forma:

```bash
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=mas-2026s2-tables
```

Cada ejecución de Ingesta y cada ejecución del Analista tiene que aparecer como una traza, con cada nodo del grafo visible como un span.

---

## 4. Arquitectura del sistema

```
                ┌─────────────────────────── UI Streamlit ───────────────────────────┐
                │   [ Pestaña Subir PDF ]                 [ Pestaña Hacer pregunta ] │
                └────────┬──────────────────────────────────────────────┬────────────┘
                         │                                              │
                         ▼                                              ▼
              ┌────────────────────┐                         ┌────────────────────┐
              │ Agente de Ingesta  │                         │  Agente Analista   │
              │    (LangGraph)     │                         │    (LangGraph)     │
              └─────────┬──────────┘                         └─────────┬──────────┘
                        │ upsert                              búsqueda │
                        ▼                                              ▼
              ┌─────────────────────────────────────────────────────────────────┐
              │        Pinecone (metadatos de tablas + embeddings)              │
              └─────────────────────────────────────────────────────────────────┘
                        │                                              ▲
                        ▼                                              │ carga de datos
              ┌─────────────────────────────────────────────────────────────────┐
              │   Almacén de datos de tablas (p. ej. archivos Parquet/CSV,      │
              │   SQLite, o JSON en los metadatos de Pinecone si entra en los   │
              │   límites de tamaño)                                            │
              └─────────────────────────────────────────────────────────────────┘
                                       Proxy LiteLLM ◄── todas las llamadas a LLM/embeddings
                                       LangSmith     ◄── todas las trazas
```

> **Nota:** Los metadatos de Pinecone tienen un tamaño limitado (unos 40 KB por registro). Las tablas grandes no van a entrar, así que guardá los datos de las tablas en algún lugar desde donde el Agente Analista pueda volver a cargarlos (por ejemplo, un archivo local por tabla, identificado por un `table_id` guardado en los metadatos de Pinecone). Explicá tu elección en el README.

---

## 5. Agente 1 — Agente de Ingesta

### 5.1 Entrada

Un PDF subido a través de la UI de Streamlit. Puede contener una o más tablas distribuidas en varias páginas.

#### Alcance obligatorio

- **Solo PDFs legibles por máquina.** El PDF tiene que tener una capa de texto: texto que se pueda seleccionar y copiar, como en los PDFs exportados desde Word, LaTeX, Excel o herramientas de reportes. Tu sistema solo tiene que manejar tablas cuyas celdas se puedan leer desde esa capa de texto.
- **Los PDFs escaneados o compuestos solo por imágenes quedan fuera del alcance.** No necesitás OCR, y ninguno de los PDFs de ejemplo ni de las preguntas de prueba debería depender de él. Si agregás OCR, es bienvenido y suma para el bonus (Sección 10), pero no se evalúa como parte del trabajo obligatorio.
- **Las tablas que continúan en otra página tienen que unirse.** Cuando una tabla sigue en la página siguiente, guardala como **una** tabla lógica, no como una tabla por página. Una continuación es una tabla al comienzo de la página siguiente que tiene la misma cantidad de columnas y que repite la misma fila de encabezado o no tiene fila de encabezado. Registrá todas las páginas que abarca en `source`. Al menos uno de tus PDFs de ejemplo tiene que incluir una tabla así.
- Si el sistema encuentra una tabla o una página que no puede leer (por ejemplo, una imagen incrustada de una tabla), debe saltearla e **informarlo en la UI**, en lugar de fallar o inventar contenido.

### 5.2 Responsabilidades

El agente tiene que ser un **`StateGraph` de LangGraph** con al menos estos pasos (podés agregar, dividir o fusionar nodos):

1. **Cargar el PDF** — leer el archivo y conservar el texto de las páginas (lo necesitás para el contexto).
2. **Extraer tablas** — detectar y extraer cada tabla de la capa de texto (ver [Alcance obligatorio](#alcance-obligatorio)) como datos estructurados (por ejemplo, un `pandas.DataFrame`). Manejar al menos:
   - encabezados de varias filas / celdas combinadas (aplanarlos de forma razonable),
   - celdas vacías,
   - números formateados como texto (separadores de miles, `%`, símbolos de moneda, paréntesis para negativos, etc.),
   - tablas que continúan en otra página (unirlas en una sola tabla).
3. **Recolectar contexto** — para cada tabla, tomar el texto cercano: epígrafe, título, texto anterior y posterior, título de la sección, notas al pie.
4. **Enriquecer con el LLM** — usando el contenido de la tabla **y** su contexto, generar metadatos estructurados (usá la salida estructurada de LangChain con un modelo de Pydantic), como mínimo:

   | Campo            | Descripción                                                                 |
   |------------------|-----------------------------------------------------------------------------|
   | `table_id`       | Identificador único y estable                                               |
   | `name`           | Nombre corto y legible (inferido si el PDF no da ninguno)                   |
   | `description`    | 2 a 4 oraciones sobre qué mide la tabla, sus unidades, período y alcance    |
   | `columns`        | Lista de encabezados de columna, cada uno con una breve descripción y el tipo/unidad inferido |
   | `row_headers`    | Lista de etiquetas de fila (índice), si la tabla las tiene                  |
   | `source`         | Nombre del archivo y número(s) de página                                    |

5. **Validar** — verificar que los metadatos coincidan con la tabla extraída (por ejemplo, que la cantidad de columnas coincida y que los nombres de columna existan). Si la validación falla, **volver** al nodo de enriquecimiento con feedback, hasta un límite de reintentos configurable.
6. **Indexar** — armar un texto para embeddings a partir de los metadatos (nombre + descripción + columnas + encabezados de fila), generar su embedding a través de LiteLLM y hacer **upsert** en Pinecone junto con los metadatos. Guardar los datos de la tabla en el almacén de datos elegido.

### 5.3 Salida

Mostrar en la UI un resumen de cada tabla ingerida: nombre, descripción, encabezados, una vista previa de los datos y la página de la que proviene.

---

## 6. Agente 2 — Agente Analista

### 6.1 Entrada

Una pregunta en lenguaje natural, por ejemplo:

- *"¿Cuál fue el crecimiento interanual promedio de los ingresos entre 2021 y 2024?"*
- *"¿Qué región tuvo la tasa de desempleo más alta en 2023, y por cuánto superó al promedio nacional?"*
- *"¿Qué proporción de los gastos totales representó I+D en cada año?"*

### 6.2 Responsabilidades

El agente tiene que ser un **`StateGraph` de LangGraph** con al menos estos pasos:

1. **Entender la consulta** — identificar las entidades, métricas, períodos y operaciones que necesita la pregunta. Reescribir o expandir la consulta para la recuperación si eso ayuda.
2. **Recuperar tablas candidatas** — buscar en Pinecone las *k* tablas candidatas más relevantes.
3. **Seleccionar tablas** — hacer que el LLM revise los metadatos de las candidatas (y opcionalmente una pequeña vista previa de los datos) y decida **qué tablas se necesitan realmente**, incluyendo los casos en los que hay que combinar **más de una tabla**. Si las tablas indexadas no contienen todo lo que la pregunta necesita, el agente tiene que seguir las [reglas de respondibilidad](#64-reglas-de-respondibilidad) en lugar de inventar una respuesta.
4. **Generar código** — escribir un programa Python autocontenido que:
   - cargue las tablas seleccionadas como `pandas.DataFrame`s (tu código le pasa los datos o las rutas de los archivos),
   - haga el cálculo,
   - imprima o devuelva la respuesta final en un formato estructurado.
5. **Ejecutar el código** — correr el código generado **en un entorno controlado** (como mínimo, un subproceso separado con timeout, y sin acceso a la red ni al sistema de archivos más allá de los datos de las tablas). Capturar stdout, stderr y el resultado.
6. **Autocorregirse** — si el código falla o la salida no es válida, enviar el error de vuelta al nodo de generación de código y reintentar, hasta un límite configurable.
7. **Responder** — redactar una respuesta final en lenguaje natural que incluya:
   - el/los resultado(s) numérico(s),
   - qué tablas (y páginas) se usaron,
   - una breve explicación del cálculo.

### 6.3 Salida

En la UI, mostrar:

- la respuesta final,
- las tablas seleccionadas (con links o expanders que muestren sus datos),
- el código Python generado (en un bloque de código),
- la salida de la ejecución,
- la cantidad de reintentos necesarios, si hubo,
- el estado de la respuesta (`answered`, `partial` o `cannot_answer`; ver 6.4) y, cuando no sea `answered`, qué datos faltan.

### 6.4 Reglas de respondibilidad

El Analista puede usar **solo datos de las tablas indexadas**. No debe usar el conocimiento general del LLM, fuentes externas ni estimaciones. Toda respuesta termina con uno de tres estados:

| Estado | Cuándo |
|--------|------|
| `answered` | Todos los valores que necesita el cálculo están en las tablas seleccionadas. |
| `partial` | La pregunta tiene varias partes y solo algunas se pueden calcular. Responder esas partes, e indicar cada parte que no se puede responder y qué datos le faltan. |
| `cannot_answer` | No se puede calcular ninguna parte. Explicar qué falta y, si existe una tabla relacionada, qué contiene (por ejemplo: "La tabla *Ingresos por año* cubre 2019–2023; 2024 no está disponible"). No se muestra ningún resultado numérico. |

Que exista una tabla relevante **no** alcanza. El Analista tiene que devolver `cannot_answer` (o `partial` para esa parte) cuando:

- **Falta un período o una entidad necesarios.** Por ejemplo, la pregunta es sobre 2024 pero la tabla llega hasta 2023, o pregunta por una región que la tabla no incluye. No interpolar, no extrapolar ni usar el año más cercano.
- **Una conversión de unidades necesaria requiere datos externos.** Por ejemplo, convertir monedas (tipos de cambio), ajustar por inflación, o convertir un conteo en una cifra per cápita sin una tabla de población. Las conversiones que no requieren datos externos están permitidas y tienen que indicarse en la respuesta. Algunos ejemplos son el cambio de escala (miles → millones), porcentaje ↔ fracción, y sumar valores mensuales para obtener un año cuando están todos los meses.
- **Falta un valor de comparación.** Por ejemplo, "comparado con el promedio nacional" cuando ninguna tabla informa un promedio nacional. No lo reemplaces por un valor calculado por vos (como el promedio de las regiones listadas), salvo que la pregunta lo defina explícitamente de esa manera.
- **El significado o la unidad de una columna no es claro** y la elección cambia el resultado; por ejemplo, cuando los metadatos no dicen si los valores son nominales o reales. Informar la ambigüedad en lugar de elegir una opción.

El estado tiene que decidirlo la lógica del agente (por ejemplo, después de la selección de tablas y/o a partir de la salida del código generado), no quedar librado a la redacción de la respuesta final. Guardalo en el estado del grafo para que se pueda testear.

---

## 7. Front end en Streamlit

Como mínimo:

- Una página o pestaña para **subir PDFs** y seguir el progreso de la ingesta (por ejemplo, mostrando en streaming las actualizaciones de los nodos del grafo).
- Una página o pestaña para **explorar las tablas indexadas** (nombre, descripción, fuente, vista previa de los datos).
- Una página o pestaña para **hacer preguntas**, con un historial tipo chat dentro de la sesión.
- Una forma clara de ver los errores (por ejemplo, una falla del LLM o de Pinecone no debería hacer caer la app sin avisar).

---

## 8. Resumen de requisitos

### Funcionales (obligatorios)

- [ ] Agente de Ingesta construido como un grafo de LangGraph con un ciclo de validación/reintento.
- [ ] Tablas extraídas de PDFs legibles por máquina (las tablas que continúan en otra página se unen en una sola), con su nombre, descripción, columnas y encabezados de fila inferidos a partir del contexto.
- [ ] Metadatos de las tablas convertidos en embeddings y guardados en Pinecone; datos de las tablas guardados en un almacén desde el que el Analista pueda cargarlos.
- [ ] Agente Analista construido como un grafo de LangGraph que recupera, selecciona (una o más tablas), genera código, lo ejecuta y se autocorrige.
- [ ] El Analista sigue las reglas de respondibilidad (Sección 6.4) y devuelve el estado `answered` / `partial` / `cannot_answer`.
- [ ] UI de Streamlit que cubre subir, explorar y preguntar.
- [ ] Todas las llamadas a LLM/embeddings pasan por el proxy LiteLLM de la universidad.
- [ ] Todas las ejecuciones de los agentes tienen trazas en LangSmith.

### No funcionales (obligatorios)

- [ ] Secretos cargados desde variables de entorno / `.env` (con un `.env.example` versionado, nunca el `.env` real).
- [ ] El código generado se ejecuta aislado del proceso de Streamlit, con timeout.
- [ ] Setup reproducible (`uv`, `poetry` o `requirements.txt`) e instrucciones claras para ejecutarlo.
- [ ] Volver a ingerir el mismo PDF no debe crear registros duplicados en Pinecone (`table_id`s idempotentes).

---

## 9. Entregables

### 9.1 Cómo entregar

Entregá exactamente dos cosas antes de la fecha límite:

- **Link al repositorio:** la URL de tu repositorio Git (GitHub, GitLab, etc.). Si es privado, dale acceso de lectura al equipo docente.
- **ID de commit:** el SHA completo del commit que querés que se corrija (`git rev-parse HEAD`).

Solo se corrige ese commit. Los commits subidos después de la fecha límite se ignoran, y el timestamp del propio commit tiene que ser anterior a la fecha límite.

### 9.2 Qué tiene que contener el repositorio

El repositorio, en el commit entregado, tiene que contener:

1. **Código fuente** con una estructura clara, por ejemplo:
   ```
   .
   ├── app.py                  # punto de entrada de Streamlit
   ├── agents/
   │   ├── ingestion/          # grafo, nodos, prompts, schemas
   │   └── analyst/            # grafo, nodos, prompts, schemas
   ├── services/               # pinecone, litellm, almacén de tablas, sandbox
   ├── data/sample_pdfs/       # PDFs usados para la demo
   ├── tests/
   ├── .env.example
   ├── pyproject.toml / requirements.txt
   └── README.md
   ```
2. **README.md** que incluya:
   - instrucciones de instalación y ejecución,
   - un diagrama de arquitectura de ambos grafos (Mermaid está bien; también podés exportar `graph.get_graph().draw_mermaid()`),
   - decisiones de diseño (librería de extracción, almacén de tablas, estrategia de chunking/embeddings, enfoque de sandboxing),
   - limitaciones conocidas.
3. **Al menos 2 PDFs de ejemplo legibles por máquina** con tablas (al menos uno con una tabla que continúe en otra página), más **al menos 8 preguntas de prueba** con sus respuestas esperadas, que incluyan:
   - al menos 2 preguntas que necesiten **más de una tabla**,
   - al menos 1 pregunta con la que **ninguna** tabla indexada esté relacionada (`cannot_answer`),
   - al menos 1 pregunta para la que exista una tabla relevante pero le falte un año, una entidad, una conversión de unidades o un valor de comparación necesarios (`cannot_answer` o `partial`).

   Para cada pregunta, anotá el estado esperado junto con la respuesta esperada.
4. **Informe de evaluación** (una sección del README o un `EVALUATION.md` aparte): resultados de correr tus preguntas de prueba (correcta/incorrecta, reintentos, notas), más **links a 3 trazas representativas de LangSmith** (compartidas públicamente o como capturas de pantalla).
5. **Video demo** (máximo 5 minutos) que muestre la ingesta, la exploración y al menos tres preguntas respondidas de punta a punta. Poné un link al video (por ejemplo, YouTube no listado, Google Drive) en el README; no subas el archivo de video al repositorio.

---

## 10. Criterios de evaluación

| Criterio                                                                               | Peso |
|----------------------------------------------------------------------------------------|:------:|
| **Agente de Ingesta** — calidad de la extracción, metadatos basados en el contexto, ciclo de validación, indexación | 25%  |
| **Agente Analista** — recuperación y selección de tablas, razonamiento multitabla, generación de código, ejecución y autocorrección | 30% |
| **Diseño en LangGraph** — estado claro, nodos bien delimitados, aristas condicionales razonables, límites de reintentos | 10% |
| **UI de Streamlit** — usabilidad, transparencia (se muestran código/tablas/salida), manejo de errores | 10%    |
| **Observabilidad** — trazas de LangSmith útiles, nombres de ejecución y metadatos significativos | 5%     |
| **Evaluación** — calidad del set de pruebas, reporte honesto de resultados y limitaciones | 10%    |
| **Calidad de código y documentación** — estructura, tipado, README, reproducibilidad, seguridad en la ejecución de código | 10% |

### Bonus (hasta +10%)

- **Human-in-the-loop:** permitir que el usuario confirme o modifique las tablas seleccionadas antes de generar el código (`interrupt` de LangGraph).
- **Visualización:** el Analista también genera un gráfico (por ejemplo, con matplotlib/plotly) que se muestra en Streamlit cuando corresponde.
- **PDFs escaneados:** soporte de OCR para páginas o tablas compuestas solo por imágenes (fuera del alcance obligatorio; ver Sección 5.1).
- **Evals automatizados:** un dataset + evaluador de LangSmith que puntúe automáticamente tus preguntas de prueba.

---

## 11. Reglas e integridad académica

- Podés usar asistentes de programación con IA, pero tenés que entender y poder explicar **cada parte** de tu entrega. El equipo docente puede pedirte que expliques tu código en una instancia oral.
- Dá crédito a cualquier código externo, tutorial o template en el que te hayas basado.
- **No** subas API keys al repositorio. Una clave filtrada implica una penalización en la nota y la revocación de la clave.
- Usá el proxy LiteLLM de forma responsable: cacheá resultados durante el desarrollo y evitá ciclos de reintento sin límite.
- Entregas tardías: se rigen por la política de la materia. No se aceptan entregas después de que las soluciones se discutan en clase.

---

## 12. Consejos

- Empezá con **un PDF simple** y hacé funcionar todo el pipeline de punta a punta antes de encarar diseños de tablas difíciles.
- Guardá una **pequeña vista previa** de cada tabla (las primeras filas) en los metadatos de Pinecone; ayuda mucho durante la selección de tablas.
- Pasale al prompt de generación de código los nombres **exactos** de las columnas del DataFrame y sus dtypes; la mayoría de las fallas de código vienen de nombres de columna adivinados.
- Usá las trazas de LangSmith para depurar una mala selección de tablas antes de retocar los prompts a ciegas.
- Pedile al Analista que imprima su resultado final en un formato legible por máquina (por ejemplo, una línea JSON) para poder parsearlo de forma confiable.

¡Éxitos!
