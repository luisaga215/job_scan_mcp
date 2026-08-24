# Error Log — Sesión de prueba (MCP job-scan-mcp)

Sesión del 2026-08-22. Estado: pendientes de arreglar salvo los marcados como FIXED.

## 1. `sync_cv` calculó `experience_years` incorrecto (5.5 en vez de ~6.8)

- **Síntoma:** El perfil extraído de `resume_lc.pdf` reportó 5.5 años; el cálculo
  real desde fechas (Nov 2019 → Ago 2026) es 6.8 años.
- **Causa raíz:** `cv_service.py:54-64` delega el campo `experience_years` al LLM
  (`with_structured_output(ParsedProfile)`). Es un valor **adivinatorio y no
  determinístico**: el LLM hace aritmética de fechas sin conocer la fecha actual y
  sin reglas explícitas. En este caso trató "Present" como ~mayo 2025 en vez de
  hoy (ago 2026). No es un crash; es un fallo de diseño.
- **Fix sugerido:**
  - Incluir la fecha actual (`date.today().isoformat()`) en el prompt del parser.
  - Calcular `experience_years` de forma determinística con regex sobre rangos de
    fechas en el CV (mes/año → mes/año), sumando y evitando overlaps (ej. Terra
    Capital 2023 se superpone con Dematic, no debe duplicarse).
  - Usar el valor determinístico como autoritativo y el del LLM solo como apoyo.
- **Nota:** Ya se corrigió el dato en BD manualmente (6.8), pero el parser seguirá
  regresándolo mal hasta que se aplique el fix.
- **Estado:** PENDIENTE

## 2. `ParsedProfile` no tiene campo de autorización de trabajo / visa

- **Síntoma:** El caso de uso "aplicar en EE. UU. sin visa, requiere sponsorship /
  relocation" no se puede expresar en el perfil.
- **Causa raíz:** `models.py:11-19` (`ParsedProfile`) no incluye
  `work_authorization` / `visa_status` / `relocation_ok`. El screening y la
  evaluación profunda no pueden juzgar elegibilidad migratoria de forma explícita.
- **Fix sugerido:** Agregar campos (`visa_status`, `requires_sponsorship`,
  `relocation_willing`) y propagarlos a los prompts de screening/evaluación.
- **Estado:** PENDIENTE

## 3. `fetch_and_filter_jobs` no filtraba por sponsorship / relocation

- **Causa raíz:** `apply_filters` solo manejaba remoto y salario; no había
  detección de visa/relocation.
- **Fix aplicado:**
  - `job_service.analyze_visa_fit()`: detecta keywords positivos/negativos
    (H-1B, TN, sponsorship, relocation, "must be authorized", etc.) con
    precedencia negativa (ej. "no visa sponsorship" no se lee como positivo).
  - Nuevos flags en `Job`: `sponsorship`, `relocation_support`, `us_eligible`,
    `visa_keywords` (+ migración SQLite en `database.py`).
  - `require_visa_friendly=True`: modo estricto que conserva solo vacantes con
    sponsorship/relocation explícito (expuesto en `fetch_and_filter_jobs`).
  - `fetch_and_save_jobs` también hace backfill de flags en re-fetches.
  - Screening prioriza los jobs con sponsorship/relocation (ordenado en repo).
- **Estado:** FIXED (2026-08-22)

## 4. Screening / deep evaluation ignoraban elegibilidad migratoria

- **Fix parcial aplicado:** El prompt de `run_fast_screening` ahora inyecta los
  señales de visa/relocation del job y ordena rechazar postings hostiles
  ("must be authorized to work", "US citizens only", "no visa sponsorship").
- **Pendiente:** `run_deep_evaluation` aún no inyecta estado migratorio del
  candidato ni lo agrega a `DeepEvaluationResult.red_flags`.
- **Estado:** PARCIAL (screening FIXED; evaluación PENDIENTE)

## 5. `get_pipeline_status` mostraba defaults hardcodeados de OpenAI

- **Síntoma:** Reportaba `openai (default)` / `gpt-4o-mini` / `gpt-4o` aunque el
  `.env` apuntaba a Gemini.
- **Fix aplicado:** `mcp_server.py` ahora lee `config.DEFAULT_*_MODEL`.
- **Estado:** FIXED (2026-08-22)

## 6. `.env` usaba modelos no disponibles + `load_dotenv` no encontraba el `.env`

- **Síntoma:** `gemini-1.5-flash`/`gemini-1.5-pro` devolvían 404 (modelos retirados
  para cuentas nuevas); además, al lanzar el server desde otro CWD, el `.env` del
  proyecto no se cargaba y `GEMINI_API_KEY` quedaba vacío.
- **Fix aplicado:** `.env` → `gemini/gemini-3.6-flash`; `config.py` ahora hace
  `load_dotenv()` del CWD y fallback al root del proyecto.
- **Estado:** FIXED (2026-08-22)

## 7. Extras menores (observados, sin impacto crítico)

- `sync_cv` usa el LLM de la etapa `fast_screening` para parsear el CV; si se
  configura un modelo barato, la calidad de extracción baja. Considerar un modelo
  dedicado.
- El resume contiene un rol solapado (Terra Capital 2023 con Dematic); el parser no
  maneja overlaps → riesgo de doble conteo de experiencia.

## 8. Tools de larga duración excedían el timeout del cliente MCP

- **Síntoma:** `run_fast_screening(batch_size=151)` (y por extensión
  `fetch_and_filter_jobs`) se cortaron con `MCP error -32001: Request timed out`;
  el pipeline quedó intacto (los jobs seguían PENDING) porque el cliente mató la
  petición mientras Gemini procesaba.
- **Fix aplicado:**
  - Batching nativo por chunks (`SCREENING_CHUNK_SIZE=10`, `EVALUATION_CHUNK_SIZE=5`).
  - **Presupuesto de tiempo** (default 25s): la tool devuelve resultados parciales
    (`partial: True`, `pending_remaining`) en vez de bloquear al cliente.
  - Timeout estricto por job (`asyncio.wait_for`: screening 40s, evaluación 60s)
    para que un LLM colgado no cuelgue el lote.
  - Semáforos estrictos (concurrencia 3) configurables vía las tools.
  - `opencode.json` → `experimental.mcp_timeout: 600000` (aplica al reiniciar).
- **Estado:** FIXED (2026-08-22)

## 9. Concurrencia: "Session is already flushing" (causa raíz del screening atascado)

- **Síntoma:** En la prueba en vivo, 151 jobs quedaron PENDING con 0 progreso.
  Cada job fallaba con `Session is already flushing`.
- **Causa raíz:** Los tasks concurrentes (`asyncio.gather` con semáforo)
  compartían el MISMO `AsyncSession` de SQLAlchemy y llamaban a
  `repo.save_job()` (flush) simultáneamente → SQLAlchemy lanza "Session is
  already flushing".
- **Fix aplicado:** Las llamadas al LLM se ejecutan en paralelo, pero la
  **persistencia es secuencial** en el batch (después del `gather`), evitando
  flushes concurrentes sobre el session compartido.
- **Estado:** FIXED (2026-08-22)

## 10. DeepSeek no soporta `response_format` (structured output de LangChain)

- **Síntoma:** Con `provider=deepseek/model=deepseek-chat`, todos los jobs
  fallaron con `400 - This response_format type is unavailable now`.
- **Causa raíz:** `with_structured_output(PydanticSchema)` en `ChatOpenAI` envía
  `response_format` (json_object/json_schema), que la API de DeepSeek no soporta.
- **Fix aplicado:** `DeepSeekChatLLM(ChatOpenAI)` en `llm_factory.py` sobreescribe
  `with_structured_output` para devolver `JSONStructuredLLM`, un wrapper que pide
  JSON puro al modelo, extrae el bloque `{...}` y valida con Pydantic (retryable).
  Los call sites (screening/evaluación/cv) quedan sin cambios.
- **Estado:** FIXED (2026-08-22)

## 11. El reporte HTML salía vacío (faltaba inyectar los jobs)

- **Síntoma:** El dashboard abría pero no mostraba jobs (tabla vacía).
- **Causa raíz:** `report_template.html` leía `window.jobsData || []` (L386) pero
  NUNCA definía `window.jobsData`. `report.py` pasaba `jobs_json` al render, y el
  template no tenía ninguna salida `{{ ... }}` para inyectarlo → array vacío.
- **Fix aplicado:** `report.py` pasa la lista `jobs=formatted_jobs`; el template
  ahora emite `window.jobsData = {{ jobs | tojson }};` (tojson escapa `</script>`
  y caracteres no seguros). Test de regresión `tests/test_report.py`.
- **Nota:** El tool MCP `generate_html_report` de la sesión viva aún usa el código
  viejo; el fix aplica tras reiniciar el server.
- **Estado:** FIXED (2026-08-22; requiere reinicio para que aplique en el tool MCP)

## 12. UI/reporte: sin historial, sin filtros por fecha y filtros insuficientes

> Reportado por el usuario 2026-08-23. **NO ARREGLADO — pendiente de diseño.**

### 12.1 No existe historial de reportes

- `generate_html_report` siempre sobreescribe el mismo `reports/index.html`.
- No hay snapshots por ejecución ni forma de navegar a reportes anteriores.
- No se puede comparar "qué encontré la semana pasada vs hoy".
- **Idea:** generar `reports/index-<fecha-hora>.html` + un índice, y en la UI un
  selector de "reporte/ejecución".

### 12.2 Los jobs no tienen dimensión temporal usable

- El modelo `Job` sí tiene `date_posted` y la card lo muestra
  (template L267-268: `job.date_posted || 'Recently'`), pero:
  - **No hay filtro por fecha de publicación** (rango, "últimos 7/14/30 días").
  - **No hay sort por fecha**.
  - **No hay priorización por frescura**: un job de hace 3 semanas pesa igual que
    uno de hoy, y el usuario no quiere revisar viejos con la misma prioridad.
  - Cuando `date_posted` es `None`, la UI muestra "Recently" → falso positivo.
- **Idea:** filtro rango de fechas + sort por fecha + badge "NUEVO" para
  `days_since_posted <= 3/7`, y opción de ocultar jobs viejos por defecto.

### 12.3 Filtros actuales (pocos y deficientes)

Estado actual de `filteredJobs` (template L427-472):
1. Búsqueda de texto: solo `title`, `company`, `location`.
2. Remote filter: `<select>` con 3 opciones: All / Remote Only / **On-site + Hybrid juntos**.
3. State filter: PENDING/RELEVANT/REJECTED/EVALUATED.
4. Fit score mínimo (`minFit`).
5. Sort: solo `fit`, `interview_probability`, `company` (3 botones).

Faltan (candidatos):
- **Salario**: min/max por rango, moneda, "con salario publicado".
- **Fecha de publicación**: rango / últimos N días (ver 12.2).
- **Sponsorship/visa**: `sponsorship`, `relocation_support`, `us_eligible`
  (ya están en el modelo pero no expuestos en UI).
- **Seniority** (`true_seniority_alignment`) y `application_friction`
  (Low/Medium/High).
- **Red flags**: con/sin, conteo.
- **Ubicación**: multi-ciudad/estado; **empresa**: multi-select.
- **Puntaje compuesto** o combinaciones (fit + probability).

Sorts faltantes: fecha, salario, seniority, friction, red flags, sponsorship.

### 12.4 Bug de UX: "On-site / Hybrid" están combinados

- En el `<select>` remoto la opción es `On-site / Hybrid` (L126), y en el
  `filteredJobs` (L447-449) el filtro `onsite` es `j.is_remote !== true`, es
  decir **hibrido cae como on-site**.
- Se pide: opciones **separadas** (On-site / Hybrid / Remote) y **multi-select**
  (seleccionar ninguna, una, dos o las tres), estilo filtros de LinkedIn para
  búsqueda de posiciones.

### Estado
PENDIENTE (diseño propuesto arriba; no implementar aún).

## 13. Backlog de mejoras UI/UX (ideas varias — continuar mañana)

> Backlog abierto 2026-08-23. Combinar con las ideas del usuario mañana. Nada implementado.

### Filtros avanzados
- Multi-select de ubicación (ciudades/estados) estilo chips.
- Multi-select de empresa.
- Rango de salario (slider dual min-max) + toggle "solo con salario publicado".
- Rango de fecha de publicación (últimos 7/14/30 días o custom).
- Toggles de sponsorship/visa: `sponsorship`, `relocation_support`, `us_eligible`.
- Fit por seniority (`true_seniority_alignment` match vs desalineado).
- `application_friction`: Low/Med/High.
- Conteo de red flags: 0 / ≥1 / ≥2.
- Estado multi-select (combinar PENDING+RELEVANT, etc.).
- Fit score con rango (no solo mínimo).

### Sorts
- Por fecha de publicación.
- Por salario (min).
- Por seniority alignment y friction (Low primero).
- Por número de red flags (menos primero).
- Multi-sort primario/secundario.
- Persistir preferencia de sort.

### Cards / visualización
- Badge "NEW" para jobs recientes (≤3/7 días) y badge de sponsorship/relocation.
- Mostrar fecha relativa ("hace 2 días") en vez de fecha cruda.
- Chips del core stack overlap en la card.
- Link "Aplicar" destacado a `job_url`.
- Resaltar jobs sin salario publicado.

### Encabezado / estadísticas
- Contador de resultados del filtro activo + avg fit del filtro actual.
- Distribución de fit scores, friction y red flags (mini-barras).
- Top ubicaciones/empresas del set actual.

### Experiencia general
- Persistir filtros en `localStorage`.
- Exportar la lista filtrada (CSV / texto) y copiar enlaces.
- Comparación side-by-side de 2-3 jobs.
- Dark/light toggle, responsive, atajos de teclado.
- Render virtualizado/paginado (151+ jobs sin colgar el DOM).

### Datos / tracking
- Guardar fecha de scraping por job y mostrar "antigüedad vs cuándo lo encontré".
- Estado de aplicación por job (aplicado/descartado/interesado) persistido en BD.
- Notas por job.

### Historial / ejecuciones (relacionado a 12.1)
- Selector de ejecuciones anteriores; diff entre ejecuciones (jobs nuevos/perdidos).
- Export snapshot JSON por ejecución.

### Inspiración UI: LinkedIn Jobs
- **Panel de filtros lateral** (left rail) con secciones y checkboxes multi-select
  CON conteo por opción (Date Posted, Remote, Easy Apply, Experience Level,
  Company, Industry, Job Function, Location, Salary Estimate).
- **Easy Apply** como filtro/toggle (apply en un clic).
- **Layout maestro-detalle**: lista compacta a la izquierda + panel de detalle del
  job seleccionado a la derecha (en vez de acordeones).
- **Guardar jobs** (bookmark) → pestaña "Saved jobs".
- **Toggles de tiempo**: "Past 24h / Past week / Past month" (chips).
- **Sort**: Most recent / Most relevant como toggle arriba de la lista.
- **Contador** "showing X of Y" + resultados por filtro activo.
- Autocomplete en el buscador (títulos, keywords, empresas).
- Match % en la card del job.

### Inspiración UI: Jobright AI
- **Fit score prominente** (badge/anillo con %) como primer elemento de cada card.
- **Razones de match**: bullets de "por qué encaja" (skills del candidato vs
  requeridos del job) visibles sin expandir.
- **Skill gap**: "requerido pero te falta" vs "cubierto".
- **Indicadores upfront**: salary, remote/onsite, visa/sponsorship y ubicación en
  la card, sin abrir detalle.
- **Ranking diario curado** por fit (el mejor primero).
- **Acciones rápidas** tipo swipe: "Interesado" / "No me interesa" (→ tracking).
- **Kanban de aplicaciones**: Applied / Interviewing / Offered / Rejected.
- **Insights de empresa** y sugerencias de adaptar el CV por rol.

### Datos del post: aplicantes, links multi-plataforma y métricas
- **Cantidad de aplicantes**: mostrar "# aplicantes" en la card/detalle
  (patrón LinkedIn). Nota de factibilidad: jobspy puede no exponerlo; habría que
  scrapearlo por plataforma o marcarlo como "desconocido".
- **Múltiples links de la misma req** en distintas plataformas (Indeed, LinkedIn,
  ZipRecruiter, Glassdoor, sitio de la empresa):
  - Hoy el modelo `Job` guarda UN solo `job_url` y dedup por
    hash(company|title|location) → la URL de la 2ª plataforma se pierde.
  - **Idea:** campo `platform_urls` (JSON: [{platform, url}]) + mantener el hash
    dedup pero concatenando fuentes; badge por plataforma en la card.
- **Datos relacionados** por post: fecha por fuente, rango salarial por fuente,
  "posted hace X", Easy Apply sí/no, fuente preferida para aplicar, si aparece en
  N plataformas (mayor señal = más real/activo).
- **Métricas derivadas**: score de "actividad/frescura" (fecha + nº plataformas),
  alerta si la req salió en más de una plataforma hace <7 días.

### Recomendaciones estratégicas (usuario — 2026-08-23) → prioridad alta
Basadas en estándares de LinkedIn, Jobright AI y ATS modernos.

**1. Panel de "Acciones Rápidas" y Estado de Aplicación (workflow de búsqueda)**
- Cerrar el ciclo operativo: el pipeline evalúa pero no permite gestionar la
  postulación.
- Kanban / estados interactivos en cada card: **Por aplicar, Aplicado, Entrevista,
  Descartado** → actualizan la BD (o JSON local) para usar el reporte como tablero
  Kanban diario.
- Botón **"Quick Apply" con portapapeles**: copia un resumen estructurado (o los
  bullets de por qué haces match) para pegarlo en el formulario de la empresa.

**2. Análisis de "Skill Gap" visual (desglose de stack)**
- Perfil requiere precisión en el stack (Java, AWS, Distributed Systems).
- **Matriz de coincidencia de stack** por card (barra/chips de colores):
  - 🟢 Skills que posees y pide el rol (ej. Java, AWS).
  - 🟡 Skills transferibles (ej. conoces GCP pero piden AWS).
  - 🔴 Gaps críticos (ej. piden Kubernetes experto y lo tienes básico).

**3. Estimador de Competitividad / Demanda**
- **Ratio de atracción**: aplicantes (si la fuente lo provee) vs días publicado →
  indicador visual (ej. "Baja competencia / Alta demanda") para priorizar vacantes
  donde tengas más probabilidad de ser visto antes de que se saturen.

**4. Tarjeta "Resumen Ejecutivo de IA" para Entrevistas**
- Al expandir una vacante (o modal dedicado), además de pros/contras, mostrar un
  bloque **"Cheat Sheet para Entrevista"**:
  - Qué destacar de tu experiencia (LLM conecta tu paso por Audible/Dematic con
    los requisitos del puesto).
  - Posibles preguntas trampa basadas en las Red Flags detectadas.

**5. Exportación y Sincronización de Persistencia**
- **Exportar CSV / JSON**: botón en la cabecera para descargar los datos filtrados
  (Excel/Notion).
- **Persistencia de filtros (localStorage)**: al recargar o reabrir, recordar
  preferencias (ej. Remoto + Salario > $130k + Fit > 85%).

### Estado
PENDIENTE — continuar diseño mañana.

### Implementado (2026-08-23) — avance del backlog
- **12.4 FIXED:** Modalidad multi-select separada: Remote / Hybrid / On-site
  (checkboxes, ninguna = todas; ya no se combinan onsite+hybrid).
- **12.2/12.3 PARCIAL→implementado:**
  - Filtros: fecha (24h/7d/30d), salario (min/max + solo con salario), visa
    (sponsorship/relocación/us_eligible), fricción (Low/Med/High), red flags,
    rango de fit, estado multi-select, ubicación y empresa multi-select,
    estado de postulación.
  - Sorts: fit, prob, salario, fecha, fricción, red flags, empresa + dirección
    asc/desc. Contador "X de Y" + avg fit del filtro activo.
  - Fechas relativas, badge NEW (≤7d), badges visa/relocación.
  - Skill gap chips 🟢match / 🟡transferible / 🔴gap (determinístico en
    `report.compute_skill_chips`, sin LLM extra).
- **Kanban (localStorage):** estados Por aplicar / Aplicado / Entrevista /
  Descartado por job (`jobscan.app.v1`) + filtro por estado de postulación.
  Persistencia en BD vía MCP: PENDIENTE.
- **Quick Apply:** copia pitch estructurado al portapapeles (con fallback).
- **Cheat Sheet de entrevista** (determinístico): qué destacar + preguntas trampa
  por red flags. Versión LLM: PENDIENTE.
- **Export CSV / JSON** de la lista filtrada.
- **Persistencia de filtros** en `localStorage` (`jobscan.filters.v1`).
- **Theme switch** Light / Dark / System (inversión CSS + `prefers-color-scheme`,
  persistido en `jobscan.theme.v1`).
- **Nota:** el tool MCP `generate_html_report` de la sesión viva usa el código
  anterior; los cambios aplican al reiniciar el server.
- **Validación:** 42 pytest (86% cov); 17 asserts funcionales de JS headless
  (filtros, sorts, persistencia, kanban, cheat sheet, export, tema) en Node.

### Implementado (2026-08-23) — ronda 2 (feedback del usuario)
- **Fix theme buttons:** estaban fuera del scope de `x-data` (en el header, con el
  app en `<main>`) → no funcionaban. Se movió `x-data="jobScreeningApp()"` al
  contenedor raíz.
- **Export clarificado:** etiqueta "Exportar: CSV / JSON" + tooltips; "Reset" →
  "Limpiar filtros".
- **Fechas:** la card muestra fecha relativa + fecha real (`date_posted`); filtro
  por **rango custom** (desde/hasta) además de chips 24h/7d/30d.
- **Historial de reportes:** `generate_report` ahora escribe `index.html` +
  snapshot `report-<ts>.html` y actualiza `manifest.json` (últimos 30). Side panel
  "Reportes" en la UI lista los snapshots con fecha, conteos, avg fit y modelos →
  navegación entre reportes.
- **Estados de aplicación persistidos:** columna `application_status` (+
  `application_status_updated_at`) en `Job` con migración SQLite; tool MCP
  `set_job_application_status(job_id, status)`; el reporte embebe el estado y la
  UI lo siembra en el kanban. (Requiere reinicio para que el tool esté disponible
  en el cliente.)
- **Vista "Aplicadas":** tab dedicado que consolida vacantes aplicadas/
  entrevista/descartadas (no por reporte), con filtro por estado, fecha en que se
  marcó, Quick Apply y link a la vacante.
- **Tests:** +manifest/snapshot, +application_status en payload, +tool
  `set_job_application_status`; fixture que aísla `REPORTS_DIR` (los tests ya no
  ensucian el historial real).
- **Nota estados del pipeline:** a la hora de generar el reporte los estados
  PENDING/RELEVANT suelen ser 0 (todo quedó EVALUATED/REJECTED); los estados
  transitorios solo se ven si generas a mitad del pipeline. El estado persistente
  es el de aplicación (kanban).
- **Validación ronda 2:** 45 pytest (87% cov); 23 asserts funcionales JS headless
  (incluye rango de fechas, vista aplicadas, persistencia de estados, manifest,
  tema).

### Implementado (2026-08-23) — ronda 3 (feedback: "pantalla en blanco" + rejected)
- **Investigación "pantalla en blanco":** el reporte renderiza correctamente en
  Chrome headless (sin errores de consola). La causa probable en el navegador del
  usuario era localStorage con esquema viejo de filtros o cache del archivo.
- **Guard de esquema de filtros:** `LS_FILTERS` bump a `v2` con campo `v`; si el
  estado guardado no coincide con el esquema actual, se descarta (evita UI rota
  por estados viejos).
- **Rejected ocultos por defecto:** nuevo toggle `Ocultar rechazados` (default ON);
  el explore ya no muestra los REJECTED salvo que se desactive. Esto también
  reduce la cantidad de cards iniciales (151 → ~33).
- **Validación ronda 3:** 45 pytest; 31 asserts funcionales JS headless; dump DOM
  de Chrome headless confirma cards renderizadas (25× "Backend Engineer").

## Cobertura / tests (ciclo de desarrollo 2026-08-22)

- `uv run pytest --cov=src`: **30 passed** (3 corridas consecutivas), 84% cobertura.
- Tests nuevos: timeouts por job, partials por presupuesto de tiempo, prompts con
  contexto de visa, ordenamiento priorizado por visa, análisis de visa,
  `require_visa_friendly`, flags persistidos, params de batching en tools MCP.
