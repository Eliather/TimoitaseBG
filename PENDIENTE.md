# Lista de Pendientes y Mejoras — TimoitaseBG

Documento de trabajo para abordar de forma incremental los bugs, mejoras técnicas, ajustes de UX/seguridad y tareas de CI/CD detectadas en el proyecto.

## Índice

- [Bloque 1: Bugs críticos, consistencia de estado y limpieza](#bloque-1-bugs-críticos-consistencia-de-estado-y-limpieza)
- [Bloque 2: Arquitectura, concurrencia, VRAM y rendimiento](#bloque-2-arquitectura-concurrencia-vram-y-rendimiento)
- [Bloque 3: Seguridad, integridad de modelos, UX e i18n](#bloque-3-seguridad-integridad-de-modelos-ux-e-i18n)
- [Bloque 4: CI/CD, empaquetado, pruebas y documentación](#bloque-4-cicd-empaquetado-pruebas-y-documentación)
- [Bloque 5 (Especial): Modularización Inteligente (Anti-Monolito)](#bloque-5-especial-modularización-inteligente-anti-monolito)

---

## Bloque 1: Bugs críticos, consistencia de estado y limpieza

- [x] **Corregir borrador manual en imágenes RGB sin canal alfa**
  - **Solución aplicada:** Normalización automática a `Format_RGBA8888` en `CanvasWidget.set_image` / `set_original_image` y en `BrushEngine` para evitar píxeles negros al borrar imágenes RGB.

- [x] **Aplicar operaciones globales a todos los fotogramas de una secuencia**
  - **Solución aplicada:** Propagación multi-fotograma en `start_restore` (`SequenceRestoreWorker`), `perform_autocrop` (`DocumentController.apply_autocrop` con union-bbox) y `perform_isolate_subject` (`DocumentController.apply_isolate_subject_sequence`).

- [x] **Completar limpieza de estado al abrir videos y animaciones**
  - **Solución aplicada:** Centralizado en `MainWindow._reset_workspace_ui()` para imágenes, animaciones y videos.

- [x] **Resolver conflicto de hojas QSS y tema de chips YOLO**
  - **Solución aplicada:** Eliminado `app/ui/styles.qss` y su carga en `app/main.py`; los paneles aplican tema dinámico uniforme mediante `apply_theme()`.

- [x] **Alinear `model_fetcher`, variantes `_int8.onnx` y la caché de `rembg`**
  - **Solución aplicada:** Añadido `resolve_local_model` y `load_rembg_session` compartidos en `app/core/model_fetcher.py` (con soporte `_int8.onnx`), usados en `bg_remover.py`, `gpu_worker.py` y `SAMManager.is_available()`.

- [x] **Corregir tamaño inicial vs. mínimo de ventana**
  - **Solución aplicada:** `MainWindow` inicializa con `max(1200, WINDOW_MIN_WIDTH)` × `max(780, WINDOW_MIN_HEIGHT)`.

- [x] **Soportar Drag & Drop de imágenes en memoria (`hasImage()`)**
  - **Solución aplicada:** `DropZoneWidget` y `CanvasWidget` emiten `imageDropped` para `mime.hasImage()` conectada a `MainWindow.load_new_pil_image`.

- [x] **Eliminar código muerto y artefactos residuales del workspace**
  - **Solución aplicada:** Removidos `_probe_restorer_gpu_worker`, `_draw_empty_state`, `refine_with_vitmatte`, `app/models/yolo11n-seg.tmp`, `app/models/lama_fp32*.onnx` y duplicados en `app/models/models/`.

---

## Bloque 2: Arquitectura, concurrencia, VRAM y rendimiento

- [x] **Desacoplar clases monolíticas de interfaz**
  - **Solución aplicada:** Extraídos `DocumentController`, `AICoordinator`, `ToolSidebar` + 4 paneles, `BrushEngine`, `SelectionTool`, `TopBar`, `SequenceBar`, `StatusBar` y `register_shortcuts`.

- [x] **Aislar MobileSAM y YOLO11 de la gestión de VRAM del proceso principal**
  - **Solución aplicada:** `YOLODetector` implementa `unload_sessions()` y respeta `gpu_manager.is_gpu_enabled()`; `AICoordinator.switch_device_mode` libera sesiones MobileSAM y YOLO11 al conmutar GPU/CPU.

- [x] **Permitir restauración GPU con tiling en imágenes mayores a 1080p**
  - **Solución aplicada:** `app/core/restorer.py` restringe `MAX_GPU_PIXELS` solo cuando `not use_tiling`.

- [x] **Implementar cancelación cooperativa y cierre limpio de hilos**
  - **Solución aplicada:** Señal `cancelled` e `isInterruptionRequested()` en workers, `cancel_check` en `model_fetcher.ensure_model_file`, botón Cancelar en `StatusBar` y `get_gpu_manager().shutdown(blocking=False)` en `AICoordinator.shutdown_workers`.

- [x] **Mover exportación de video/animaciones fuera del hilo UI**
  - **Solución aplicada:** `SaveSequenceWorker` ejecuta `doc.save_to_path` en segundo plano al exportar secuencias desde la UI (la carga y `SequenceSession` conservan lectura compatible con acceso aleatorio por fotograma).

- [x] **Reducir asignaciones por evento al pintar con pincel suave o selección activa**
  - **Solución aplicada:** Reutilización de `_scratch_buf` y composición acotada por rectángulo sucio en `app/ui/canvas/brush_engine.py`.

---

## Bloque 3: Seguridad, integridad de modelos, UX e i18n

- [x] **Validar integridad con `SHA256` y habilitar reanudación HTTP en descargas**
  - **Solución aplicada:** Soporte `Range` HTTP `206` sobre `.tmp` parciales, `verify_sha256` y `MODEL_SHA256` en `app/core/model_fetcher.py`.

- [x] **Gestionar degradación SSL con advertencia explícita y verificación**
  - **Solución aplicada:** `ensure_model_file` registra advertencia clara ante fallo de verificación SSL, expone `allow_insecure_fallback` y valida integridad mediante `MODEL_SHA256` cuando hay digest registrado.

- [x] **Corregir `scripts/quantize_models.py`**
  - **Solución aplicada:** Usa `MODELS_DIR` por defecto, `sys.exit(1)` y validación posterior en `onnxruntime.InferenceSession`.

- [x] **Eliminar textos obsoletos de inpainting en la UI e i18n**
  - **Solución aplicada:** Actualizada la clave `brush_desc` en los 4 idiomas (`es`, `en`, `ru`, `zh`) describiendo el pincel restaurador y borrador reales.

- [x] **Completar traducción en vivo (`retranslate_ui`) y eliminar textos crudos en español**
  - **Solución aplicada:** `StatusBar`, `TopBar`, `ToolSidebar` y los 4 paneles implementan `retranslate_ui()`; `MainWindow` usa claves `tr(...)` en estados y diálogos.

- [x] **Reportar fallos por archivo en procesamiento por lotes**
  - **Solución aplicada:** `BatchWorkerThread` acumula `(archivo, motivo)` en `errors`, emite `finishedWithErrors` y `MainWindow` presenta el resumen de archivos fallidos.

- [x] **Migrar persistencia de idioma fuera de `$HOME`**
  - **Solución aplicada:** Migrado a `QSettings` en `app/i18n.py`.

---

## Bloque 4: CI/CD, empaquetado, pruebas y documentación

- [x] **Añadir compuertas de pruebas y análisis estático en CI**
  - **Solución aplicada:** Añadido job previo `test` (`compileall` + `pytest` headless) en `.github/workflows/release_build.yml` con `needs: test` en `build`.

- [x] **Corregir metadatos de `build.spec`, MSIX multi-idioma, iconos y checksums**
  - **Solución aplicada:** `build.spec` lee `APP_VERSION` dinámicamente e integra iconos `.ico`/`.icns`; `scripts/package_release.py` genera iconos multiplataforma y declara `es-ES`, `en-US`, `ru-RU` y `zh-CN` en `AppxManifest.xml`; CI genera `SHA256SUMS.txt`.

- [x] **Corregir descripción de `iniciar.bat` en `README.md`**
  - **Solución aplicada:** `README.md` describe con exactitud la función de `iniciar.bat`.

- [x] **Actualizar versión en `THIRD_PARTY_NOTICES.md`**
  - **Solución aplicada:** Actualizado encabezado a `TimoitaseBG v1.2.3`.

- [x] **Actualizar `scripts/download_releases.py`**
  - **Solución aplicada:** Incorpora CLI `--tag` leyendo `APP_VERSION`, docstring de los 6 artefactos y verificación de `SHA256SUMS.txt`.


---

## Bloque 5 (Especial): Modularización Inteligente (Anti-Monolito)

### Análisis del estado actual

Actualmente casi la mitad de todo el código bajo `app/` se concentra en tres archivos de interfaz (`app/ui/main_window.py` con 1,900 líneas, `app/ui/toolbar.py` con 1,699 líneas y `app/ui/canvas_widget.py` con 1,358 líneas — **4,957 líneas en total**), además de tres módulos núcleo extensos (`app/i18n.py` con 884 líneas, `app/core/bg_remover.py` con 867 líneas y `app/config.py` con 743 líneas).

Esta concentración explica varios de los fallos recurrentes del proyecto:
1. **Rutas de estado fragmentadas:** `MainWindow` manipula directamente `image_state`, `sequence_session`, `sam_manager`, `yolo_detector`, `gpu_manager` y comparador, lo que provoca resets incompletos (`open_image_file` en videos/GIFs) y operaciones que olvidan propagarse a secuencias (`start_restore`, `perform_autocrop`, `perform_isolate_subject`).
2. **Lógica de visión y E/S mezclada con Qt:** `_refine_yolo_subject_mask` vive dentro de `MainWindow`, dificultando probar algoritmos sin instanciar una ventana Qt completa; y `MainWindow` contiene más de 40 importaciones locales dispersas dentro de sus métodos.
3. **Paneles monolíticos sin contrato común:** `ToolSidebar` construye los 4 paneles en métodos de hasta 321 líneas (`_setup_brush_panel`), provocando olvidos en `retranslate_ui()` y estilos QSS desincronizados al cambiar de tema (chips YOLO).
4. **Cámara, pintado y selección en una sola clase:** `CanvasWidget` mezcla transformaciones de viewport, motor de pincel, varita mágica y renderizado de UI vacía/comparación.

### Principios de modularización (sin sobreingeniería)

- **Separación por responsabilidad de cambio:** Core/CV (cero `PySide6.QtWidgets`), Coordinadores/Estado (orquestación sin layout) y Vistas/Paneles (solo presentación y señales).
- **Evitar fragmentación excesiva:** Módulos cohesivos de 150–450 líneas en lugar de decenas de micro-archivos innecesarios.
- **Compatibilidad incremental:** Mantener re-exports y fachadas públicas en `MainWindow`, `ToolSidebar`, `CanvasWidget`, `app.config` y `app.core.bg_remover` para no romper la suite existente en `tests/` ni el empaquetado de `build.spec` (`datas = [('app/ui', 'app/ui')]`) durante la transición.

### Plan por fases

#### Fase 1: Extracción de lógica pura fuera de la UI y partición de Core/i18n
- [x] Mover `_refine_yolo_subject_mask` (`app/ui/main_window.py`) hacia `app/core/yolo_detector.py` (`refine_yolo_subject_mask`) para desacoplarlo de Qt y poder testearlo unitariamente (incluyendo `unload_sessions()` y verificación de GPU).
- [x] Dividir `app/core/bg_remover.py` (867 líneas) extrayendo las utilidades puras de pre/postproceso (`letterbox_*`, `apply_guided_filter*`, `postprocess_*`, `generate_trimap`, `fuse_clothing_protection_core_only`) a `app/core/postprocess.py`, re-exportando los símbolos usados por `tests/test_birefnet.py` y eliminando `refine_with_vitmatte`.
- [x] Extraer la detección de hardware GPU (`_query_windows_gpus_fast`, `get_gpu_name`, `detect_device`, `get_best_dml_device_id`, `is_integrated_gpu`, `is_gpu_available`, `get_inference_providers`, etc.) de `app/config.py` (743 líneas) a `app/core/hardware.py`, manteniendo re-exports en `app/config.py` para compatibilidad con los tests existentes.
- [x] Externalizar los diccionarios de `app/i18n.py` (884 líneas) a `app/locales/{es,en,ru,zh}.json` (agregando `('app/locales', 'app/locales')` en `build.spec`) y reducir `app/i18n.py` a un cargador ligero con `QSettings`.

#### Fase 2: Capa de Coordinadores (`app/controllers/`) para adelgazar `MainWindow`
- [x] Crear `app/controllers/document_controller.py` para centralizar `image_state`, `sequence_session`, historial undo/redo, carga/guardado/portapapeles, un único método de reseteo y helpers compartidos.
- [x] Crear `app/controllers/ai_coordinator.py` para unificar la orquestación de workers (`WorkerThread`, `SamEmbeddingWorker`, `YoloDetectionWorker`, `GifWorkerThread`, `BatchWorkerThread`), cierre y ciclo GPU/VRAM (`gpu_manager` + `SAMManager` + `YOLODetector`).

#### Fase 3: Descomposición modular de `ToolSidebar` (1,699 L -> shell + paneles)
- [x] Extraer los widgets reutilizables (`AutoResizingStackedWidget`, `ToggleSwitch`, `GpuToggle`, `_GpuToggleAdapter`) a `app/ui/widgets/toggles.py`.
- [x] Dividir los 4 paneles en `app/ui/panels/` (`bg_panel.py`, `restore_panel.py`, `brush_panel.py`, `batch_panel.py`), definiendo en cada panel un contrato uniforme `retranslate_ui()` y `apply_theme(is_dark: bool)` con señales tipadas y reestilización dinámica de chips YOLO.
- [x] Mantener `ToolSidebar` (`app/ui/toolbar.py`, 435 líneas) como contenedor ligero que conserva las propiedades/señales esperadas por `MainWindow` y los tests.

#### Fase 4: Descomposición de `CanvasWidget` y limpieza final de `MainWindow`
- [x] Extraer el motor de pincel y recorte de selección con buffer reutilizable a `app/ui/canvas/brush_engine.py`, dejando en `CanvasWidget` la cámara/zoom/pan, eventos y composición visual.
- [x] Extraer la barra superior (`CustomTopBar`, `TopBar`), diálogos pop-up (`FramelessPopup`) y barra de paginación de secuencias (`SequenceBar`) a `app/ui/components/` (`top_bar.py`, `sequence_bar.py`).
- [x] Extraer `SelectionTool` a `app/ui/canvas/selection_tool.py`, `StatusBar` y `register_shortcuts` a `app/ui/components/` y completar el adelgazamiento modular de `CanvasWidget` y `MainWindow`.
