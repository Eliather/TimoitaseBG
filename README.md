# TimoitaseBG

![Version](https://img.shields.io/badge/Version-1.2.3-7c3aed?style=for-the-badge)
![License](https://img.shields.io/badge/License-GNU_AGPLv3-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PySide6](https://img.shields.io/badge/GUI-PySide6_Qt6-41cd52?style=for-the-badge&logo=qt&logoColor=white)
![Platform](https://img.shields.io/badge/OS-Windows_%7C_macOS_%7C_Linux-lightgrey?style=for-the-badge)
![Inference](https://img.shields.io/badge/Inference-ONNX_Runtime_Local-005a9e?style=for-the-badge)

**TimoitaseBG** es una aplicación de escritorio multiplataforma para la eliminación de fondos, segmentación interactiva de objetos y restauración de imágenes, animaciones y secuencias de video de forma completamente local. Está construida sobre **Python 3.11+**, **PySide6 (Qt6)** y **ONNX Runtime**, organizada bajo una arquitectura modular por capas con aislamiento de procesos para inferencia acelerada por hardware y herramientas de edición pixel a pixel sobre el lienzo.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características Principales](#características-principales)
- [Compatibilidad de Hardware y Aceleradores](#compatibilidad-de-hardware-y-aceleradores)
- [Arquitectura y Funcionamiento Interno](#arquitectura-y-funcionamiento-interno)
  - [Estructura del Proyecto](#estructura-del-proyecto)
  - [Arquitectura Modular por Capas](#arquitectura-modular-por-capas)
  - [Ciclo de Ejecución y Aislamiento de GPU](#ciclo-de-ejecución-y-aislamiento-de-gpu)
  - [Detección de Hardware y Gestión de Memoria](#detección-de-hardware-y-gestión-de-memoria)
  - [Pipelines de Visión y Edición de Imagen](#pipelines-de-visión-y-edición-de-imagen)
  - [Modelos ONNX, Cuantización INT8, Reanudación HTTP e Integridad](#modelos-onnx-cuantización-int8-reanudación-http-e-integridad)
- [Atajos de Teclado y Ratón](#atajos-de-teclado-y-ratón)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Pruebas Automatizadas](#pruebas-automatizadas)
- [Compilación y Empaquetado Multiplataforma](#compilación-y-empaquetado-multiplataforma)
- [Atribución y Licencias de Terceros](#atribución-y-licencias-de-terceros)
- [Autora y Licencia](#autora-y-licencia)

---

## Descripción General

TimoitaseBG ejecuta modelos neuronales de segmentación, detección y superresolución directamente en el dispositivo del usuario sin telemetría, sin límites de resolución impuestos por servidores externos y sin subir archivos a la nube. Los modelos ONNX se descargan bajo demanda con reanudación HTTP (`Range`), verificación de integridad `SHA256` y soporte nativo para variantes cuantizadas `INT8`, permitiendo trabajar con ilustraciones, anime, fotografía de producto, retratos humanos, animaciones GIF/WebP/APNG y secuencias de video.

El flujo de trabajo combina inferencia automática (cuatro arquitecturas especializadas de eliminación de fondo, detección de sujetos con YOLO11-seg, segmentación por puntos con MobileSAM y escalado con Real-ESRGAN) con retoque manual no destructivo mediante pinceles de restauración/borrado, varita mágica de color o IA, operaciones booleanas de selección e historial de estados por imagen o fotograma.

---

## Características Principales

### 1. Eliminación de Fondo y Protección Semántica de Vestimenta

El motor de segmentación expone cuatro modelos ONNX seleccionables según el tipo de contenido:

| Identificador | Arquitectura | Uso Recomendado |
| :--- | :--- | :--- |
| `isnet-anime` | IS-Net (Anime) | Ilustración 2D, anime, manga, line-art y gráficos con bordes definidos. |
| `inspyrenet` | InSPyReNet Swin-B | Uso general de alta fidelidad (fotografía, cabello complejo y detalle fino). |
| `birefnet-general` | BiRefNet General | Escenas de alta complejidad y objetos con contornos intricados. |
| `u2net_human_seg` | U2-Net Human Seg | Segmentación especializada de siluetas humanas y vestimenta. |

- **Protección Semántica de Ropa**: Opción activable para `inspyrenet` y `birefnet-general` que ejecuta `u2net_human_seg` en paralelo para reforzar la retención de prendas blancas o de bajo contraste frente a fondos claros mediante inyección exclusiva en el núcleo erosionado, evitando halos externos.
- **Preservación Exacta de RGB**: Las máscaras alpha generadas se aplican respetando los canales RGB de la imagen original, con postprocesamiento morfológico, recuperación topológica de huecos internos y suavizado de bordes.

### 2. Segmentación Semántica e Interacción por Sujetos

- **Detección Multi-Sujeto (`YOLO11n-seg`)**: Detecta hasta 80 categorías de objetos COCO en una pasada, generando fichas interactivas estilizadas por tema en el panel lateral para inspeccionar, seleccionar o aislar individualmente cada elemento presente en la escena.
- **Segmentación Interactiva (`MobileSAM`)**: Codifica la imagen una sola vez (`(1, 256, 64, 64)`) y mantiene el embedding en memoria para ejecutar decodificaciones guiadas por clic en menos de 15 ms, además de colaborar con YOLO11-seg en el aislamiento fino de sujetos individuales (propagable a todos los fotogramas de una secuencia).

### 3. Restauración y Escalado de Alta Resolución

- **Superresolución (`Real-ESRGAN x4plus Anime 6B`)**: Soporta modos de mejora sin cambio de tamaño (`enhance_only` 1x), escalado `x2` y escalado `x4`, incluyendo propagación completa sobre todos los fotogramas de animaciones y videos mediante `SequenceRestoreWorker`.
- **Canal Alfa Independiente**: El canal de transparencia se separa del RGB antes de la inferencia neuronal y se escala por separado con interpolación Lanczos para conservar contornos limpios sin artefactos alrededor de áreas transparentes.
- **Tiling Adaptativo**: Procesa la imagen por mosaicos con solapamiento en GPU (incluso por encima de `1920x1080` cuando `use_tiling=True`), aplicando límite seguro de VRAM en pasadas sin tiling (`MAX_GPU_PIXELS`) y respaldo automático en CPU para evitar saturación de memoria en GPUs discretas o integradas.

### 4. Lienzo Interactivo, Pinceles y Varita Híbrida

- **Herramienta Pintar (Restauración)**: Recupera píxeles directamente desde la referencia original (`original_qimage`) asociada al estado actual, con controles de tamaño (`2` a `200 px`), suavizado (`0` a `100 %`), opacidad (`1` a `100 %`), dureza (`0` a `100 %`) y presets rápidos (`2, 10, 25, 50, 100 px`).
- **Herramienta Borrador**: Normaliza automáticamente imágenes fuente sin canal alfa a `Format_RGBA8888` para que el borrado manual (`CompositionMode_DestinationOut`) genere transparencia real con recorte estricto dentro de selecciones activas.
- **Varita Mágica (Color e IA)** (`SelectionTool`):
  - *Modo Color*: Selección por similitud RGBA con tolerancia ajustable (`0–255`), opción contigua mediante `floodFill` de 4 vecinos y suavizado anti-alias gaussiano.
  - *Modo IA*: Utiliza MobileSAM para seleccionar semánticamente el objeto bajo el cursor con un clic.
  - *Operaciones Booleanas*: Soporta modos `Nuevo`, `Añadir`, `Restar` e `Intersecar` (con atajos Shift/Alt), además de `Seleccionar todo`, `Invertir selección`, `Deseleccionar` y `Eliminar píxeles seleccionados`.
- **Inspección Visual, Drag & Drop y Acabado**: Zoom fluido hasta `30x`, paneo, cortina interactiva de comparación Antes/Después, recorte automático (`Autocrop` aplicado sobre la envolvente unión de la secuencia), visualización sobre tablero de ajedrez, aplicación de fondos sólidos y carga por arrastre de archivos o imágenes en memoria (`mime.hasImage()`).

### 5. Secuencias Animadas y Video

- **Animaciones Raster**: Lectura y exportación frame por frame con soporte de canal alfa nativo para **GIF** (cuantización en paralelo por `ThreadPoolExecutor` preservando paleta de 255 colores + índice transparente), **WebP** y **APNG**.
- **Soporte de Video**: Importación de archivos `.mp4`, `.mov`, `.webm`, `.mkv`, `.avi`, `.m4v`, `.wmv`, `.flv`, `.mpeg`, `.mpg`, `.ogv`, `.3gp` y `.ts` con submuestreo uniforme hasta un máximo de `600` fotogramas.
- **Estabilidad Temporal**: Filtro condicional de suavizado temporal (`temporal_smoothing = 0.22`) entre fotogramas consecutivos para mitigar parpadeo de máscaras cuando la variación media de alpha es moderada (`< 45.0`).
- **Exportación Asíncrona**: `SaveSequenceWorker` ejecuta en segundo plano la exportación de **PNGs transparentes en `.zip`**, video con canal alfa nativo mediante FFmpeg CLI (`.webm` VP9 `yuva420p`, `.mkv` FFV1 `bgra`, `.mov` PNG `rgba`) o video directo vía `cv2.VideoWriter` sobre el fondo sólido elegido o Verde Chroma `#00FF00`.

### 6. Procesamiento por Lotes, Cancelación y Portapapeles RGBA

- **Lotes Asíncronos con Reporte Detallado**: `BatchWorkerThread` procesa múltiples archivos en cola con barra de progreso global, acumula los fallos por archivo y emite un resumen detallado de archivos no procesados al finalizar.
- **Cancelación Cooperativa**: El botón Cancelar en la barra de estado propaga `requestInterruption()` a los hilos activos y `cancel_check` a descargas de modelos en curso para detener operaciones de forma segura sin bloquear la interfaz.
- **Portapapeles Completo**: `Ctrl+C` copia el lienzo conservando el canal alfa en formatos compatibles; `Ctrl+V` admite datos de imagen del sistema (`QImage`), flujos PNG binarios con transparencia y URLs de archivos locales.

### 7. Interfaz Modular, Temas e Internacionalización

- **Componentes Independientes**: Ventana principal sin marco (`Frameless`) con `TopBar`, barra de reproducción `SequenceBar`, barra de estado `StatusBar` (con indicador interactivo GPU/CPU, porcentaje determinista `0–100 %` y control de cancelación), ayuda de atajos en `shortcuts_setup.py` y panel lateral `ToolSidebar` compuesto por cuatro subpaneles dedicados (`BackgroundPanel`, `RestorePanel`, `BrushPanel`, `BatchPanel`).
- **Temas e Iconografía Vectorial**: Hojas de estilos QSS (`style.qss` oscuro y `style_light.qss` claro) sincronizadas mediante `apply_theme(is_dark)` en todos los paneles y barras, con iconos vectoriales SVG (Lucide) cacheados en `icons.py`.
- **Multi-idioma en Vivo y Persistencia Nativa**: Diccionarios JSON externalizados en `app/locales/` (`es`, `en`, `ru`, `zh`) con contrato uniforme `retranslate_ui()` en los componentes visuales y preferencia de idioma persistida en `QSettings` a través de `app/i18n.py`.

---

## Compatibilidad de Hardware y Aceleradores

TimoitaseBG evalúa las capacidades del equipo al iniciar y configura la cadena de proveedores de ejecución de ONNX Runtime en orden de prioridad desde `app/core/hardware.py`:

| Plataforma / Fabricante | Proveedor ONNX (`Provider`) | Notas de Ejecución |
| :--- | :--- | :--- |
| NVIDIA (Windows / Linux) | `CUDAExecutionProvider` | Aceleración CUDA directa con respaldo en CPU. |
| AMD Radeon (Linux) | `ROCMExecutionProvider` | Aceleración ROCm en sistemas Linux compatibles. |
| Qualcomm Snapdragon (Windows ARM64) | `QNNExecutionProvider` | Aceleración por NPU/HTP con optimización ARM de hilos y spinning. |
| Apple Silicon / Intel Mac (macOS) | `CoreMLExecutionProvider` | Aceleración nativa mediante CoreML y Metal/ANE. |
| AMD / Intel / NVIDIA (Windows DX12) | `DmlExecutionProvider` | DirectML con detección rápida de GPU discreta vs. integrada vía Registro de Windows. |
| Cualquier x86_64 / ARM64 | `CPUExecutionProvider` | Ejecución universal en CPU con límite óptimo de hilos intra-op (`<= 4`). |

Los usuarios pueden conmutar entre GPU/NPU y CPU desde la interfaz; al desactivar la aceleración GPU, `AICoordinator` descarga las sesiones activas de MobileSAM y YOLO11 y cierra de forma no bloqueante el subproceso GPU para liberar VRAM inmediatamente.

---

## Arquitectura y Funcionamiento Interno

### Estructura del Proyecto

```text
TimoitaseBG/
├── app/
│   ├── main.py                      # Arranque QApplication, spawn POSIX y carga de style*.qss
│   ├── config.py                    # Rutas (portable/OS), SSL CA, APP_VERSION y constantes globales
│   ├── i18n.py                      # Cargador ligero JSON con persistencia en QSettings
│   ├── locales/
│   │   ├── es.json                  # Traducciones en español
│   │   ├── en.json                  # Traducciones en inglés
│   │   ├── ru.json                  # Traducciones en ruso
│   │   └── zh.json                  # Traducciones en chino
│   ├── controllers/
│   │   ├── document_controller.py   # Estado de documento/secuencia, undo/redo, autocrop e I/O
│   │   └── ai_coordinator.py        # Orquestación de workers IA, cancelación y ciclo GPU/VRAM
│   ├── core/
│   │   ├── hardware.py              # Enumeración de GPU (Registro/sysfs/sysctl) y providers ONNX
│   │   ├── postprocess.py           # Pre/postproceso puro: letterbox, guided filter y protección ropa
│   │   ├── bg_remover.py            # Sesiones ONNX y despacho BiRefNet, InSPyReNet, IS-Net, U2-Net
│   │   ├── restorer.py              # Superresolución Real-ESRGAN con tiling adaptativo y alpha
│   │   ├── sam_manager.py           # Codificador y decodificador MobileSAM (soporta *_int8.onnx)
│   │   ├── yolo_detector.py         # Detección multiobjeto YOLO11-seg y refine_yolo_subject_mask
│   │   ├── selection.py             # Varita mágica RGBA, floodFill y operaciones booleanas
│   │   ├── image_state.py           # ImageState (historial acotado) y SequenceSession
│   │   ├── image_utils.py           # Puente stride-safe Qt/NumPy, autocrop, checkerboard y alfa
│   │   ├── gif_processor.py         # Extracción de fotogramas y codificación GIF/WebP/APNG
│   │   ├── video_processor.py       # Submuestreo OpenCV, suavizado temporal y export FFmpeg/OpenCV
│   │   ├── gpu_manager.py           # Orquestador singleton GPUWorkerManager, watchdog y kill tree
│   │   ├── gpu_worker.py            # Subproceso PersistentGPUWorker aislado por IPC
│   │   ├── model_fetcher.py         # Descarga HTTP Range 206, SHA256, SSL y load_rembg_session
│   │   └── workers.py               # QThread de baja prioridad, cancelación cooperativa y secuencias
│   └── ui/
│       ├── main_window.py           # Shell de ventana principal y coordinación de vistas
│       ├── canvas_widget.py         # Cámara, zoom/pan, cortina Antes/Después y composición
│       ├── canvas/
│       │   ├── brush_engine.py      # Motor de pincel restaurador/borrador con _scratch_buf
│       │   └── selection_tool.py    # Superposición de selección, marching ants y varita mágica
│       ├── toolbar.py               # Contenedor ToolSidebar con señales unificadas de los paneles
│       ├── panels/
│       │   ├── bg_panel.py          # Controles de eliminación de fondo, YOLO y MobileSAM
│       │   ├── restore_panel.py     # Controles de superresolución y tiling
│       │   ├── brush_panel.py       # Controles de pinceles, borrador, varita y recortes
│       │   └── batch_panel.py       # Cola de procesamiento por lotes
│       ├── components/
│       │   ├── top_bar.py           # Barra de título frameless, tema, idioma y diálogos pop-up
│       │   ├── sequence_bar.py      # Controles de reproducción y navegación de fotogramas
│       │   ├── status_bar.py        # Estado, progreso, botón Cancelar y badge GPU/CPU
│       │   └── shortcuts_setup.py   # Registro centralizado de QShortcut
│       ├── widgets/
│       │   └── toggles.py           # ToggleSwitch, GpuToggle y AutoResizingStackedWidget
│       ├── drop_zone.py             # Zona inicial con soporte de rutas y mime.hasImage()
│       ├── shortcuts_dialog.py      # Modal de ayuda con tabla de atajos
│       ├── icons.py                 # Carga y caché en memoria de iconos SVG Lucide
│       ├── style.qss                # Hoja de estilos oscura
│       ├── style_light.qss          # Hoja de estilos clara
│       └── assets/                  # Recursos gráficos vectoriales
├── scripts/
│   ├── quantize_models.py           # Cuantización dinámica a *_int8.onnx con validación ONNX
│   ├── package_release.py           # Empaquetado nativo (.msix multi-idioma, .AppImage e iconos)
│   └── download_releases.py         # Descargador de releases desde GitHub con validación SHA256
├── tests/                           # Suite pytest (pipelines, IPC, watchdog, modularidad y UI)
├── .github/workflows/
│   └── release_build.yml            # CI/CD: compuerta test (compileall + pytest) y build matriz
├── build.spec                       # Receta PyInstaller con APP_VERSION dinámica e iconos
├── iniciar.bat                      # Lanzador de entorno virtual en Windows
└── requirements.txt                 # Dependencias Python del proyecto
```

### Arquitectura Modular por Capas

El sistema separa de forma estricta la presentación gráfica, el control de flujo de documento/IA, los algoritmos puros de visión por computadora y la ejecución aislada de modelos en GPU:

```text
+---------------------------------------------------------------------------------+
| Capa de Presentación (app/ui/)                                                  |
|   MainWindow (shell)                                                            |
|     +-- TopBar / SequenceBar / StatusBar (con Cancelar y badge GPU/CPU)         |
|     +-- ToolSidebar -> BackgroundPanel | RestorePanel | BrushPanel | BatchPanel |
|     +-- CanvasWidget -> BrushEngine + SelectionTool + DropZoneWidget            |
+---------------------------------------+-----------------------------------------+
                                        | Comandos de UI y señales tipadas
                                        v
+---------------------------------------------------------------------------------+
| Capa de Controladores / Coordinadores (app/controllers/)                        |
|   - DocumentController: ImageState, SequenceSession, Undo/Redo, I/O y autocrop  |
|   - AICoordinator: QThread workers, cancelación cooperativa, descarga/VRAM      |
+-------------------+-----------------------------------------+-------------------+
                    | Operación síncrona / pura               | QThread (LowPriority)
                    v                                         v
+---------------------------------------+   +-------------------------------------+
| Capa Core Pura (app/core/)            |   | Workers y Subproceso GPU            |
|   - postprocess.py / image_utils.py   |   |   - workers.py -> GPUWorkerManager  |
|   - selection.py / gif & video proc.  |   |   - gpu_worker.py (IPC spawn)       |
|   - hardware.py / model_fetcher.py    |   |   - bg_remover / restorer / sam /   |
+---------------------------------------+   |     yolo_detector                   |
                                            +-------------------------------------+
```

### Ciclo de Ejecución y Aislamiento de GPU

Para impedir que un error del controlador gráfico (TDR en Windows) o una fuga de VRAM en proveedores nativos cierre o bloquee la interfaz de usuario, las operaciones pesadas de GPU se ejecutan fuera del proceso principal:

```text
+-------------------------------------------------------------------------+
| Proceso Principal (Qt GUI - app/main.py & MainWindow)                   |
|   - CanvasWidget / BrushEngine (recorte activo y repintados QRect)      |
|   - AICoordinator & DocumentController                                  |
+------------------------------------+------------------------------------+
                                     | Señales Qt (progressPercent, result, cancelled, error)
                                     v
+-------------------------------------------------------------------------+
| Hilos QThread de Baja Prioridad (app/core/workers.py)                   |
|   - WorkerThread, SequenceRestoreWorker, SaveSequenceWorker             |
|   - BatchWorkerThread (finishedWithErrors), GifWorkerThread             |
|   - SamEmbeddingWorker, YoloDetectionWorker                             |
+------------------------------------+------------------------------------+
                                     | Llama con RLock y timeouts dinámicos
                                     v
+-------------------------------------------------------------------------+
| GPUWorkerManager (app/core/gpu_manager.py - Singleton + atexit)         |
|   - Supervisa latidos {"status": "progress"} reiniciando el watchdog    |
|   - Termina árbol completo en timeout (taskkill /F /T / os.killpg)      |
|   - Conmuta automáticamente a cpu_fallback_fn si hay proceso huérfano   |
+------------------------------------+------------------------------------+
                                     | Colas multiprocessing (spawn)
                                     v
+-------------------------------------------------------------------------+
| PersistentGPUWorker (app/core/gpu_worker.py - Subproceso Aislado)       |
|   - Sincroniza MODELS_DIR y certificados SSL al arrancar                |
|   - Carga sesiones vía load_rembg_session (incluyendo *_int8.onnx)      |
|   - Cachea sesiones ONNX y libera modelos pesados al cambiar de tarea   |
|   - Cascada interna de respaldo (QNN -> DirectML -> CPU)                |
+-------------------------------------------------------------------------+
```

1. **Prioridad de Interfaz**: Los `QThread` arrancan con `QThread.Priority.LowPriority` para ceder ciclos de CPU al bucle de eventos de Qt, manteniendo fluido el zoom, el paneo y la cancelación.
2. **Watchdog con Latidos de Progreso**: `GPUWorkerManager` calcula un timeout dinámico por tarea (`50 s` base, `+15 s` con protección de ropa, `+40 s` para InSPyReNet o BiRefNet, `60 s` para Real-ESRGAN y `+20 s` adicionales cuando el lado mayor supera `2000 px`). Cada actualización de progreso enviada desde el subproceso reinicia el temporizador del watchdog (`t0`).
3. **Recuperación ante Fallos y Cierre No Bloqueante**: Si el subproceso no responde o muere de forma inesperada, `kill_worker()` destruye todo el árbol de procesos (`taskkill /F /T /PID` en Windows u `os.killpg`/`SIGKILL` en POSIX) y delega transparentemente en la CPU del proceso principal (`cpu_fallback_fn`). Al conmutar a CPU o cerrar la ventana, `shutdown(blocking=False)` libera el subproceso y la VRAM sin congelar la UI.

### Detección de Hardware y Gestión de Memoria

- **Enumeración Rápida en Windows** (`app/core/hardware.py`): `_query_windows_gpus_fast()` consulta directamente la rama Display del Registro de Windows (filtrando adaptadores virtuales) con respaldo WMI, permitiendo identificar si el equipo cuenta con GPU dedicada o integrada (`is_integrated_gpu()`) y seleccionar el `device_id` óptimo para DirectML (`get_best_dml_device_id()`). En macOS consulta `sysctl` y en Linux inspecciona `sysfs`/árbol de dispositivos ARM.
- **Gestión Diferenciada de Caché (Discreta vs. UMA)**: En GPUs integradas con arquitectura de memoria unificada (UMA), `PersistentGPUWorker` libera las sesiones anteriores al alternar entre modelos y fuerza el procesamiento por mosaicos en Real-ESRGAN.
- **Optimización de Sesiones ONNX**: `get_optimized_session_options()` activa `ORT_ENABLE_ALL`, limita `intra_op_num_threads` a un máximo de `4` hilos, fija `inter_op_num_threads = 1` y desactiva `session.intra_op.allow_spinning` en arquitecturas ARM para reducir el consumo energético.

### Pipelines de Visión y Edición de Imagen

#### Eliminación de Fondo (`app/core/bg_remover.py` y `app/core/postprocess.py`)
- **InSPyReNet Swin-B**: Preprocesa sobre tensores de `1024x1024` (normalización ImageNet), escala la salida por interpolación bicúbica y ejecuta `postprocess_inspyrenet()` con eliminación vectorizada de ruido por componentes conectados, recuperación topológica de orificios internos mediante `cv2.floodFill` y anti-aliasing con filtro bilateral.
- **BiRefNet e IS-Net**: Ejecutan pre/postprocesamiento con refinamiento por banda perimetral usando `apply_guided_filter()` (`cv2.ximgproc.guidedFilter`, con respaldo `boxFilter`), cierre morfológico guiado por histéresis Canny y relleno acotado de agujeros.
- **Protección Semántica de Ropa**: Reutiliza `u2net_human_seg` (escalando internamente imágenes mayores a `640 px`) e inyecta la máscara de vestimenta únicamente dentro de su núcleo erosionado (`fuse_clothing_protection_core_only()`), protegiendo prendas claras sin introducir halos externos.

#### Aislamiento por Objeto en Tres Etapas (`app/core/yolo_detector.py` + `app/core/sam_manager.py`)
1. **Detección Global**: `YOLODetector` normaliza la imagen en letterbox de `640x640`, aplica supresión de no máximos (`cv2.dnn.NMSBoxes`) y proyecta los coeficientes de máscara contra los prototipos (`(N, 32) @ (32, 25600)`) recortados al cuadro delimitador de cada objeto.
2. **Codificación y Decodificación MobileSAM**: `SAMManager` calcula una sola vez el embedding (`1024x1024` preservando relación de aspecto) mediante `mobile_sam_image_encoder.onnx` y decodifica el punto interior (`cv2.distanceTransform`) + bounding box en `sam_mask_decoder_single.onnx`.
3. **Refinamiento Puro de Bordes**: `refine_yolo_subject_mask()` (alojada fuera de la UI, en `app/core/yolo_detector.py`) fusiona MobileSAM cuando el IoU supera `0.40` y aplica `apply_guided_filter()` sobre la ROI expandida alrededor del sujeto.

#### Superresolución Real-ESRGAN (`app/core/restorer.py`)
- Separa RGB y alpha (`separate_alpha()`), procesa las teselas escribiendo en un único búfer HWC `uint8` preasignado para evitar picos de memoria en fusiones, y reescala el canal alfa de forma independiente una sola vez con Lanczos. Con `use_tiling=True` ejecuta incluso imágenes mayores a `1920x1080` en GPU de forma segura; únicamente desactiva GPU hacia CPU cuando la resolución excede `MAX_GPU_PIXELS` sin tiling habilitado.

#### Motor de Estado, Historial y Lienzo (`image_state.py`, `document_controller.py`, `canvas/`)
- **Historial Acotado**: `ImageState` mantiene una referencia base (`_original_image`) y una pila undo/redo de hasta `MAX_HISTORY_STATES = 25` tuplas `(current_image, brush_reference_image)`, restaurando la imagen de referencia exacta para que el pincel `Pintar` sea coherente tras aplicar escalados o cambios globales.
- **Comprobación Rápida de Transparencia**: `has_transparency()` inspecciona `img.getextrema()[3][0] < 255` en `O(1)` sin recorrer matrices NumPy completas.
- **Secuencias**: `SequenceSession` encapsula un `ImageState` independiente por fotograma; `DocumentController` propaga sobre todos los fotogramas operaciones de autocrop (`apply_autocrop()` con bounding box unión), aislamiento por sujeto (`apply_isolate_subject_sequence()`) y restauración.
- **Puente Seguro Qt / NumPy y Dibujo Parcial**: `qimage_to_numpy()` y `numpy_to_qimage()` manejan explícitamente `bytesPerLine`. `BrushEngine` reutiliza un búfer temporal (`_scratch_buf`) con composición restringida al rectángulo sucio del trazo y recorte opcional por selección, mientras `SelectionTool` mantiene la superposición y el contorno animado a `25 FPS`.

### Modelos ONNX, Cuantización INT8, Reanudación HTTP e Integridad

- **Resolución de Directorio de Modelos (`resolve_models_dir()`)**:
  - En binarios empaquetados (`sys.frozen`), detecta automáticamente una carpeta `./models` junto al ejecutable si ya contiene archivos `.onnx` o tiene permisos de escritura (modo portable).
  - En instalaciones estándar o de solo lectura junto al binario, utiliza la ruta nativa de datos de usuario: `%LOCALAPPDATA%\TimoitaseBG\models` (Windows), `~/Library/Application Support/TimoitaseBG/models` (macOS) o `$XDG_DATA_HOME/TimoitaseBG/models` (Linux).
  - En ejecución desde código fuente, emplea `app/models`.
- **Resolución Unificada y Cuantización `INT8`**: `resolve_local_model()` en `app/core/model_fetcher.py` prioriza automáticamente variantes locales `<nombre>_int8.onnx` válidas (`> 1 MB`) para todos los motores (`bg_remover.py`, `gpu_worker.py`, `restorer.py`, `yolo_detector.py` y `sam_manager.py`). `load_rembg_session()` carga pesos locales (incluyendo `_int8.onnx`) sin disparar descargas redundantes en `pooch`. `scripts/quantize_models.py` convierte modelos a `*_int8.onnx` y valida la sesión resultante en `onnxruntime.InferenceSession`.
- **Reanudación HTTP `Range`, `SHA256` y Cancelación**: `ensure_model_file()` escribe sobre archivos `.tmp` parciales reutilizando descargas interrumpidas mediante cabeceras HTTP `Range` (`206 Partial Content`), soporta interrupción cooperativa vía `cancel_check` y valida el digest cuando está configurado en `MODEL_SHA256` (`verify_sha256()`).
- **Resolución SSL Multiplataforma**: `configure_ssl_certificates()` localiza automáticamente el almacén CA entre `certifi.where()` y rutas estándar del sistema en Linux y macOS. Ante un fallo de verificación SSL del entorno, `ensure_model_file()` registra una advertencia explícita y expone el parámetro `allow_insecure_fallback`.

---

## Atajos de Teclado y Ratón

El registro centralizado (`app/ui/components/shortcuts_setup.py`), el modal integrado (`F1` o `Ctrl+/`) y el lienzo soportan los siguientes controles:

| Atajo / Acción | Función |
| :--- | :--- |
| `F1` / `Ctrl + /` | Mostrar ventana de ayuda y atajos de teclado |
| `Ctrl + O` | Abrir imagen, GIF, WebP/APNG animado o video |
| `Ctrl + S` | Guardar resultado actual |
| `Ctrl + C` | Copiar imagen al portapapeles conservando canal alfa RGBA |
| `Ctrl + V` | Pegar imagen, flujo PNG o ruta local desde el portapapeles |
| `Ctrl + Z` | Deshacer último cambio (hasta 25 estados) |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Rehacer cambio |
| `Ctrl + 0` | Ajustar la imagen al tamaño de la ventana |
| `Ctrl + 1` | Restablecer zoom real al 100 % (1:1) |
| `Supr` / `Backspace` | Borrar píxeles dentro de la selección activa |
| `Ctrl + D` / `Esc` | Deseleccionar o cancelar selección actual |
| `Ctrl + A` | Seleccionar todo el lienzo |
| `Ctrl + Shift + I` | Invertir la selección activa |
| `Rueda del ratón` | Zoom progresivo centrado en el cursor (hasta 30x) |
| `Clic central` o `Espacio + Arrastrar` | Desplazar (paneo) el lienzo ampliado |
| `Espacio` (en animaciones/video) | Reproducir o pausar secuencia de fotogramas |
| `Flecha Izquierda` / `Flecha Derecha` | Retroceder o avanzar un fotograma |
| `Inicio` / `Fin` | Ir al primer o último fotograma de la secuencia |

---

## Requisitos del Sistema

- **Sistema Operativo**: Windows 10/11 (x64 o ARM64), macOS 12+ (Apple Silicon / Intel) o Linux (x86_64 / aarch64).
- **Python (solo ejecución desde fuente)**: Python 3.11 o superior.
- **Memoria RAM**: 8 GB mínimo (16 GB recomendados para `birefnet-general`, secuencias de video o `Real-ESRGAN x4`).
- **Almacenamiento**: ~500 MB para el entorno base más ~1 GB–2.5 GB adicionales según los modelos ONNX descargados.
- **FFmpeg (opcional)**: Recomendado en `PATH` únicamente si se requiere exportar archivos de video `.webm`, `.mkv` o `.mov` con canal alfa nativo; la importación y exportación estándar de video funciona directamente mediante OpenCV.

---

## Instalación y Ejecución

### Opción 1: Ejecución Directa en Windows (`iniciar.bat`)

Una vez creado el entorno virtual e instaladas las dependencias de `requirements.txt`, puedes iniciar la aplicación con el lanzador `iniciar.bat` (que reutiliza `.venv` e inyecta su `PYTHONPATH` para evitar bloqueos de Device Guard / AppLocker sobre ejecutables en carpetas de usuario):

```cmd
iniciar.bat
```

### Opción 2: Instalación Manual Multiplataforma (Windows, macOS y Linux)

```bash
git clone https://github.com/Eliather/TimoitaseBG.git
cd TimoitaseBG

python -m venv .venv
```

Activa el entorno virtual según tu sistema:

- **Windows (PowerShell)**: `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux**: `source .venv/bin/activate`

Instala las dependencias base:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Configura el paquete de ONNX Runtime acorde a tu acelerador de hardware antes de iniciar si deseas reemplazar `onnxruntime` genérico:

- **Windows (DirectML para AMD / Intel / NVIDIA)**: `pip install onnxruntime-directml`
- **NVIDIA CUDA (Windows / Linux)**: `pip install onnxruntime-gpu`
- **Windows ARM64 (Qualcomm QNN)**: `pip install onnxruntime-qnn`
- **macOS (CoreML) / Linux CPU**: incluido en `onnxruntime` estándar (`requirements.txt`).

Inicia la aplicación:

```bash
python -m app.main
```

### Cuantización Opcional de Modelos a INT8

Si ya has descargado modelos en `MODELS_DIR` y deseas generar variantes `INT8` (`*_int8.onnx`) validadas en ONNX Runtime para reducir uso de disco o mejorar velocidad en CPU:

```bash
python -m scripts.quantize_models
```

---

## Pruebas Automatizadas

La configuración en `pytest.ini` ejecuta la suite sobre `tests/` cubriendo pre/postprocesamiento de BiRefNet e InSPyReNet, procesador de video, rutas de distribución portable/OS, watchdog del subproceso GPU, selección MobileSAM, detección YOLO11-seg, seguridad de hilos, regresión modular y portapapeles:

```bash
pytest tests/ -v
```

En entornos headless o sin acceso al portapapeles de escritorio del sistema operativo (como CI Linux sin servidor X):

```bash
pytest tests/ -k "not copy_to_clipboard" -v
```

---

## Compilación y Empaquetado Multiplataforma

El repositorio incluye receta de PyInstaller (`build.spec`), empaquetador nativo (`scripts/package_release.py`), descargador de lanzamientos (`scripts/download_releases.py`) y flujo automatizado en GitHub Actions (`.github/workflows/release_build.yml`).

- **Compuerta de CI**: El job `test` ejecuta verificación de sintaxis (`python -m compileall -q app scripts tests`) y pruebas automatizadas headless (`pytest tests/ -k "not copy_to_clipboard" -v`) antes de compilar los binarios de distribución en la matriz multiplataforma y generar `SHA256SUMS.txt`.
- **Modo Estándar vs. Modelos Embebidos**: Definir la variable de entorno `TIMOITASE_BUNDLE_MODELS=1` al compilar incluye dentro del artefacto los pesos `.onnx` presentes en `app/models` para crear distribuciones completamente offline. Por defecto empaqueta `app/ui`, `app/locales` y lee `APP_VERSION` e iconos de plataforma de forma automática.
- **Generación de Iconos y Compilación con PyInstaller**:

```bash
python -m scripts.package_release icons
pyinstaller build.spec --clean -y
```

- **Generación de Instaladores Nativos (`scripts/package_release.py`)**:

| Plataforma | Arquitecturas | Formato Generado | Notas |
| :--- | :--- | :--- | :--- |
| Windows | `x64`, `arm64` | Paquete `.msix` / ejecutable PyInstaller | `AppxManifest.xml` con recursos declarados para `es-ES`, `en-US`, `ru-RU` y `zh-CN`. |
| macOS | `arm64` (Apple Silicon), `x64` (Intel) | Imagen de disco `.dmg` | Paquete `.app` con icono nativo `.icns` e infolists. |
| Linux | `x64`, `aarch64` | Ejecutable portable `.AppImage` | Estructura `AppDir` con `.desktop`, `AppRun` y certificado CA resuelto en runtime. |

- **Descarga Verificada de Releases (`scripts/download_releases.py`)**:

```bash
python -m scripts.download_releases
```

Descarga los artefactos publicados en GitHub Releases para la versión definida en `APP_VERSION` (o una etiqueta específica pasada como argumento) en `releases/` y verifica su integridad contra `SHA256SUMS.txt`.

---

## Atribución y Licencias de Terceros

TimoitaseBG integra exclusivamente modelos de pesos abiertos e inferencia local con licencias compatibles con distribución libre:

- **BiRefNet** (`birefnet-general`): *Bilateral Reference for High-Resolution Dichotomous Image Segmentation* — Peng Zheng et al. (MIT License).
- **InSPyReNet** (`inspyrenet`): *Revisiting Image Pyramid Structure for High Resolution Salient Object Detection* — Taehun Kim et al. (MIT License).
- **IS-Net** (`isnet-anime`) y **U2-Net** (`u2net_human_seg`): *Highly Accurate Dichotomous Image Segmentation* / *U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection* — Xuebin Qin et al. (Apache 2.0 License).
- **MobileSAM**: *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications* — Chaoning Zhang et al. (Apache 2.0 License).
- **YOLO11-seg** (`yolo11n-seg`): Ultralytics YOLO11 Segmentation (GNU AGPLv3 License).
- **Real-ESRGAN** (`RealESRGAN_x4plus_anime_6B`): *Training Real-World Blind Super-Resolution with Pure Synthetic Data* — Xintao Wang et al. (BSD 3-Clause License).
- **Lucide Icons**: Colección de iconografía vectorial SVG (MIT License).
- **Ecosistema Base**: PySide6 / Qt6 (LGPLv3), ONNX Runtime (MIT), OpenCV (Apache 2.0), Pillow (HPND) y NumPy (BSD 3-Clause).

```bibtex
@inproceedings{zheng2024birefnet,
  title     = {Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author    = {Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal   = {CAAI Artificial Intelligence Research},
  year      = {2024}
}

@inproceedings{kim2022inspyrenet,
  title     = {Revisiting Image Pyramid Structure for High Resolution Salient Object Detection},
  author    = {Kim, Taehun and Kim, Kunhee and Lee, Joonyeong and Cha, Dongmin and Lee, Jiho and Kim, Daijin},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2022}
}

@inproceedings{qin2022highly,
  title     = {Highly Accurate Dichotomous Image Segmentation},
  author    = {Qin, Xuebin and Dai, Hang and Hu, Xiaobin and Fan, Deng-Ping and Shao, Ling and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}

@article{zhang2023faster,
  title     = {Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author    = {Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal   = {arXiv preprint arXiv:2306.14289},
  year      = {2023}
}

@inproceedings{wang2021realesrgan,
  title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author    = {Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2021}
}
```

El detalle completo de avisos y licencias se encuentra en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Autora y Licencia

**TimoitaseBG v1.2.3** ha sido desarrollado por **Eliather** bajo la licencia **GNU Affero General Public License v3.0 (GNU AGPLv3)**. Consulta el archivo [LICENSE](LICENSE) para conocer las condiciones completas de uso, modificación y redistribución.

Apoyo al desarrollo del proyecto: [https://ko-fi.com/eliather](https://ko-fi.com/eliather)
