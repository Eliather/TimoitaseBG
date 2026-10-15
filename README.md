# TimoitaseBG

<p align="center">
  <a href="https://github.com/Eliather/TimoitaseBG/tags">
    <img src="https://img.shields.io/github/v/tag/Eliather/TimoitaseBG?style=for-the-badge&logo=github&label=Versi%C3%B3n&color=FF4F79&sort=semver" alt="Versión">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/stargazers">
    <img src="https://img.shields.io/github/stars/Eliather/TimoitaseBG?style=for-the-badge&logo=github&label=Estrellas&color=FFD700" alt="Estrellas">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/network/members">
    <img src="https://img.shields.io/github/forks/Eliather/TimoitaseBG?style=for-the-badge&logo=github&label=Forks&color=28A745" alt="Forks">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Eliather/TimoitaseBG?style=for-the-badge&logo=gnu&logoColor=white&label=Licencia&color=326CE5" alt="Licencia GNU AGPLv3">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/commits/main">
    <img src="https://img.shields.io/github/last-commit/Eliather/TimoitaseBG?style=for-the-badge&logo=git&logoColor=white&label=%C3%9Altimo%20Commit&color=6366F1" alt="Último Commit">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PySide6%20%2F%20Qt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/Multiplataforma-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20ARM64-0078D4?style=for-the-badge&logo=linux&logoColor=white" alt="Multiplataforma">
  <img src="https://img.shields.io/badge/Aceleraci%C3%B3n-CUDA%20%7C%20ROCm%20%7C%20CoreML%20%7C%20DirectML%20%7C%20QNN-FF6F00?style=for-the-badge&logo=nvidia&logoColor=white" alt="Aceleración">
  <img src="https://img.shields.io/badge/Privacidad-100%25%20Local%20%26%20Offline-6366F1?style=for-the-badge&logo=shield&logoColor=white" alt="Privacidad">
</p>

<p align="center">
  <b>Estación de trabajo de escritorio local y multiplataforma para eliminación de fondos, segmentación con IA y retoque digital de alta precisión.</b>
</p>

---

## Descripción General

**TimoitaseBG** es una solución de escritorio **100% local, multiplataforma**, moderna y de alto rendimiento desarrollada con **PySide6 / Qt6** y **ONNX Runtime**. Diseñada para ilustradores, artistas de anime y manga, diseñadores gráficos y fotógrafos, integra los modelos de visión por computadora más avanzados (**BiRefNet**, **MobileSAM**, **YOLO11-seg**, **InSPyReNet**, **IS-Net Anime** y **Real-ESRGAN**) con herramientas profesionales de selección manual y enmascarado subpíxel estilo Photoshop.

La aplicación ofrece compatibilidad nativa integral con una amplia gama de arquitecturas y aceleradores de hardware:
- **Windows (x86_64 y ARM64):** NVIDIA CUDA, AMD Radeon / Intel Arc vía DirectML, y procesadores **Qualcomm Snapdragon X Elite / X Plus** con aceleración NPU Hexagon (**QNN**) y GPU Adreno (**DirectML ARM64**).
- **macOS (Apple Silicon e Intel):** Aceleración nativa **CoreML** en GPUs y Apple Neural Engine de chips M1, M2, M3, M4 y M5, así como Macs Intel con gráficos integrados o GPUs dedicadas AMD Radeon Pro.
- **Linux (x86_64 y ARM64 `aarch64`):** NVIDIA CUDA (Desktop y **NVIDIA Jetson Orin / Nano**), **AMD ROCm / MIOpen**, y microcomputadoras/SoCs ARM (**Raspberry Pi 5**, Rockchip) con aceleración vectorial NEON.
- **Protección Automática para Memoria Unificada (UMA / APUs):** Reconocimiento inteligente de memoria compartida en APUs AMD Ryzen, Intel iGPU, Apple Silicon y SoCs ARM para activar procesamiento por bloques (*tiling*) y prevenir saturaciones de RAM.

Todo el procesamiento se realiza de manera totalmente local en el equipo del usuario: sin dependencia de servicios en la nube, sin suscripciones periódicas y garantizando la privacidad absoluta de los datos e imágenes.

---

## Tabla de Contenidos

1. [¿Por qué TimoitaseBG?](#por-qué-timoitasebg)
2. [Compatibilidad de Hardware y Aceleradores](#compatibilidad-de-hardware-y-aceleradores)
3. [Características Principales](#características-principales)
   - [Eliminación de Fondos por IA](#1-eliminación-de-fondos-por-ia)
   - [Varita Mágica Híbrida (Color e IA)](#2-varita-mágica-híbrida-color-e-ia)
   - [Detección y Aislamiento Multiobjeto](#3-detección-y-aislamiento-multiobjeto-yolo11-seg)
   - [Restauración y Super-Resolución](#4-restauración-y-super-resolución-real-esrgan)
   - [Herramientas de Edición Manual](#5-herramientas-de-edición-manual)
   - [Lienzo Interactivo y Comparación](#6-lienzo-interactivo-y-comparación)
   - [Procesamiento por Lotes](#7-procesamiento-por-lotes)
   - [Diseño e Internacionalización](#8-diseño-e-internacionalización)
   - [Paginación y Edición por Fotograma en GIFs](#9-paginación-y-edición-por-fotograma-en-gifs)
4. [Arquitectura y Concurrencia Defensiva](#arquitectura-y-concurrencia-defensiva)
5. [Requisitos del Sistema](#requisitos-del-sistema)
6. [Instalación y Puesta en Marcha](#instalación-y-puesta-en-marcha)
   - [Windows (x86_64 y ARM64 Snapdragon)](#opción-a-windows-x86_64-y-arm64-snapdragon)
   - [macOS (Apple Silicon e Intel)](#opción-b-macos-apple-silicon-e-intel)
   - [Linux (x86_64, NVIDIA Jetson y Raspberry Pi)](#opción-c-linux-x86_64-nvidia-jetson-y-raspberry-pi-arm64)
7. [Pruebas Automatizadas](#pruebas-automatizadas)
8. [Compilación a Ejecutable (.exe)](#compilación-a-ejecutable-exe)
9. [Atribución, Licencias de Modelos y Citas Académicas](#atribución-licencias-de-modelos-y-citas-académicas)
10. [Licencia del Software](#licencia-del-software)

---

## ¿Por qué TimoitaseBG?

- **Privacidad Absoluta:** Procesamiento local sin telemetría, analíticas ni envío de información a servidores externos.
- **Aceleración Multiplataforma Nativa:** Soporte universal para aceleradores CUDA (NVIDIA), ROCm (AMD Linux), CoreML (Apple Silicon), DirectML (Windows x64 / ARM64) y QNN (Qualcomm Hexagon NPU).
- **Aislamiento en Subproceso con Watchdog:** Supervisor continuo contra fallos de controlador gráfico (TDR en Windows, bloqueos CUDA/ROCm en Linux o desbordamientos de memoria) con conmutación instantánea a CPU.
- **Gestión Inteligente de Memoria Unificada (UMA):** Ajuste automático de segmentación por bloques (*tiling*) en APUs AMD Ryzen, Intel Iris/UHD, Apple Silicon y SoCs ARM para evitar sobrecargar la memoria RAM compartida.
- **Optimización para CPUs Heterogéneas (big.LITTLE):** Asignación controlada de hilos (`intra_op_num_threads <= 4`) y supresión de giro activo (*no-spinning*) en procesadores ARM para optimizar consumo energético y prevenir sobrecalentamiento.
- **Fidelidad Cromática del 100%:** Conserva la paleta de colores RGB íntegra sin conversiones destructivas ni alteración en el canal alfa.
- **Flujo de Trabajo No Destructivo:** Historial continuo de deshacer/rehacer con soporte multicanal y portapapeles bidireccional.

---

## Compatibilidad de Hardware y Aceleradores

TimoitaseBG integra una capa de detección de hardware multiplataforma que sondea en tiempo de ejecución los buses del sistema (PCIe, SoC, WMI, sysctl, DirectML, CUDA, ROCm y CoreML), asignando el proveedor óptimo de ONNX Runtime con conmutación en caliente (*hot-swap fallback*) ante fallos:

| Plataforma / OS | Arquitectura | Hardware / Acelerador | Proveedor ONNX Runtime | Manejo de Memoria |
| :--- | :--- | :--- | :--- | :--- |
| **Windows 10 / 11** | `x86_64` (AMD64) | NVIDIA GeForce / RTX | `CUDAExecutionProvider` / `DmlExecutionProvider` | VRAM Dedicada |
| **Windows 10 / 11** | `x86_64` (AMD64) | AMD Radeon RX / Vega | `DmlExecutionProvider` (DirectX 12) | VRAM Dedicada |
| **Windows 10 / 11** | `x86_64` (AMD64) | Intel Arc / Iris Xe | `DmlExecutionProvider` (DirectX 12) | VRAM Dedicada |
| **Windows 10 / 11** | `x86_64` (AMD64) | **APU AMD Ryzen (Radeon Graphics)** | `DmlExecutionProvider` (DirectX 12) | Memoria Compartida (UMA) 🛡️ |
| **Windows 10 / 11** | `x86_64` (AMD64) | **Intel Graphics (UHD / HD)** | `DmlExecutionProvider` (DirectX 12) | Memoria Compartida (UMA) 🛡️ |
| **Windows on ARM** | `ARM64` | **Qualcomm Snapdragon X Elite / Plus** | `QNNExecutionProvider` (NPU Hexagon) | Memoria Unificada (UMA) 🛡️ |
| **Windows on ARM** | `ARM64` | **GPU Qualcomm Adreno** | `DmlExecutionProvider` (DirectML ARM64) | Memoria Unificada (UMA) 🛡️ |
| **macOS 12+** | `arm64` | **Apple Silicon (M1, M2, M3, M4, M5)** | `CoreMLExecutionProvider` (GPU + NPU) | Memoria Unificada (UMA) 🛡️ |
| **macOS 12+** | `x86_64` | Mac Intel con AMD Radeon Pro | `CoreMLExecutionProvider` (GPU) | VRAM Dedicada |
| **macOS 12+** | `x86_64` | Mac Intel con Iris / UHD | `CoreMLExecutionProvider` / CPU | Memoria Compartida (UMA) 🛡️ |
| **Linux (Ubuntu, etc.)** | `x86_64` | NVIDIA GeForce / RTX / Tesla | `CUDAExecutionProvider` | VRAM Dedicada |
| **Linux (Ubuntu, etc.)** | `x86_64` | **AMD Radeon (ROCm / MIOpen)** | `ROCMExecutionProvider` | VRAM Dedicada |
| **Linux ARM64** | `aarch64` | **NVIDIA Jetson (Orin, Xavier, Nano)** | `CUDAExecutionProvider` (JetPack) | Memoria Unificada (UMA) 🛡️ |
| **Linux ARM64** | `aarch64` | **Raspberry Pi 5 / Rockchip RK3588** | `CPUExecutionProvider` (Vectorial NEON) | Memoria Unificada (UMA) 🛡️ |
| **Universal (Fallback)**| Cualquiera | Procesador Intel / AMD / ARM | `CPUExecutionProvider` (Multihilo) | Memoria RAM Sistema |

> 🛡️ **Protección Automática de Memoria Unificada (UMA / APUs):**  
> En arquitecturas de memoria compartida (APUs AMD Ryzen, gráficos Intel integrados, Apple Silicon, Qualcomm Snapdragon y SoCs ARM en Linux), la función `is_integrated_gpu()` retorna `True`, forzando el procesamiento por mosaicos (*tiling 512×512*) en Real-ESRGAN y liberando buffers entre fases para impedir saturaciones de la RAM del sistema.

---

## Características Principales

### 1. Eliminación de Fondos por IA

Pipeline de segmentación de última generación ejecutado sobre **ONNX Runtime** con posprocesamiento morfológico adaptativo:

- **BiRefNet (General High Definition 1024×1024 - MIT):** Arquitectura dicotómica SOTA de máxima precisión para la extracción de detalles complejos como cabello, pelaje, texturas finas y bordes translúcidos.
- **InSPyReNet (Swin-B Plus Ultra):** Red neuronal piramidal para refinamiento subpíxel y corrección cromática en bordes (*despill*).
- **IS-Net Anime:** Modelo especializado en arte digital 2D, manga, cómics y cel-shading.
- **U2-Net Human:** Segmentación anatómica enfocada en el aislamiento integral de personas.
- **Detector Semántico de Ropa:** Filtro inteligente para evitar perforaciones accidentales en vestimentas claras sobre fondos de tonalidad similar.

### 2. Varita Mágica Híbrida (Color e IA)

Motor dual de selección con dos modos de operación complementarios:

#### A. Modo IA (MobileSAM)
- **Selección Semántica Interactiva:** Genera máscaras de alta precisión al hacer clic sobre cualquier objeto con tiempos de respuesta inferiores a **15 ms**.
- **Arquitectura Optimizada:** El codificador de imagen (*Image Encoder*) precalcula la representación en un hilo secundario de fondo, permitiendo decodificaciones instantáneas (*Mask Decoder*).

#### B. Modo Color (Estilo Photoshop)
- **Tolerancia Configurable (0–255):** Ajuste de sensibilidad basado en distancia euclidiana en el espacio cromático RGB.
- **Modos Contiguo y Global:** Selección de áreas continuas conectadas o islas de color distribuidas en toda la imagen.
- **Suavizado (Anti-Aliasing):** Atenuación perimetral para evitar bordes dentados (*aliased edges*).

#### Operaciones Booleanas y Hormigas Marchantes
- **4 Modos de Combinación:** Nueva Selección, Añadir (`Shift`), Restar (`Alt`) e Intersecar (`∩`).
- **Límite Animado (Marching Ants):** Contorno vectorial animado a 25 FPS para delimitar con precisión la selección activa.
- **Atajos Directos:** Borrar selección (`Supr`), Invertir (`Ctrl+Shift+I`), Seleccionar todo (`Ctrl+A`) y Deseleccionar (`Ctrl+D` / `Esc`).

### 3. Detección y Aislamiento Multiobjeto (YOLO11-seg)

- **Segmentación Multiobjeto:** Detección e identificación simultánea de hasta 80 clases de objetos mediante **YOLO11-seg**.
- **Panel Lateral Interactivo:** Presentación de objetos detectados con su respectiva etiqueta, porcentaje de confianza y color identificativo.
- **Acciones Rápidas:** Aislamiento del sujeto con un solo clic o conversión del contorno a selección activa para edición manual.

### 4. Restauración y Super-Resolución (Real-ESRGAN)

- **Modelo RealESRGAN_x4plus_anime_6B:** Reducción de ruido, eliminación de artefactos JPEG y reescalado de ilustraciones con líneas de tinta definidas.
- **Modos de Trabajo:** Mejora de nitidez (1x), aumento 2x y aumento 4x.
- **Procesamiento por Bloques (Tiling 512×512):** Manejo eficiente de memoria en imágenes de alta resolución.
- **Protección de VRAM:** Conmutación automática a CPU si la imagen supera el umbral de memoria seguro (~2.07 Mpx).

### 5. Herramientas de Edición Manual

- **Pincel Restaurador:** Recuperación selectiva de áreas originales de la imagen.
- **Borrador de Precisión:** Eliminación manual sobre capas y máscaras.
- **Parámetros Ajustables:** Tamaño (2–200 px), suavizado (0–100%), opacidad (1–100%) y dureza (0–100%).

### 6. Lienzo Interactivo y Comparación

- **Deslizador Antes/Después:** Comparación interactiva en tiempo real entre la imagen original y el resultado procesado.
- **Ajuste Automático (Auto-Crop):** Recorte de bordes transparentes con un margen configurable de 10 px.
- **Visualización de Fondo:** Alternancia entre fondo transparente, colores sólidos (blanco, negro, gris) o valores personalizados RGB/HEX.
- **Control de Navegación:** Zoom fluido, ajuste a pantalla y desplazamiento libre (*pan*) mediante arrastre.

### 7. Procesamiento por Lotes

- Procesamiento automático de carpetas completas de imágenes sin intervención manual.
- Aplicación uniforme del modelo de IA seleccionado y la protección de prendas.
- Exportación secuencial en formato PNG transparente con barra de progreso en tiempo real.

### 8. Diseño e Internacionalización

- **Iconografía Vectorial SVG (Lucide Icons):** Interfaz adaptativa e independiente de la resolución o escala DPI de pantalla.
- **Temas Visuales:** Modos oscuro y claro implementados mediante Qt Style Sheets (QSS).
- **Soporte Multilingüe (i18n):** Español (`es`), Inglés (`en`), Ruso (`ru`) y Chino Simplificado (`zh`).

### 9. Paginación y Edición por Fotograma en GIFs

- **Navegación Cuadro a Cuadro:** Inspección y control de secuencias animadas mediante barra HUD inferior flotante (`SequenceSession`).
- **Edición Individual Aislada:** Modificación directa de cualquier fotograma con pincel, borrador, varita MobileSAM o YOLO11 sin sobreescribir ni contaminar el resto de la animación.
- **Reproducción Fluida en Lienzo:** Bucle interactivo (*Play / Pausa*) respetando la duración exacta de cada fotograma.
- **Estabilización Temporal Anti-Flicker:** Suavizado dinámico de bordes en inferencia secuencial para eliminar la vibración (*jitter*) entre fotogramas consecutivos.
- **Exportación Limpia de Paleta Indexada:** Conversión a paleta de 255 colores adaptativos con canal alfa de 1-bit sin halos oscuros en los contornos.

---

## Arquitectura y Concurrencia Defensiva

TimoitaseBG implementa una arquitectura defensiva dividida en subprocesos para garantizar la estabilidad continua del entorno en todas las plataformas:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Proceso Principal                             │
│               (Interfaz Gráfica PySide6 / Event Loop Qt)                │
│  - Renderizado nativo a 60 FPS (Hardware acelerado)                     │
│  - CanvasWidget (Hormigas marchantes, zoom, pincel, selección)          │
│  - MainWindow, SequenceSession (GIFs) & ToolSidebar                     │
└────────────────────┬────────────────────────────────┬───────────────────┘
                     │                                │
              (IPC Queue / Mutex)             (QThreads Controlados)
                     ▼                                ▼
┌───────────────────────────────────────┐  ┌──────────────────────────────┐
│       GPU Worker Aislado (IPC)        │  │     Workers de Fondo UI      │
│         (Subproceso dedicado)         │  │ - SamEmbeddingWorker         │
│  - Backends: CUDA | ROCm | CoreML |   │  │ - YoloDetectionWorker        │
│    DirectML (x64/ARM64) | QNN (NPU)   │  │ - BatchWorkerThread          │
│  - BiRefNet General 1024×1024 (MIT)  │  │ Prioridad: LowPriority       │
│  - InSPyReNet Swin-B                  │  │ CPU Thread Cap: <= 4         │
│  - Real-ESRGAN x4plus (Tiling 512)    │  │ Afinidad ARM: allow_spin=0   │
│  - Watchdog Supervisor & Auto-Kill    │  │ Re-entrancia bloqueada       │
│  - Fallback Dinámico a CPU            │  │                              │
└───────────────────────────────────────┘  └──────────────────────────────┘
```

1. **Aislamiento del Subproceso GPU (`PersistentGPUWorker`):** La inferencia en GPU/NPU se ejecuta en un proceso de sistema operativo independiente (`multiprocessing.Process`). Ante un fallo del controlador gráfico (TDR en Windows DirectX/DirectML, kernel panics en Linux CUDA/ROCm, o excepciones de operadores no soportados en drivers ARM), el watchdog termina el subproceso colgado y conmuta de forma transparente a CPU sin congelar la interfaz ni perder el trabajo en curso.
2. **Priorización y Conmutación en Cascada:**
   `CUDA` ➔ `ROCm` ➔ `QNN` ➔ `CoreML` ➔ `DirectML` ➔ `CPU (NEON/AVX2)`.
   Si un modelo falla al compilar operadores en NPU o GPU Adreno, conmuta automáticamente a DirectML o CPU en el mismo worker sin colapsar el sistema.
3. **Control Térmico y Energético en CPUs Heterogéneas (big.LITTLE):** En procesadores ARM (Snapdragon, Apple Silicon, Raspberry Pi), ONNX Runtime restringe los hilos (`intra_op_num_threads <= 4`) y desactiva el giro activo (`session.intra_op.allow_spinning = 0`) para evitar que hilos intensivos caigan en núcleos lentos de eficiencia, previniendo drenaje de batería y estrangulamiento térmico (*thermal throttling*).
4. **Sincronización:** Uso de cerrojos reentrantes (`RLock`) para gestionar las peticiones a la GPU de manera estrictamente secuencial y segura.
5. **Finalización Controlada:** Manejo de eventos de cierre (`closeEvent`) para asegurar la detención limpia y liberación de VRAM antes de salir.

---

## Requisitos del Sistema

- **Sistemas Operativos Soportados:**
  - **Windows:** Windows 10 y Windows 11 (arquitecturas `x86_64` y `ARM64`).
  - **macOS:** macOS 12 Monterey, macOS 13 Ventura, macOS 14 Sonoma y macOS 15 Sequoia (Nativo en `Apple Silicon` M1–M5 e Intel Mac `x86_64`).
  - **Linux:** Distribuciones modernas (Ubuntu 22.04+, Debian 12+, Arch Linux, Fedora) en `x86_64` y `aarch64` / `armv8` (NVIDIA Jetson, Raspberry Pi 5).
- **Procesador (CPU):** Intel Core i3 / AMD Ryzen 3 / Apple Silicon / Snapdragon X / ARMv8 o superior (4 núcleos recomendados).
- **Memoria RAM:** 8 GB mínimo (16 GB recomendados para imágenes 4K).
- **Aceleradores Soportados (Opcional, con fallback automático a CPU):**
  - **NVIDIA:** GeForce GTX/RTX, Quadro, Tesla, o NVIDIA Jetson (CUDA 11.8+ / 12+).
  - **AMD:** Radeon RX 400+, Vega, RDNA/RDNA2/RDNA3 (DirectML en Windows o ROCm en Linux).
  - **Apple:** GPU de 8 a 40 núcleos y Neural Engine (M1–M5) mediante CoreML nativo.
  - **Qualcomm:** Snapdragon X Elite / Plus con NPU Hexagon (QNN) y GPU Adreno (DirectML ARM64).
  - **Intel:** Intel Arc Alchemist / Battlemage o gráficos integrados Iris Xe / UHD.
  - **APUs y Gráficos Integrados:** Compatibilidad nativa total con auto-tiling para proteger la memoria compartida.
- **Almacenamiento:** ~1.5 GB de espacio disponible para dependencias y modelos ONNX locales.

---

## Instalación y Puesta en Marcha

### Opción A: Windows (x86_64 y ARM64 Snapdragon)

1. **Clonar el repositorio:**
   ```bash
   git clone git@github.com:Eliather/TimoitaseBG.git
   cd TimoitaseBG
   ```

2. **Inicio Rápido (con script automático):**
   ```bat
   iniciar.bat
   ```

3. **O Instalación Manual:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   # Opcional para Windows ARM64 (Snapdragon X Elite / Plus con NPU Hexagon):
   # pip install onnxruntime-qnn
   python app/main.py
   ```

### Opción B: macOS (Apple Silicon e Intel)

En macOS, `onnxruntime` incluye soporte nativo para **CoreML** (`CoreMLExecutionProvider`) sin necesidad de paquetes externos:

1. **Clonar e ingresar al repositorio:**
   ```bash
   git clone git@github.com:Eliather/TimoitaseBG.git
   cd TimoitaseBG
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias y ejecutar:**
   ```bash
   pip install -r requirements.txt
   python app/main.py
   ```

### Opción C: Linux (x86_64, NVIDIA Jetson y Raspberry Pi ARM64)

1. **Instalar dependencias del sistema (librerías multimedia para Qt y OpenCV):**
   *En Ubuntu / Debian / Raspberry Pi OS:*
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 libegl1
   ```

2. **Clonar y preparar entorno virtual:**
   ```bash
   git clone git@github.com:Eliather/TimoitaseBG.git
   cd TimoitaseBG
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Aceleración Específica por Hardware (Opcional):**
   - **Para GPU NVIDIA en Linux (x86_64 o NVIDIA Jetson Orin/Nano aarch64):**
     ```bash
     pip install onnxruntime-gpu
     ```
   - **Para GPU AMD ROCm en Linux:**
     ```bash
     pip install onnxruntime-rocm --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-rocm-release/pypi/simple/
     ```
   - **Para Raspberry Pi 5 y SBCs ARM64:** El paquete oficial `onnxruntime` instala wheels compiladas para `manylinux_2_27_aarch64` con aceleración vectorial NEON integrada de fábrica.

4. **Ejecutar:**
   ```bash
   python app/main.py
   ```

---

## Pruebas Automatizadas

El proyecto cuenta con una suite integral de pruebas unitarias y de integración, incluyendo validaciones simuladas multiplataforma (Windows ARM64, macOS Apple Silicon, Linux ROCm/ARM SoCs y APUs integradas):

```bash
# Ejecutar la suite de pruebas del Watchdog Aislado y Detección Multiplataforma:
pytest tests/test_gpu_worker_watchdog.py -v

# Ejecutar pruebas de Estabilidad de Hilos, Concurrencia y Modelos de Segmentación:
pytest tests/test_thread_and_resource_safety.py tests/test_sam_selection.py tests/test_yolo_detection.py tests/test_birefnet.py -v
```

---

## Compilación a Ejecutable (.exe)

Para generar la distribución binaria independiente para Windows mediante **PyInstaller**:

```bash
uv pip install pyinstaller
pyinstaller build.spec
```

El binario resultante se ubicará en: `dist/TimoitaseBG/TimoitaseBG.exe`.

---

## Atribución, Licencias de Modelos y Citas Académicas

TimoitaseBG hace uso de modelos y librerías de código abierto de la comunidad científica e industrial. Ningún modelo incluido impone restricciones no comerciales (`CC BY-NC`):

### Tabla de Atribución de Componentes

| Componente / Modelo | Autores / Organización | Licencia Original | Función en TimoitaseBG | Referencia |
| :--- | :--- | :--- | :--- | :--- |
| **BiRefNet** | Zheng et al. | **MIT License** | Eliminación dicotómica de fondos en alta resolución (1024×1024) | [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| **InSPyReNet** | Kim et al. (POSTECH) | **MIT License** | Eliminación de fondo con pirámide inversa y anti-aliasing | [GitHub](https://github.com/plemeri/InSPyReNet) |
| **IS-Net Anime** | SkyTNT / Qin et al. | **Apache 2.0** | Segmentación especializada en ilustración 2D, manga y cel-shading | [Hugging Face](https://huggingface.co/skytnt/anime-seg) |
| **MobileSAM** | Zhang et al. (Kyung Hee Univ.) | **Apache 2.0** | Varita mágica por IA interactiva (<15 ms) | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |
| **Segment Anything (SAM)** | Meta AI Research | **Apache 2.0** | Modelo base y decodificador de máscaras | [GitHub](https://github.com/facebookresearch/segment-anything) |
| **YOLO11-seg** | Ultralytics Inc. | **GNU AGPLv3** | Detección y segmentación multiobjeto (hasta 80 clases) | [GitHub](https://github.com/ultralytics/ultralytics) |
| **Real-ESRGAN** | Wang et al. (Tencent PCG) | **BSD 3-Clause** | Super-resolución 2x/4x y restauración de nitidez | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| **Detector de Ropa** | Levin Dabhi / Qin et al. | **MIT / Apache 2.0** | Filtro protector de prendas claras (`u2net_cloth_seg`) | [GitHub](https://github.com/levindabhi/cloth-segmentation) |
| **U²-Net** | Qin et al. (Univ. of Alberta) | **Apache 2.0** | Segmentación semántica de personas | [GitHub](https://github.com/xuebinqin/U-2-Net) |
| **ViTMatte** | Yao et al. (HUST & Kuaishou) | **Apache 2.0** | Refinamiento perimetral con Vision Transformers | [GitHub](https://github.com/hustvl/ViTMatte) |
| **Lucide Icons** | Lucide Project | **ISC License** | Iconografía vectorial SVG | [Lucide.dev](https://lucide.dev) |
| **PySide6 / Qt** | The Qt Company | **LGPLv3** | Interfaz gráfica nativa multiplataforma | [Qt.io](https://www.qt.io) |

Para detalles exhaustivos, fuentes de descarga y hashes de verificación, consulte el archivo [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Citas Académicas (BibTeX)

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Fan, Deng-Ping and Tian, Xiao-Bin and others},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}

@article{zhang2023mobilesam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, SeungKyu and Cho, Choongsang},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}

@software{yolo11_ultralytics,
  author={Glenn Jocher and Jing Qiu},
  title={Ultralytics YOLO11},
  version={11.0.0},
  year={2024}
}

@InProceedings{wang2021realesrgan,
  author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  booktitle={International Conference on Computer Vision Workshops (ICCVW)},
  year={2021}
}

@inproceedings{kim2022inspyrenet,
  title={InSPyReNet: Inverse Saliency Pyramid Recurrent Network for Salient Object Detection},
  author={Kim, Taehun and Kim, Kunhee and Lee, Joonyeong and Cha, Dongmin and Lee, Jiho and Kim, Daijin},
  booktitle={ACM International Conference on Multimedia},
  year={2022}
}

@InProceedings{qin2022dis,
  title={Highly Accurate Dichotomous Image Segmentation},
  author={Qin, Xuebin and Dai, Hang and Tian, Xiaobin and Fan, Deng-Ping and Shao, Ling and Van Gool, Luc},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

## Licencia del Software

Este programa es software libre y de código abierto distribuido bajo los términos de la **GNU Affero General Public License Version 3 (GNU AGPLv3)**. Para más detalles, consulte el archivo [LICENSE](LICENSE).

```text
TimoitaseBG - Copyright (C) 2026 Eliather

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.
```

### Relación con Licencias de Componentes y Modelos de Terceros

- **Licencia del Código Fuente:** El código de TimoitaseBG está licenciado bajo **GNU AGPLv3**. Cualquier persona o entidad es libre de usar, estudiar, modificar y redistribuir la aplicación (incluso para fines comerciales), bajo la condición de que las versiones modificadas o desplegadas en red compartan su código fuente bajo esta misma licencia.
- **Independencia de Licencias de Modelos:** Cada modelo de inteligencia artificial de terceros conserva su propia licencia otorgada por sus respectivos autores (MIT, Apache 2.0, BSD 3-Clause, AGPL-3.0). La licencia AGPLv3 de TimoitaseBG cubre el código de la aplicación y la integración, no sustituye ni modifica las licencias originales de los pesos pre-entrenados.
- **100% Libre de Restricciones Comerciales (Sin CC BY-NC):** Todos los modelos incorporados o descargados bajo demanda permiten el uso comercial y la libre redistribución. No existen obstáculos de licencias no comerciales para recibir donaciones, comercializar soluciones derivadas o utilizar la herramienta en flujos profesionales.
- Para consultar los avisos completos, procedencia de datasets y términos individuales, consulte [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
