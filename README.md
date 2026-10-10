# TimoitaseBG

<p align="center">
  <a href="https://github.com/Eliather/TimoitaseBG/releases">
    <img src="https://img.shields.io/badge/Versi%C3%B3n-v1.0.0-FF4F79?style=for-the-badge&logo=github" alt="Versión">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/releases">
    <img src="https://img.shields.io/badge/Descargas-100%2B-28A745?style=for-the-badge&logo=github" alt="Descargas">
  </a>
  <a href="https://github.com/Eliather/TimoitaseBG/stargazers">
    <img src="https://img.shields.io/badge/Estrellas-50%2B-FFD700?style=for-the-badge&logo=github" alt="Estrellas">
  </a>
  <a href="https://www.gnu.org/licenses/agpl-3.0.html">
    <img src="https://img.shields.io/badge/Licencia-GNU%20AGPLv3-326CE5?style=for-the-badge&logo=gnu&logoColor=white" alt="Licencia GNU AGPLv3">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PySide6%20%2F%20Qt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/Inferencia-ONNX%20Runtime%20%7C%20DirectML-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="DirectML">
  <img src="https://img.shields.io/badge/Privacidad-100%25%20Local%20%26%20Offline-6366F1?style=for-the-badge" alt="Privacidad">
</p>

<p align="center">
  <b>Estación de trabajo de escritorio local para eliminación de fondos, segmentación con IA y retoque digital de alta precisión.</b>
</p>

---

## 📌 Descripción General

**TimoitaseBG** es una solución de escritorio **100% local**, moderna y de alto rendimiento desarrollada con **PySide6 / Qt6**. Diseñada para ilustradores, artistas de anime y manga, diseñadores gráficos y fotógrafos, integra los modelos de visión por computadora más avanzados (**RMBG-2.0**, **MobileSAM**, **YOLO11-seg**, **InSPyReNet** y **Real-ESRGAN**) con herramientas profesionales de selección manual y enmascarado subpíxel estilo Photoshop.

Todo el procesamiento se realiza de manera totalmente local en el equipo del usuario: sin dependencia de servicios en la nube, sin suscripciones periódicas y garantizando la privacidad absoluta de los datos e imágenes.

---

## 📑 Tabla de Contenidos

1. [¿Por qué TimoitaseBG?](#-por-qué-timoitasebg)
2. [Características Principales](#-características-principales)
   - [Eliminación de Fondos por IA](#1-eliminación-de-fondos-por-ia)
   - [Varita Mágica Híbrida (Color e IA)](#2-varita-mágica-híbrida-color-e-ia)
   - [Detección y Aislamiento Multiobjeto](#3-detección-y-aislamiento-multiobjeto-yolo11-seg)
   - [Restauración y Super-Resolución](#4-restauración-y-super-resolución-real-esrgan)
   - [Herramientas de Edición Manual](#5-herramientas-de-edición-manual)
   - [Lienzo Interactivo y Comparación](#6-lienzo-interactivo-y-comparación)
   - [Procesamiento por Lotes](#7-procesamiento-por-lotes)
   - [Diseño e Internacionalización](#8-diseño-e-internacionalización)
3. [Arquitectura y Concurrencia](#-arquitectura-y-concurrencia)
4. [Requisitos del Sistema](#-requisitos-del-sistema)
5. [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
6. [Pruebas Automatizadas](#-pruebas-automatizadas)
7. [Compilación a Ejecutable (.exe)](#-compilación-a-ejecutable-exe)
8. [Atribución, Licencias de Modelos y Citas Académicas](#-atribución-licencias-de-modelos-y-citas-académicas)
9. [Licencia del Software](#-licencia-del-software)

---

## 💡 ¿Por qué TimoitaseBG?

- **🔒 Privacidad Garantizada:** Procesamiento local sin telemetría ni envío de información a servidores externos.
- **⚡ Aceleración por Hardware DirectML:** Inferencia optimizada para GPUs en Windows (NVIDIA, AMD Radeon e Intel Arc) a través de DirectX 12, con soporte y conmutación automática a CPU.
- **🛡️ Gestión Eficiente de Recursos:** Control de concurrencia y subprocesos para mantener la interfaz fluida a 60 FPS sin saturar la CPU.
- **💎 Fidelidad Cromática del 100%:** Mantiene la paleta de colores RGB original sin compresión ni alteración cromática.
- **🎨 Flujo de Trabajo No Destructivo:** Historial de deshacer/rehacer ilimitado con restauración de estado y soporte para portapapeles.

---

## 🚀 Características Principales

### 1. Eliminación de Fondos por IA

Pipeline de segmentación de última generación ejecutado sobre **ONNX Runtime** con posprocesamiento morfológico adaptativo:

- **RMBG-2.0 (BiRefNet SOTA 1024×1024):** Modelo de alta precisión para la extracción de detalles complejos como cabello, pieles, texturas finas y objetos transparentes.
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

---

## ⚙️ Arquitectura y Concurrencia

TimoitaseBG implementa una arquitectura defensiva dividida en subprocesos para garantizar la estabilidad continua del entorno:

```text
┌────────────────────────────────────────────────────────┐
│                   Proceso Principal                    │
│     (Interfaz Gráfica PySide6 / Event Loop Qt)         │
│  - Renderizado a 60 FPS                                │
│  - CanvasWidget (Hormigas marchantes, zoom, pincel)    │
│  - MainWindow & ToolSidebar                            │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
     (IPC Queue / Mutex)             (QThreads Controlados)
            ▼                                ▼
┌────────────────────────┐      ┌──────────────────────────┐
│  GPU Worker Aislado    │      │ Workers de Fondo UI      │
│  (Subproceso dedicado) │      │ - SamEmbeddingWorker     │
│  - RMBG-2.0 / BiRefNet │      │ - YoloDetectionWorker    │
│  - InSPyReNet          │      │ - BatchWorkerThread      │
│  - Real-ESRGAN         │      │ Prioridad: LowPriority   │
│  - Watchdog de Timeout │      │ CPU Thread Cap: <= 4     │
│  - Fallback a CPU      │      │ Re-entrancia bloqueada   │
└────────────────────────┘      └──────────────────────────┘
```

1. **Aislamiento del Subproceso GPU (`PersistentGPUWorker`):** La inferencia en GPU se ejecuta en un proceso de sistema operativo independiente (`multiprocessing.Process`). Ante un fallo del controlador gráfico (TDR en DirectX/DirectML), el sistema commuta de forma transparente a CPU sin congelar la interfaz.
2. **Control de Uso de CPU:** Configuración de hilos en ONNX Runtime (`intra_op_num_threads <= 4`) para evitar picos de carga y mantener la fluidez del sistema operativo.
3. **Sincronización:** Uso de cerrojos reentrantes (`RLock`) para gestionar las peticiones a la GPU de manera estrictamente secuencial.
4. **Finalización Controlada:** Manejo de eventos de cierre (`closeEvent`) para asegurar la detención limpia de subprocesos antes del desmontaje.

---

## 💻 Requisitos del Sistema

- **Sistema Operativo:** Windows 10 o Windows 11 (64-bit).
- **Procesador (CPU):** Intel Core i3 / AMD Ryzen 3 o superior (4 núcleos recomendados).
- **Memoria RAM:** 8 GB mínimo (16 GB recomendados para imágenes 4K).
- **Aceleración Gráfica (GPU):**
  - Tarjeta gráfica compatible con **DirectX 12 (DirectML)**: NVIDIA GeForce, AMD Radeon o Intel Arc.
  - Alternativamente, aceleración mediante **NVIDIA CUDA**.
  - *Nota:* Si no hay GPU dedicada presente, la aplicación funciona al 100% en CPU.
- **Almacenamiento:** ~1.5 GB de espacio disponible para ejecutable y modelos ONNX.

---

## 📦 Instalación y Puesta en Marcha

### Opción 1: Inicio Rápido (Windows)

Si el repositorio ya cuenta con el entorno configurado, ejecute el script de inicio:

```bat
iniciar.bat
```

### Opción 2: Instalación Manual

1. **Clonar el repositorio:**
   ```bash
   git clone git@github.com:Eliather/TimoitaseBG.git
   cd TimoitaseBG
   ```

2. **Crear y activar el entorno virtual:**
   *Con `uv`:*
   ```bash
   uv venv .venv --python 3.11
   .venv\Scripts\activate
   ```
   *Con `venv` estándar:*
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   uv pip install -r requirements.txt
   # O alternativamente:
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación:**
   ```bash
   python app/main.py
   ```

---

## 🧪 Pruebas Automatizadas

El proyecto cuenta con una suite integral de **75 pruebas unitarias y de integración**:

```bash
# Ejecutar todas las pruebas
pytest -v

# Ejecutar pruebas específicas de estabilidad y modelos
pytest tests/test_thread_and_resource_safety.py tests/test_sam_selection.py tests/test_yolo_detection.py tests/test_rmbg2.py -v
```

---

## 🛠️ Compilación a Ejecutable (.exe)

Para generar la distribución binaria independiente para Windows mediante **PyInstaller**:

```bash
uv pip install pyinstaller
pyinstaller build.spec
```

El binario resultante se ubicará en: `dist/TimoitaseBG/TimoitaseBG.exe`.

---

## 📚 Atribución, Licencias de Modelos y Citas Académicas

TimoitaseBG hace uso de modelos y librerías de código abierto de la comunidad científica e industrial:

### Tabla de Atribución de Componentes

| Componente / Modelo | Autores / Organización | Licencia Original | Función en TimoitaseBG | Referencia |
| :--- | :--- | :--- | :--- | :--- |
| **RMBG-2.0** | BRIA AI & ZhengPeng7 | CC BY-NC 4.0 | Eliminación de fondo a 1024×1024 | Hugging Face / Paper |
| **BiRefNet** | Zheng et al. | Apache 2.0 | Arquitectura de segmentación dicotómica | GitHub |
| **MobileSAM** | Zhang et al. (Kyung Hee Univ.) | Apache 2.0 | Varita mágica por IA | GitHub |
| **Segment Anything (SAM)** | Meta AI Research | Apache 2.0 | Modelo base de segmentación | GitHub |
| **YOLO11-seg** | Ultralytics Inc. | AGPL-3.0 | Detección y segmentación multiobjeto | GitHub |
| **InSPyReNet** | Kim et al. (POSTECH) | MIT | Eliminación de fondo con pirámide inversa | GitHub |
| **Real-ESRGAN** | Wang et al. (Tencent PCG) | BSD 3-Clause | Super-resolución y restauración | GitHub |
| **IS-Net (DIS5K)** | Qin et al. | Apache 2.0 | Segmentación para ilustración 2D | GitHub |
| **U^2-Net** | Qin et al. (Univ. of Alberta) | Apache 2.0 | Segmentación humana y de prendas | GitHub |
| **ViTMatte** | Yao et al. (HUST & Kuaishou) | Apache 2.0 | Refinamiento de bordes con Transformers | GitHub |
| **Lucide Icons** | Lucide Project | ISC License | Iconografía vectorial SVG | Lucide.dev |
| **PySide6 / Qt** | The Qt Company | LGPLv3 | Interfaz gráfica nativa | Qt.io |

### Citas Académicas (BibTeX)

```bibtex
@article{bria_rmbg_2_0,
  title={BRIA RMBG-2.0: Background Removal Model with Bilateral Reference},
  author={BRIA AI and Zheng, Peng},
  journal={arXiv preprint arXiv:2411.14444},
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

## 📄 Licencia del Software

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

### Compatibilidad con Licencias de Terceros

La adopción de **GNU AGPLv3** garantiza:
- **Cumplimiento Nativo con YOLO11-seg:** Alineación al 100% con los requerimientos de la licencia AGPL-3.0 de Ultralytics.
- **Integración con Componentes Permisivos:** Compatibilidad total con librerías bajo **LGPLv3** (PySide6 / Qt6), **Apache 2.0** (MobileSAM, SAM, IS-Net, U2-Net, ViTMatte), **BSD 3-Clause** (Real-ESRGAN) e **ISC** (Lucide Icons).
- **Software Libre y Transparente:** Asegura que cualquier derivado o mejora conserve las libertades de auditoría y distribución abierta para la comunidad.
