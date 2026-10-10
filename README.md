# TimoitaseBG 🎨✂️

<p align="center">
  <img src="https://img.shields.io/badge/Versi%C3%B3n-1.0.0-FF4F79?style=for-the-badge" alt="Versión">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PySide6%20%2F%20Qt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/Inferencia-ONNX%20Runtime%20%7C%20DirectML-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="DirectML">
  <img src="https://img.shields.io/badge/Aceleraci%C3%B3n-GPU%20%26%20CPU-059669?style=for-the-badge" alt="Aceleración">
  <img src="https://img.shields.io/badge/Tests-75%2F75%20Pasados-10B981?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/Licencia-GNU%20AGPLv3-326CE5?style=for-the-badge&logo=gnu&logoColor=white" alt="Licencia GNU AGPLv3">
  <img src="https://img.shields.io/badge/Privacidad-100%25%20Local%20%26%20Offline-6366F1?style=for-the-badge" alt="Privacidad">
</p>

**TimoitaseBG** es una estación de trabajo de escritorio **100% local**, moderna y de alto rendimiento desarrollada en **PySide6 / Qt6**. Diseñada para ilustradores, artistas de anime/manga, diseñadores gráficos y fotógrafos, integra los modelos de visión por inteligencia artificial de vanguardia (**RMBG-2.0**, **MobileSAM**, **YOLO11-seg**, **InSPyReNet** y **Real-ESRGAN**) combinados con herramientas de enmascarado y selección manual con precisión de subpíxel estilo Photoshop, todo ejecutado localmente sin depender de la nube, sin suscripciones y sin enviar tus imágenes a servidores externos.

---

## 📑 Tabla de Contenidos

1. [¿Por qué TimoitaseBG?](#-por-qué-timoitasebg)
2. [Características Principales](#-características-principales)
   - [Eliminación de Fondos de Nueva Generación](#1-eliminación-de-fondos-de-nueva-generación-ai)
   - [Varita Mágica Híbrida: Color e IA (MobileSAM)](#2-varita-mágica-de-selección-híbrida)
   - [Detección y Aislamiento de Sujetos (YOLO11-seg)](#3-detección-y-aislamiento-multiobjeto-yolo11-seg)
   - [Super-Resolución y Restauración (Real-ESRGAN Anime)](#4-restauración-y-super-resolución-realesrgan)
   - [Pincel Restaurador y Borrador de Precisión](#5-pincel-restaurador-y-borrador-dinámico)
   - [Herramientas de Lienzo y Comparación](#6-herramientas-de-lienzo-y-comparación-antesdespués)
   - [Procesamiento por Lotes (Batch Studio)](#7-procesamiento-por-lotes-batch-processing)
   - [Diseño Ergonómico, Iconos Vectoriales e i18n](#8-diseño-ergonómico-iconografía-vectorial-e-i18n)
3. [Arquitectura, Concurrencia y Estabilidad](#-arquitectura-concurrencia-y-estabilidad)
4. [Requisitos del Sistema](#-requisitos-del-sistema)
5. [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
6. [Pruebas Automatizadas](#-pruebas-automatizadas)
7. [Compilación a Ejecutable (.exe)](#-compilación-a-ejecutable-exe)
8. [Créditos, Licencias de Modelos y Citas Académicas](#-créditos-licencias-de-modelos-y-citas-académicas)
9. [Licencia del Software](#-licencia-del-software)

---

## 💡 ¿Por qué TimoitaseBG?

- **🔒 Privacidad Absoluta**: Cero telemetría. Tus fotos, ilustraciones y bocetos se procesan y almacenan únicamente en tu almacenamiento local.
- **⚡ Aceleración por Hardware DirectML**: Compatible con cualquier GPU moderna en Windows (NVIDIA, AMD Radeon, Intel Arc) a través de DirectX 12, con conmutación instantánea a CPU.
- **🛡️ Cero Saturación de Recursos**: Gestión de concurrencia optimizada que asigna hilos de cálculo sin ahogar la CPU ni congelar la interfaz gráfica, manteniendo 60 FPS fluidos.
- **💎 100% Fidelidad de Color**: A diferencia de otras herramientas que comprimen o difuminan la paleta cromática, el pipeline conserva los píxeles RGB originales con 0% de alteración.
- **🎨 Flujo de Trabajo No Destructivo**: Pila de Deshacer/Rehacer ilimitada con integración al portapapeles y restauración en cualquier momento.

---

## 🚀 Características Principales

### 1. Eliminación de Fondos de Nueva Generación (AI)

Pipeline de vanguardia ejecutado nativamente en **ONNX Runtime** con preprocesamiento y postprocesamiento morfológico adaptativo:

- **RMBG-2.0 (BiRefNet SOTA 1024×1024)**: El estándar más reciente y avanzado en segmentación de imágenes dicotómicas. Extrae con asombrosa fidelidad detalles finos como hebras de cabello, pelaje, texturas transparentes y objetos complejos.
- **InSPyReNet (Swin-B Plus Ultra)**: Red neuronal piramidal inversa de alta precisión con despill cromático y refinamiento subpíxel bilateral.
- **IS-Net Anime**: Especializado en arte digital 2D, manga, cómics y cel-shading.
- **U2-Net Human**: Segmentación anatómica humana entrenada para aislar personas de forma íntegra.
- **🛡️ Detector Semántico de Ropa**: Algoritmo híbrido que analiza las prendas de vestir para evitar perforaciones accidentales en camisas, mangas o vestidos blancos sobre fondos claros.

---

### 2. Varita Mágica de Selección Híbrida

El sistema de varita mágica cuenta con dos motores complementarios:

#### A. Modo IA (MobileSAM)
- **Selección Semántica por Clic**: Haz clic en cualquier parte de la imagen para que la IA entienda el objeto completo y genere una máscara *pixel-perfect* en menos de **15 milisegundos**.
- **Arquitectura Dividida**: El *Image Encoder* precomputa el embedding de la imagen en un hilo de fondo de baja prioridad, permitiendo clics sucesivos ultrarrápidos mediante el *Mask Decoder*.

#### B. Modo Color (Estilo Photoshop)
- **Tolerancia Configurable (0–255)**: Ajusta la sensibilidad de detección basada en distancia euclidiana en el espacio RGB.
- **Contiguo vs. No Contiguo**:
  - *Contiguo*: Selecciona áreas continuas conectadas (flood-fill).
  - *No Contiguo (Global)*: Detecta y selecciona todas las islas de color similares en la imagen.
- **Anti-Aliasing**: Atenuación perimetral suave para evitar bordes dentados (*aliased edges*).

#### Operaciones Booleanas y Hormigas Marchantes
- **4 Modos de Combinación**: Nueva Selección, Añadir (`Shift`), Restar (`Alt`) e Intersecar (`∩`).
- **Marching Ants a 25 FPS**: Contorno animado vectorial de hormigas marchantes que delimita con precisión matemática el área activa.
- **Acciones Rápidas**:
  - **Borrar Selección** (`Supr` / `Delete`): Convierte el área seleccionada en transparencia limpia.
  - **Invertir Selección** (`Ctrl+Shift+I`).
  - **Seleccionar Todo** (`Ctrl+A`) y **Deseleccionar** (`Ctrl+D` / `Esc`).

---

### 3. Detección y Aislamiento Multiobjeto (YOLO11-seg)

- **Identificación Instantánea**: Detección y segmentación simultánea de hasta 80 clases de objetos comunes (personas, mascotas, prendas, vehículos, muebles, etc.) mediante **YOLO11-seg**.
- **Chips Interactivos en la Barra Lateral**: Cada sujeto detectado se presenta con su respectivo emoji, nombre de clase, porcentaje de confianza y color identificativo.
- **Acciones en 1 Clic**:
  - **Aislar Sujeto**: Vuelve transparente todo el entorno manteniendo intacto el sujeto elegido.
  - **Seleccionar Sujeto**: Carga la silueta del objeto directamente en las hormigas marchantes para su edición manual.

---

### 4. Restauración y Super-Resolución (Real-ESRGAN)

- **Modelo `RealESRGAN_x4plus_anime_6B`**: Especializado en eliminar ruido, restaurar artefactos de compresión JPEG y reescalar ilustraciones con líneas de tinta definidas.
- **Modos de Operación**:
  - *Mejora de Nitidez* (1x / denoise).
  - *Aumento 2x*.
  - *Aumento 4x*.
- **Tiling Inteligente (512×512)**: Procesa imágenes grandes en bloques para evitar el desbordamiento de memoria.
- **Protección Automática Anti-Saturación de VRAM**: Evalúa las dimensiones de la imagen; si excede el umbral seguro (~2.07 megapíxeles), commuta automáticamente a CPU para salvaguardar la GPU.

---

### 5. Pincel Restaurador y Borrador Dinámico

- **Pincel Restaurador**: Recupera selectivamente píxeles de la imagen original en áreas que hayan sido eliminadas por error.
- **Borrador de Precisión**: Elimina manualmente zonas no deseadas con bordes limpios.
- **Parámetros Ajustables**: Tamaño (2 a 200 px), Suavizado (0 a 100%), Opacidad (1 a 100%) y Dureza (0 a 100%).

---

### 6. Herramientas de Lienzo y Comparación (Antes/Después)

- **Cortina Deslizable Antes/Después**: Manija interactiva en tiempo real con badges para comparar el corte frente al lienzo original.
- **Recorte Automático (Auto-Crop)**: Recorta los márgenes transparentes vacíos ajustando el lienzo con un padding estándar de 10 px.
- **Fondo Sólido / Reemplazo**: Preajustes para comprobar contraste (blanco, negro, gris) y selector de color personalizado RGB/HEX.
- **Control de Navegación**: Zoom interactivo fluido (rueda del ratón, `Ctrl +`, `Ctrl -`, Ajustar a Vista y 100%) y paneo con clic central o barra espaciadora.

---

### 7. Procesamiento por Lotes (Batch Processing)

- Procesa carpetas enteras de imágenes automáticamente sin intervención manual.
- Aplica el modelo de IA seleccionado y la protección de prendas de manera uniforme.
- Salida secuencial en formato PNG transparente con barra de progreso en tiempo real.

---

### 8. Diseño Ergonómico, Iconografía Vectorial e i18n

- **Iconografía 100% Vectorial (Lucide Icons, Licencia ISC)**: Gráficos vectoriales SVG nítidos en cualquier escala de pantalla o DPI, con adaptación dinámica de color según el tema activo.
- **Temas Modernos Oscuro y Claro**: Estilos completos desarrollados en Qt Style Sheets (QSS).
- **Internacionalización Multilingüe (i18n)**:
  - 🇪🇸 **Español** (`es`)
  - 🇺🇸 **Inglés** (`en`)
  - 🇷🇺 **Ruso** (`ru`)
  - 🇨🇳 **Chino Simplificado** (`zh`)

---

## ⚙️ Arquitectura, Concurrencia y Estabilidad

TimoitaseBG implementa una arquitectura defensiva diseñada para garantizar estabilidad continua en entornos de producción:

```
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

1. **Aislamiento en Subproceso GPU (`PersistentGPUWorker`)**:
   La inferencia en GPU se ejecuta en un proceso de sistema operativo independiente (`multiprocessing.Process`). Si el controlador de pantalla sufre un cuelgue por *Timeout Detection and Recovery* (TDR de DirectX/DirectML), el proceso supervisor lo neutraliza y commuta de forma transparente a CPU sin congelar la ventana principal.
2. **Prevención de Saturación de CPU (`get_optimized_session_options`)**:
   ONNX Runtime limita dinámicamente `intra_op_num_threads` a un máximo de 4 hilos (o la mitad de los núcleos físicos) e `inter_op_num_threads = 1`. Esto previene picos al 100% de uso de CPU y garantiza que el sistema operativo y la aplicación permanezcan completamente ágiles.
3. **Serialización Segura y Re-entrancia**:
   Las peticiones a los modelos están sincronizadas mediante cerrojos re-entrantes (`RLock`), impidiendo colisiones en la cola de ejecución de DirectML y garantizando que las tareas de varita e inferencia se ejecuten de manera estrictamente secuencial.
4. **Cierre Limpio de la Aplicación**:
   El evento `closeEvent` notifica interrupción (`requestInterruption()`) y espera la culminación ordenada de todos los hilos secundarios antes del desmontaje.

---

## 💻 Requisitos del Sistema

- **Sistema Operativo**: Windows 10 o Windows 11 (64-bit).
- **Procesador (CPU)**: Intel Core i3 / AMD Ryzen 3 o superior (4 núcleos recomendados).
- **Memoria RAM**: 8 GB mínimo (16 GB recomendado para imágenes 4K).
- **Aceleración GPU (Opcional pero recomendada)**:
  - Cualquier tarjeta gráfica compatible con **DirectX 12 (DirectML)**: NVIDIA GeForce, AMD Radeon o Intel Arc.
  - O aceleración mediante **NVIDIA CUDA**.
  - *Nota*: Si no se dispone de GPU dedicada, la aplicación opera al 100% en CPU de forma segura.
- **Espacio en Disco**: ~1.5 GB libres para el entorno y caché de pesos ONNX.

---

## 📦 Instalación y Puesta en Marcha

### Opción 1: Inicio Rápido (Recomendada en Windows)
Si el repositorio ya cuenta con el entorno virtual preparado, haz doble clic en:
```bat
iniciar.bat
```
El script localiza automáticamente el intérprete de Python, inyecta las variables de entorno e inicia la aplicación esquivando posibles restricciones de *Device Guard* o *AppLocker*.

---

### Opción 2: Instalación Manual desde la Terminal

#### 1. Clonar el repositorio
```bash
git clone https://github.com/Eliather/TimoitaseBG.git
cd TimoitaseBG
```

#### 2. Crear y activar el entorno virtual
Recomendado con `uv` (ultrarrápido):
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate
```
O con el módulo nativo de Python:
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Instalar las dependencias requeridas
```bash
uv pip install -r requirements.txt
# O con pip estándar:
pip install -r requirements.txt
```

#### 4. Ejecutar la aplicación
```bash
python app/main.py
```

*Los pesos de los modelos de IA se descargarán de forma automática, progresiva y bajo demanda desde Hugging Face la primera vez que utilices cada herramienta correspondiente.*

---

## 🧪 Pruebas Automatizadas

TimoitaseBG cuenta con una suite integral de **75 pruebas unitarias y de integración** que validan la seguridad de hilos, los modelos de inferencia, la interfaz gráfica y las utilidades del lienzo:

```bash
# Ejecutar la suite completa de pruebas
pytest -v

# Ejecutar pruebas específicas de seguridad y modelos
pytest tests/test_thread_and_resource_safety.py tests/test_sam_selection.py tests/test_yolo_detection.py tests/test_rmbg2.py -v
```

---

## 🛠️ Compilación a Ejecutable (.exe)

Para compilar la aplicación en un ejecutable binario autónomo para Windows mediante **PyInstaller**:

```bash
uv pip install pyinstaller
pyinstaller build.spec
```
El ejecutable resultante se generará en la carpeta `dist/TimoitaseBG/TimoitaseBG.exe`.

---

## 📚 Créditos, Licencias de Modelos y Citas Académicas

TimoitaseBG hace uso de modelos de aprendizaje profundo y librerías de código abierto desarrollados por la comunidad de investigación internacional. En cumplimiento de las licencias de software y buenas prácticas de atribución académica, se reconoce y acredita formalmente a los autores originales:

### Tabla de Atribución de Modelos

| Modelo / Componente | Autores / Organización | Licencia | Propósito en TimoitaseBG | Referencia / Repositorio |
| :--- | :--- | :--- | :--- | :--- |
| **RMBG-2.0** | [BRIA AI](https://bria.ai/) & ZhengPeng7 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Eliminación de fondo de máxima resolución (1024×1024) y pelo fino | [Hugging Face](https://huggingface.co/briaai/RMBG-2.0) • [Paper](https://arxiv.org/abs/2411.14444) |
| **BiRefNet** | Zheng et al. (BiRefNet) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Arquitectura de segmentación dicotómica bilateral | [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| **MobileSAM** | Zhang et al. (Kyung Hee Univ.) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Varita mágica por IA interactiva en tiempo real | [GitHub](https://github.com/ChaoningZhang/MobileSAM) • [Acly ONNX](https://huggingface.co/Acly/MobileSAM) |
| **Segment Anything (SAM)** | Meta AI Research | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Modelo fundacional de segmentación de objetos | [GitHub](https://github.com/facebookresearch/segment-anything) |
| **YOLO11-seg** | Ultralytics Inc. (Glenn Jocher et al.) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | Detección y segmentación multiobjeto instantánea | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **InSPyReNet** | Kim et al. (POSTECH) | [MIT](https://opensource.org/licenses/MIT) | Eliminación de fondo con pirámide inversa de salience | [GitHub](https://github.com/TaehunKim/InSPyReNet) |
| **Real-ESRGAN** | Wang et al. (ARC Lab, Tencent PCG) | [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) | Super-resolución, denoise y restauración de anime | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| **IS-Net (DIS5K)** | Qin et al. (DIS5K) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Modelo de segmentación especializado en arte 2D | [GitHub](https://github.com/xuebinqin/DIS) |
| **U^2-Net** | Qin et al. (Univ. of Alberta) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Segmentación de silueta humana y ropa | [GitHub](https://github.com/xuebinqin/U-2-Net) |
| **ViTMatte** | Yao et al. (HUST & Kuaishou) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Refinamiento de bordes y matting con Vision Transformers | [GitHub](https://github.com/hustvl/ViTMatte) |
| **Lucide Icons** | Lucide Project | [ISC License](https://lucide.dev/license) | Iconografía vectorial SVG escalable | [Lucide.dev](https://lucide.dev/) |
| **PySide6 / Qt** | The Qt Company | [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html) | Interfaz gráfica nativa multiplataforma | [Qt.io](https://www.qt.io/) |

---

### Aviso Legal sobre el Uso de Modelos

> [!IMPORTANT]
> - **BRIA RMBG-2.0**: Distribuido bajo licencia **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**. Está permitido su uso personal, educativo y de investigación. Para aplicaciones comerciales directas, se debe obtener una licencia comercial correspondiente con [BRIA AI](https://bria.ai/).
> - **Ultralytics YOLO11**: Distribuido bajo licencia **GNU Affero General Public License v3.0 (AGPL-3.0)**. Ultralytics requiere que el software que integre o distribuya este modelo cumpla con las disposiciones de la licencia AGPL-3.0 o cuente con una licencia comercial de Ultralytics.
> - **Modelos con Licencias Apache 2.0, MIT y BSD 3-Clause**: Permiten su integración manteniendo los avisos de derechos de autor y licencias originales correspondientes.

---

### Citas Académicas

Si utilizas TimoitaseBG o sus modelos integrados en proyectos académicos o de investigación, te sugerimos citar las publicaciones originales:

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
  year={2024},
  url={https://github.com/ultralytics/ultralytics}
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

Este proyecto es software libre y de código abierto distribuido bajo los términos de la **Licencia Pública General Affero de GNU versión 3.0 (GNU AGPLv3)**. Consulta el archivo [LICENSE](file:///c:/Users/danie/Documents/Trabajos/Cosas/TimoitaseBG/LICENSE) para el texto completo oficial.

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
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

### 🤝 Compatibilidad Total con el Ecosistema y Modelos de IA
Al publicarse bajo la licencia **GNU AGPLv3**:
- **Ultralytics YOLO11-seg**: Cumple al 100% y de forma nativa con los requerimientos de la licencia **GNU AGPL-3.0** de Ultralytics, eliminando cualquier incompatibilidad o conflicto de derechos de autor.
- **Librerías y Modelos Permisivos**: Es plenamente compatible con las dependencias bajo **LGPLv3** (PySide6 / Qt6), **Apache 2.0** (MobileSAM, Segment Anything, IS-Net, U2-Net, ViTMatte), **BSD 3-Clause** (Real-ESRGAN) y **MIT** (InSPyReNet, ONNX Runtime, Lucide Icons).
- **Libertad y Transparencia**: Asegura que las mejoras, optimizaciones y bifurcaciones de TimoitaseBG se mantengan como software libre, abierto y auditable para beneficio de toda la comunidad artística y tecnológica.
