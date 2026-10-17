# Avisos de Terceros, Licencias y Atribución de Modelos de IA
## Third-Party Software and Machine Learning Models Notices

**TimoitaseBG v1.2.0** — Autora: **[Eliather](https://github.com/Eliather)**  
TimoitaseBG es software libre distribuido bajo la licencia **GNU Affero General Public License v3.0 (GNU AGPLv3)**.

El código fuente de TimoitaseBG se rige por su propia licencia, mientras que las librerías externas y los modelos de visión por computadora de terceros conservan sus respectivas licencias originales concedidas por sus autores. Este documento detalla la atribución, procedencia, licencias y términos de uso de cada componente y conjunto de pesos (*weights*).

---

## 1. Política de Compatibilidad Comercial y Exclusión de Restricciones No Comerciales

> [!NOTE]
> **Ausencia de Restricciones CC BY-NC:**  
> TimoitaseBG **no incluye ni distribuye pesos bajo licencias No Comerciales** (como Creative Commons `NC`).
> En versiones previas se evaluó *RMBG-2.0* (publicado por BRIA AI bajo `CC BY-NC 4.0`); dicho modelo fue **completamente eliminado del proyecto** y reemplazado por la arquitectura abierta **BiRefNet (licencia MIT)**.  
> Esta decisión garantiza que los usuarios de TimoitaseBG puedan utilizar, redistribuir, recibir donaciones y desplegar la aplicación comercialmente sin infringir términos de licencia.

---

## 2. Inventario de Modelos de Inteligencia Artificial

A continuación se detalla cada modelo de IA soportado por TimoitaseBG:

### A. BiRefNet (Dichotomous Image Segmentation)
* **Función:** Eliminación de fondo dicotómica en alta resolución (1024×1024) para cabello, pelaje, texturas finas y bordes complejos.
* **Autores:** Peng Zheng, De-Ping Fan, Xiao-Bin Tian, et al.
* **Licencia de Código y Pesos:** **MIT License**
* **Repositorio Oficial:** [github.com/ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
* **Distribución en App:** Descarga bajo demanda a través del backend integrado `rembg` (`birefnet-general`).
* **Compatibilidad Comercial:** Totalmente permisiva (MIT).

### B. InSPyReNet (Swin-B Plus Ultra)
* **Función:** Red neuronal piramidal inversa para segmentación semántica de retratos, fotografía real y anti-aliasing subpíxel.
* **Autores:** Taehun Kim, Kunhee Kim, Joonyeong Lee, Dongmin Cha, Jiho Lee, Daijin Kim (POSTECH).
* **Licencia de Código y Pesos:** **MIT License**
* **Repositorio Oficial:** [github.com/plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
* **Fuente ONNX:** [huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX](https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX)
* **Compatibilidad Comercial:** Totalmente permisiva (MIT).

### C. IS-Net Anime (anime-segmentation)
* **Función:** Segmentación de sujetos en arte digital 2D, manga, cómics e ilustración con estilo anime/cel-shading.
* **Autores:** SkyTNT (basado en la arquitectura DIS / IS-Net de Xuebin Qin et al.).
* **Licencia de Código y Pesos:** **Apache License 2.0**
* **Repositorio Oficial:** [github.com/SkyTNT/anime-segmentation](https://github.com/SkyTNT/anime-segmentation) / [huggingface.co/skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg)
* **Notas de Dataset:** Entrenado en ilustraciones de estilo anime. La licencia de los pesos fue ratificada por su autor como Apache 2.0.

### D. MobileSAM (Faster Segment Anything)
* **Función:** Varita mágica por IA de respuesta ultra-rápida (<15 ms) para segmentación interactiva por clic.
* **Autores:** Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, SeungKyu Lee, Choongsang Cho (Kyung Hee University).
* **Licencia de Código y Pesos:** **Apache License 2.0**
* **Repositorio Oficial:** [github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
* **Fuente ONNX:** [huggingface.co/Acly/MobileSAM](https://huggingface.co/Acly/MobileSAM)
* **Compatibilidad Comercial:** Totalmente permisiva (Apache 2.0).

### E. Segment Anything (SAM Mask Decoder)
* **Función:** Decodificador de máscaras a partir del embedding precalculado por MobileSAM.
* **Autores:** Meta AI Research (Alexander Kirillov, Eric Mintun, Nikhila Ravi, et al.).
* **Licencia:** **Apache License 2.0**
* **Repositorio Oficial:** [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* **Compatibilidad Comercial:** Totalmente permisiva (Apache 2.0).

### F. YOLO11-seg (YOLO11n-seg)
* **Función:** Detección y segmentación multiobjeto simultánea de hasta 80 categorías del dataset COCO.
* **Autores:** Ultralytics Inc. (Glenn Jocher & Jing Qiu).
* **Licencia de Código y Pesos:** **GNU Affero General Public License v3.0 (GNU AGPLv3)**
* **Repositorio Oficial:** [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **Fuente ONNX:** [huggingface.co/unileon-robotics/YOLO11-ONNX](https://huggingface.co/unileon-robotics/YOLO11-ONNX)
* **Compatibilidad Comercial:** Compatible bajo AGPLv3. Quienes utilicen y redistribuyan TimoitaseBG con YOLO11 deben mantener el código fuente abierto bajo AGPLv3. Para integrar YOLO11 en productos cerrados o propietarios, Ultralytics ofrece su licencia comercial Enterprise.

### G. Real-ESRGAN (RealESRGAN_x4plus_anime_6B)
* **Función:** Super-resolución 2x/4x y restauración de líneas de dibujo y nitidez sin artefactos.
* **Autores:** Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (Tencent PCG & ARC Lab).
* **Licencia de Código y Pesos:** **BSD 3-Clause License**
* **Repositorio Oficial:** [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
* **Fuente ONNX:** [huggingface.co/deepghs/imgutils-models](https://huggingface.co/deepghs/imgutils-models)
* **Compatibilidad Comercial:** Totalmente permisiva (BSD 3-Clause).

### H. Detector de Ropa y Personas (U²-Net Human Segmentation)
* **Función:** Segmentación semántica de personas y filtro de protección interior de prendas claras (`u2net_human_seg`) para evitar perforaciones accidentales sobre fondos blancos.
* **Autores:** Xuebin Qin et al. (U²-Net, University of Alberta).
* **Licencia:** **Apache License 2.0**
* **Repositorio Oficial:** [github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

---

## 3. Librerías y Recursos del Sistema

| Componente | Autores / Organización | Licencia | Uso en TimoitaseBG |
| :--- | :--- | :--- | :--- |
| **PySide6 / Qt6** | The Qt Company | **LGPLv3** | Interfaz gráfica de usuario y bucle de eventos. |
| **ONNX Runtime** | Microsoft Corporation | **MIT License** | Motor de inferencia para CPU, CUDA, ROCm, CoreML, DirectML y QNN. |
| **Lucide Icons** | Lucide Project Contributors | **ISC License** | Iconografía vectorial SVG en temas claro y oscuro. |
| **OpenCV (opencv-python)** | OpenCV Team | **Apache 2.0** | Operaciones morfológicas, guided filter y transformaciones afines. |
| **Pillow (PIL)** | Alex Clark and Contributors | **HPND License** | Carga, manipulación de color y guardado de imágenes. |
| **NumPy & SciPy** | NumPy / SciPy Developers | **BSD 3-Clause** | Operaciones vectoriales, relleno topológico de huecos y cálculo matricial. |

---

## 4. Descarga Dinámica bajo Demanda

Para optimizar el peso del instalador y garantizar la máxima transparencia en la procedencia de los archivos:
1. El repositorio de TimoitaseBG **no contiene archivos binarios de pesos (`.onnx`) pre-empaquetados**.
2. Cada modelo se descarga dinámicamente y bajo demanda desde sus repositorios oficiales en Hugging Face en la primera ejecución del usuario, alojándose localmente en el directorio de usuario (`app/models/` o caché de `rembg`).
3. El usuario tiene control total sobre los pesos descargados en su disco local.
