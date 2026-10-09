# TimoitaseBG

TimoitaseBG es una aplicación de escritorio **100% local**, moderna y de alto rendimiento desarrollada en **PySide6**. Está diseñada para ilustradores, artistas digitales y fotógrafos, enfocada en la eliminación de fondos, selección de precisión, restauración de imágenes y enmascarado asistido por Inteligencia Artificial, sin depender de conectividad a internet ni servicios en la nube.

---

## Características Principales

### 1. Eliminación de Fondos Especializada
- **Modelos de Alta Precisión** ejecutados de forma nativa vía **ONNX Runtime**:
  - **IS-Net Anime (Predeterminado)**: Especializado en ilustración, arte digital y cel-shading. Detecta el sujeto respetando elementos blancos (mangas, guantes, piel) sobre fondos claros.
  - **U2-Net Human**: Modelo de segmentación semántica humana entrenado en cuerpo y vestimenta. Previene la confusión entre prendas blancas y fondos de estudio.
  - **BiRefNet**: Orientado a máxima resolución y refinamiento de contornos complejos.
  - **InSPyReNet**: Optimizado para fotografía del mundo real con anti-aliasing subpíxel bilateral.
- **Protección Avanzada de Prendas**: Fusión inteligente que combina la precisión de bordes de alta frecuencia con seguimiento semántico para evitar segmentos faltantes en ropa clara.
- **Fidelidad Cromática**: Retiene el 100% de la información de los píxeles originales, sin algoritmos invasivos que alteren o difuminen la paleta de colores.

### 2. Varita Mágica de Selección (Estilo Photoshop)
- **Selección Inteligente por Similitud de Color**: Detección de fondo y regiones basada en distancia euclidiana en espacio de color con tolerancia configurable (0 a 255, valor óptimo por defecto 32).
- **Control de Topología**:
  - **Contiguo**: Selecciona únicamente áreas conectadas desde el punto de clic (flood-fill).
  - **No Contiguo / Global**: Detecta y selecciona todos los píxeles de tonalidad similar en cualquier parte de la imagen.
  - **Suavizado (Anti-Aliasing)**: Difuminado de bordes para selecciones limpias sin bordes dentados.
- **4 Modos de Operación Booleana**:
  - **Nueva Selección**: Reemplaza cualquier selección previa.
  - **Añadir (+ / Shift)**: Agrega la nueva región a la selección actual.
  - **Restar (- / Alt)**: Excluye la región de la selección actual.
  - **Intersecar (∩)**: Conserva únicamente el área en común entre ambas selecciones.
- **Animación de Hormigas Marchantes (Marching Ants)**: Borde perimetral interactivo animado en tiempo real a 25 FPS para delimitar con exactitud el área seleccionada.
- **Acciones Directas sobre la Selección**:
  - **Borrar Selección**: Convierte instantáneamente el área seleccionada en transparencia limpia (`Supr` / `Delete`).
  - **Invertir Selección**: Invierte el área de trabajo activa (`Ctrl+Shift+I`).
  - **Seleccionar Todo** (`Ctrl+A`) y **Deseleccionar** (`Ctrl+D` / `Esc`).

### 3. Pincel y Borrador Interactivo Avanzado
- **Motor de Enmascarado Dinámico**: Controles integrales de tamaño, suavizado, opacidad y dureza.
- **Opacidad y Dureza (0-100%)**: Utiliza un sistema de enmascarado temporal optimizado para simular bordes suaves (estilo aerógrafo) y variaciones de opacidad sin crear artefactos de superposición gráfica.
- **Modos de Herramienta**:
  - **Pincel Restaurador**: Recupera píxeles de la imagen original sobre áreas eliminadas por error.
  - **Borrador**: Elimina píxeles con suavizado subpíxel de alta precisión.

### 4. Restauración y Super-Resolución (Real-ESRGAN)
- Arquitectura basada en `RealESRGAN_x4plus_anime_6B`.
- **Modos de Escalado**: Resoluciones nativas (solo nitidez/denoise), aumento 2x y aumento 4x.
- **Tiling Inteligente**: Procesamiento por teselas para manejar imágenes de gran tamaño sin desbordar la memoria del sistema.
- **Protección contra Saturación de VRAM**: El motor evalúa la resolución objetivo antes del procesamiento. Si excede los límites seguros, activa un fallback automático a CPU para garantizar la finalización de la tarea.

### 5. Procesamiento por Lotes (Batch Processing)
- Automatización completa para procesar directorios enteros de imágenes.
- Pipeline acelerado por GPU capaz de eliminar fondos de forma secuencial y estable.
- Aplica el modelo de IA seleccionado y la configuración de protección de prendas de manera uniforme a todo el lote.

### 6. Herramientas de Comparación y Lienzo
- **Cortina Interactiva (Antes/Después)**: Visor deslizable en tiempo real para inspeccionar la precisión de los cortes frente a la imagen original.
- **Recorte Automático (Auto-Crop)**: Elimina el espacio vacío transparente alrededor del sujeto, aplicando un margen estándar de 10 píxeles.
- **Reemplazo de Fondo**: Integración de colores sólidos preestablecidos (blanco, negro, gris) y selector de color personalizado RGB/HEX.
- **Lienzo Fluido**: Zoom interactivo (rueda del ratón, atajos y botones 100% / Ajustar) y paneo con clic central o barra espaciadora.

### 7. Iconografía Vectorial Open Source y Diseño Moderno
- **Iconos Vectoriales Open Source (Lucide Icons, Licencia MIT)**:
  - Sustitución total de emojis por gráficos vectoriales SVG nítidos a cualquier resolución o DPI.
  - Renderizado dinámico en tiempo de ejecución mediante `PySide6.QtSvg.QSvgRenderer`.
  - Coloreado adaptativo automático que ajusta el contraste de iconos entre Modo Oscuro y Modo Claro.
- **Barra Lateral Tipo Rail**: Organización ergonómica de herramientas inspirada en Canva y remove.bg, con scroll independiente por secciones.
- **Toggle Dinámico GPU / CPU**: Conmutador animado con retroalimentación en tiempo real del estado de aceleración y diagnóstico de fallos.
- **Temas Oscuro y Claro**: Estilos completos diseñados en Qt Style Sheets (QSS) con paleta moderna y bordes suaves.

### 8. Internacionalización Multilingüe (i18n)
- Soporte completo y conmutación en tiempo real para 4 idiomas:
  - **Español** (`es`)
  - **Inglés** (`en`)
  - **Ruso** (`ru`)
  - **Chino Simplificado** (`zh`)
- Tipografía limpia y profesional sin caracteres emoji incrustados.

### 9. Aceleración por Hardware y Aislamiento de Procesos
- **Conmutador DirectML / CPU**: Alternancia dinámica e inmediata entre procesamiento GPU y CPU.
- **Aislamiento en Subproceso con Watchdog**: La inferencia en GPU se ejecuta en un proceso aislado. En caso de colapso del controlador gráfico (TDR), la aplicación neutraliza el trabajador y reanuda en CPU sin congelar ni cerrar la interfaz principal.
- **Historial Ilimitado**: Capacidad completa de Deshacer/Rehacer (`Ctrl+Z` / `Ctrl+Y`) que registra el procesamiento de IA, ediciones de lienzo, borrados de varita y cambios de color.
- **Integración con Portapapeles**: Pegado directo de imágenes (`Ctrl+V`) desde navegadores, capturadores o el explorador de archivos.

---

## Requisitos del Sistema

- **Sistema Operativo**: Windows 10 o Windows 11 (64-bit).
- **Python**: 3.11+.
- **Aceleración por Hardware (Opcional)**: GPU compatible con DirectX 12 (DirectML) o NVIDIA CUDA. Si no se dispone de GPU compatible, la aplicación funciona al 100% en CPU.

---

## Instalación y Uso (Desarrollo)

### 1. Clonar el repositorio
```bash
git clone <repository_url>
cd TimoitaseBG
```

### 2. Crear y activar entorno virtual
Uso recomendado de `uv` por velocidad:
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate
```
O mediante el módulo estándar de Python:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
uv pip install -r requirements.txt
```

### 4. Iniciar la aplicación
```bash
python app/main.py
```

---

## Pruebas Automatizadas

El proyecto cuenta con una amplia suite de pruebas automatizadas que cubren el catálogo de iconos SVG, la Varita Mágica, la manipulación de estados de imagen, el layout responsivo del sidebar y las funciones del lienzo:

```bash
# Ejecutar todas las pruebas con pytest
pytest tests/

# O ejecutar suites específicas
pytest tests/test_icons.py tests/test_wand_ui_and_integration.py tests/test_sidebar_scroll_and_layout.py
```

---

## Empaquetado a Ejecutable (.exe)

Para compilar la aplicación en un ejecutable binario independiente de Windows mediante **PyInstaller**:

```bash
uv pip install pyinstaller
pyinstaller build.spec
```
El ejecutable resultante se genera en el directorio `dist/TimoitaseBG/TimoitaseBG.exe`.

---

## Licencia y Créditos de Terceros

- **TimoitaseBG**: Copyright © 2026. Todos los derechos reservados. El código fuente principal y diseño de TimoitaseBG son propietarios. Queda prohibida su copia, modificación, distribución, ingeniería inversa o reventa sin autorización explícita.
- **Iconografía**: [Lucide Icons](https://lucide.dev/) (Licencia MIT) - Colección de iconos vectoriales consistentes y de código abierto.
- **Framework UI**: [PySide6 / Qt](https://www.qt.io/) (Licencia LGPLv3).
