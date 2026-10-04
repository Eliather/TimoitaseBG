# TimoitaseBG

TimoitaseBG es una aplicación de escritorio **100% local**, moderna y de alto rendimiento desarrollada en **PySide6**. Está diseñada para ilustradores, artistas digitales y fotógrafos, enfocada en la eliminación de fondos, restauración de imágenes y enmascarado preciso mediante Inteligencia Artificial, sin depender de conectividad a internet ni servicios en la nube.

---

## Características Principales

### 1. Eliminación de Fondos Especializada
- **Modelos de Alta Precisión** ejecutados de forma nativa vía **ONNX Runtime**:
  - **IS-Net Anime (Predeterminado)**: Especializado en ilustración, arte digital y cel-shading. Detecta el sujeto respetando elementos blancos (mangas, guantes, piel) sobre fondos claros.
  - **U2-Net Human**: Modelo de segmentación semántica humana entrenado en cuerpo y vestimenta. Previene la confusión entre prendas blancas y fondos de estudio.
  - **BiRefNet**: Orientado a máxima resolución y refinamiento de contornos complejos.
  - **InSPyReNet**: Optimizado para fotografía del mundo real con anti-aliasing subpíxel bilateral.
- **Protección Avanzada de Prendas**: Fusión inteligente que combina la precisión de bordes de alta frecuencia con seguimiento semántico para evitar segmentos faltantes en ropa clara.
- **Fidelidad Cromática**: Retiene el 100% de la información de los píxeles originales, sin algoritmos invasivos que alteren o difuminen la paleta de colores original.

### 2. Procesamiento por Lotes (Batch Processing)
- Automatización completa para procesar directorios enteros de imágenes.
- Pipeline acelerado por GPU capaz de eliminar fondos de forma secuencial y estable.
- Aplica el modelo de IA seleccionado y la configuración de protección de prendas de manera uniforme a todo el lote.

### 3. Restauración y Super-Resolución (Real-ESRGAN)
- Arquitectura basada en `RealESRGAN_x4plus_anime_6B`.
- **Modos de Escalado**: Resoluciones nativas (solo nitidez/denoise), aumento 2x y aumento 4x.
- **Tiling Inteligente**: Procesamiento por teselas para manejar imágenes de gran tamaño sin desbordar la memoria del sistema.
- **Protección contra Saturación de VRAM**: El motor evalúa la resolución objetivo antes del procesamiento. Si excede los límites seguros, activa un fallback automático a CPU para garantizar la finalización de la tarea.

### 4. Pincel y Borrador Interactivo Avanzado
- **Motor de Enmascarado Dinámico**: Controles integrales de tamaño, suavizado, opacidad y dureza.
- **Opacidad y Dureza (0-100%)**: Utiliza un sistema de enmascarado temporal optimizado para simular bordes suaves (estilo aerógrafo) y variaciones de opacidad sin crear artefactos de superposición gráfica.
- **Modos de Herramienta**:
  - **Pincel Restaurador**: Recupera píxeles de la imagen original sobre áreas eliminadas por error.
  - **Borrador**: Elimina píxeles con suavizado subpíxel de alta precisión.

### 5. Herramientas de Comparación y Lienzo
- **Cortina Interactiva (Antes/Después)**: Visor deslizable en tiempo real para inspeccionar la precisión de los cortes frente a la imagen original.
- **Recorte Automático (Auto-Crop)**: Elimina el espacio vacío transparente alrededor del sujeto, aplicando un margen estándar de 10 píxeles.
- **Reemplazo de Fondo**: Integración de colores sólidos preestablecidos y selector personalizado RGB/HEX.

### 6. Aceleración por Hardware y Aislamiento de Procesos
- **Conmutador DirectML / CPU**: Alternancia dinámica e inmediata entre procesamiento GPU y CPU.
- **Aislamiento en Subproceso con Watchdog**: La inferencia en GPU se ejecuta en un proceso aislado. En caso de colapso del controlador (TDR), la aplicación aborta el trabajador y reanuda en CPU sin congelar la interfaz principal.

### 7. Experiencia de Usuario y Gestión de Estado
- **Sistema de Temas**: Soporte nativo para Modo Claro y Modo Oscuro (Enterprise-grade) mediante Qt Style Sheets (QSS).
- **Historial Ilimitado**: Capacidad completa de Deshacer/Rehacer (Ctrl+Z / Ctrl+Y) que registra el procesamiento de IA, ediciones de lienzo y cambios de color.
- **Integración con el Portapapeles**: Soporte para pegado directo (Ctrl+V) desde navegadores, herramientas de recorte o el Explorador de Windows.

---

## Requisitos del Sistema

- **Sistema Operativo**: Windows 10 o Windows 11 (64-bit).
- **Python**: 3.11+.
- **Aceleración por Hardware (Opcional)**: GPU compatible con DirectX 12 (DirectML) o NVIDIA CUDA. Si no se detecta una GPU compatible, la aplicación opera 100% en CPU.

---

## Instalación y Uso (Desarrollo)

### 1. Clonar el repositorio
```bash
git clone <repository_url>
cd TimoitaseBG
```

### 2. Crear y activar entorno virtual
Uso recomendado de `uv` por rendimiento:
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate
```
O mediante la herramienta estándar:
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

El proyecto incluye una suite de pruebas automatizadas (53 tests) que valida el pipeline de inferencia, la seguridad de los subprocesos, los límites del lienzo y la lógica de interfaz:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Empaquetado a Ejecutable (.exe)

Para compilar la aplicación en un ejecutable independiente de Windows mediante **PyInstaller**:

```bash
uv pip install pyinstaller
pyinstaller build.spec
```
El ejecutable final se generará en el directorio `dist/TimoitaseBG/TimoitaseBG.exe`.

---

## Licencia

**Copyright © 2026 TimoitaseBG. Todos los derechos reservados.**

El código fuente principal y diseño de **TimoitaseBG** son propietarios. Queda estrictamente prohibida su copia, modificación, distribución, ingeniería inversa o reventa sin autorización explícita. El software hace uso de librerías y componentes de terceros de código abierto (incluyendo PySide6 bajo LGPL), los cuales se distribuyen de acuerdo a sus propios términos legales de forma dinámica en la aplicación compilada.
