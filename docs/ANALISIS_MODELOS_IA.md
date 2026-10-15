# Auditoría Integral, Optimización de Máscaras y Estabilidad de GPU (`TimoitaseBG` v1.0.0)

**Fecha:** 14 de Octubre de 2026  
**Proyecto:** TimoitaseBG (v1.0.0)  
**Stack:** PySide6, ONNX Runtime (DirectML / CUDA / CPU), OpenCV, NumPy, SciPy  

---

## 1. Resumen Ejecutivo de Problemas Detectados

1. **Contaminación de Bordes por Unión Directa (`np.maximum`):**
   En `app/core/gpu_worker.py` (líneas 190-204) y `app/core/bg_remover.py` (líneas 723-733), la máscara continua de alta resolución (`RMBG-2.0` o `InSPyReNet` a 1024×1024) se une directamente mediante `np.maximum(final_alpha, human_alpha)` con la máscara de baja resolución de `u2net_human_seg` (320×320). Cuando el borde redondeado de U2-Net sobresale 1-3 píxeles del contorno real, destruye el corte subpíxel del modelo principal y genera un halo exterior borroso.
2. **Sesgo de Dominio en Ilustración 2D (`isnet-anime`):**
   Ejecutar `u2net_human_seg` (entrenado en fotografías reales) sobre ilustraciones de anime o manga introduce falsos positivos en fondos estilizados y duplica innecesariamente el tiempo de inferencia.
3. **Recorte Destructivo con Protomáscaras de YOLO11-seg (160×160):**
   En `app/ui/main_window.py` (`perform_isolate_subject` y `perform_select_subject`), se aplica directamente la máscara booleana cruda de YOLO11-seg (proveniente de protomáscaras de 160×160 escaladas linealmente), produciendo bordes escalonados y dentados.
4. **Desconexión de Herramientas respecto a `gpu_manager.py` y Riesgo de OOM en VRAM:**
   - **MobileSAM (`sam_manager.py`):** Inicializa sesiones ONNX directamente en el proceso principal de la UI con `get_inference_providers()` sin consultar `get_gpu_manager().is_gpu_enabled()` ni `has_orphaned_worker()`. Además, al apagar la GPU desde la UI (`_on_device_mode_changed`), las sesiones de MobileSAM permanecen alojadas en VRAM.
   - **Colisión de VRAM en `PersistentGPUWorker` (`gpu_worker.py`):** Al ejecutar `restore_image` (Real-ESRGAN), no se liberaban previamente las sesiones en `_cached_sessions` (`rmbg-2.0` + `u2net_human_seg`), manteniendo hasta 3 modelos en VRAM simultáneamente y provocando errores TDR/OOM en tarjetas de 4 GB a 6 GB.
   - **Recarga de Sesiones en CPU Fallback (`bg_remover.py`):** `_remove_background_cpu_local` instanciaba `rembg.new_session()` en cada llamada sin guardarla en `self._sessions`, penalizando severamente el procesamiento por lotes en CPU.

---

## 2. Matriz de Verificación de Herramientas con `gpu_manager.py`

| Herramienta / Módulo | ¿Delega a `GPUWorkerManager`? | Riesgo Detectado | Solución Aplicada en este Parche |
| :--- | :---: | :--- | :--- |
| **Quitar Fondo (`bg_remover.py`)** | ✅ Sí (Subproceso) | Sin caché de `rembg` en modo CPU fallback. | Nuevo método `_get_rembg_cpu_session()` cacheado en `self._sessions` + fusión `Core-Only`. |
| **Escalado (`restorer.py` / `gpu_worker.py`)** | ✅ Sí (Subproceso) | `Real-ESRGAN` se carga en VRAM sin descargar antes `RMBG-2.0` y `U2-Net`. | Llamada automática a `self._clear_rembg_sessions()` antes de inicializar `Real-ESRGAN` y viceversa. |
| **Varita Mágica IA (`sam_manager.py`)** | ⚠️ Proceso UI (Hilo QThread) | Ignora si el usuario apagó la GPU o si hay un worker huérfano bloqueado en `gpu_manager.py`. | Verificación estricta de `gpu_mgr.is_gpu_enabled()` y nuevo método `unload_sessions()` conectado al switch de GPU. |
| **Detección (`yolo_detector.py` / `main_window.py`)** | ⚠️ Proceso UI (Hilo QThread) | Usa máscara cruda de 160×160 y no precalcula MobileSAM. | Precalcula MobileSAM al terminar YOLO y refina con `BBox + Polo de Inaccesibilidad (distanceTransform)` + `Guided Filter`. |
| **Procesamiento por Lotes (`workers.py`)** | ✅ Sí (vía `bg_remover`) | Deja descriptores `Image.open(f)` abiertos durante el envío por `Queue`. | Uso de `with Image.open(f)` y `.close()` explícito en cada iteración del lote. |

---

## 3. Código de Implementación Archivo por Archivo

### 3.1. `app/core/bg_remover.py`

#### A. Añadir función `fuse_clothing_protection_core_only` (debajo de `postprocess_mask`, línea ~472)
```python
def fuse_clothing_protection_core_only(
    main_alpha: np.ndarray,
    clothing_alpha: np.ndarray,
    inner_erosion_px: int = 12,
    cloth_confidence_thresh: int = 75,
) -> np.ndarray:
    """
    Fusiona la máscara de ropa (U2-Net 320x320) con la máscara principal (1024x1024)
    restringiendo la inyección al núcleo interior. Evita perforaciones en prendas claras
    (tanto huecos cerrados como 'bahías' abiertas en cuellos/mangas) sin contaminar jamás
    el borde subpíxel exterior con halos de baja resolución.
    """
    if main_alpha is None or clothing_alpha is None:
        return main_alpha

    h, w = main_alpha.shape[:2]
    if clothing_alpha.shape[:2] != (h, w):
        clothing_alpha = cv2.resize(clothing_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    # Escalar el radio de erosión proporcionalmente a la resolución real de la imagen
    scale_factor = max(1.0, min(h, w) / 1024.0)
    eff_erosion = max(6, int(round(inner_erosion_px * scale_factor)))

    ksize = 2 * eff_erosion + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # 1. Erosionar la propia máscara de ropa para destruir su halo exterior de baja resolución
    eroded_clothing = cv2.erode(clothing_alpha, kernel)

    # 2. Calcular silueta cerrada del modelo principal para proteger huecos internos
    main_binary = (main_alpha > 30).astype(np.uint8)
    if np.any(main_binary):
        close_k = max(15, int(round(25 * scale_factor)) | 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        main_closed = cv2.morphologyEx(main_binary, cv2.MORPH_CLOSE, close_kernel)
        main_solid = scipy.ndimage.binary_fill_holes(main_closed).astype(np.uint8)
        safe_main_core = cv2.erode(main_solid, kernel)
    else:
        safe_main_core = np.zeros_like(main_alpha, dtype=np.uint8)

    # 3. Inyectar únicamente donde el núcleo erosionado de ropa sea confiable
    valid_clothing = (eroded_clothing >= cloth_confidence_thresh) | (
        (clothing_alpha >= cloth_confidence_thresh) & (safe_main_core > 0)
    )

    if not np.any(valid_clothing):
        return main_alpha

    # 4. Transición suave interna para no dejar escalones duros dentro de la prenda
    injection_alpha = np.where(valid_clothing, eroded_clothing, 0).astype(np.uint8)
    blur_k = max(5, int(round(7 * scale_factor)) | 1)
    injection_alpha = cv2.GaussianBlur(injection_alpha, (blur_k, blur_k), sigmaX=2.0)

    return np.maximum(main_alpha, injection_alpha)
```

#### B. Reemplazar `_remove_background_cpu_local` en `BackgroundRemover`
```python
    def _get_rembg_cpu_session(self, model_name: str):
        """Obtiene o reutiliza en caché una sesión de rembg en CPU para evitar recargas en lotes."""
        cache_key = f"rembg_cpu_{model_name}"
        if cache_key not in self._sessions:
            with self._lock:
                if cache_key not in self._sessions:
                    import rembg
                    self._sessions[cache_key] = rembg.new_session(
                        model_name, providers=["CPUExecutionProvider"]
                    )
        return self._sessions[cache_key]

    def _remove_background_cpu_local(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
    ) -> Image.Image:
        """
        Ejecución local en CPU con preservación 100% íntegra de colores originales.
        """
        if image.mode not in ("RGB", "RGBA"):
            input_img = image.convert("RGB")
        else:
            input_img = image.copy()

        if model_name == "rmbg-2.0":
            logger.info("[CPU Fallback/Local] Ejecutando RMBG-2.0 (BiRefNet 1024x1024)...")
            session = self.get_rmbg2_session(force_cpu=True)
            final_alpha = self._predict_rmbg2(session, input_img)
        elif model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            logger.info(f"[CPU Fallback/Local] Ejecutando {model_name} en CPU...")
            sess = self._get_rembg_cpu_session(model_name)
            out = rembg.remove(input_img, session=sess)
            final_alpha = np.array(out.split()[-1], dtype=np.uint8)
        else:
            logger.info("[CPU Fallback/Local] Ejecutando InSPyReNet (Swin-B 1024x1024)...")
            session = self.get_inspyrenet_session(force_cpu=True)
            raw_mask = self._predict_inspyrenet(session, input_img)
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        # Fusión Core-Only con Detector Semántico de Ropa (se omite en isnet-anime y u2net_human_seg)
        if enable_clothing_protection and model_name not in ("u2net_human_seg", "isnet-anime"):
            try:
                import rembg
                human_sess = self._get_rembg_cpu_session("u2net_human_seg")
                human_out = rembg.remove(input_img, session=human_sess)
                human_alpha = np.array(human_out.split()[-1], dtype=np.uint8)
                if np.any(human_alpha > 50):
                    final_alpha = fuse_clothing_protection_core_only(
                        main_alpha=final_alpha,
                        clothing_alpha=human_alpha,
                        inner_erosion_px=12,
                    )
            except Exception as e:
                logger.warning(f"Protección de ropa u2net_human_seg omitida en CPU: {e}")

        if image.mode == "RGBA":
            orig_alpha = np.array(image.split()[-1], dtype=np.uint8)
            final_alpha = np.minimum(final_alpha, orig_alpha)

        output = input_img.convert("RGB").convert("RGBA")
        output.putalpha(Image.fromarray(final_alpha, mode="L"))
        return output
```

---

### 3.2. `app/core/gpu_worker.py`

#### A. Actualizar `get_inspyrenet_session`, `get_rmbg2_session`, `remove_background` y `restore_image`
Garantiza exclusión mutua en VRAM entre los modelos de segmentación y `Real-ESRGAN`, y aplica `fuse_clothing_protection_core_only`:

```python
    def get_inspyrenet_session(self, providers=None):
        """Obtiene o inicializa una sesión ONNX para InSPyReNet en GPU Worker."""
        if "inspyrenet" not in self._cached_sessions:
            if self._cached_sessions or self._restorer_instance is not None:
                logger.info("Liberando sesiones previas de VRAM para cargar inspyrenet...")
                self._clear_rembg_sessions()

            from app.core.model_fetcher import ensure_model_file
            model_path = ensure_model_file("inspyrenet.onnx")
            import onnxruntime as ort
            eff_providers = self._get_providers(providers)
            sess_options = get_optimized_session_options() or ort.SessionOptions()
            logger.info(f"Cargando sesión inspyrenet con providers: {eff_providers}...")
            self._cached_sessions["inspyrenet"] = ort.InferenceSession(
                str(model_path), sess_options, providers=eff_providers
            )
        return self._cached_sessions["inspyrenet"]

    def get_rmbg2_session(self, providers=None):
        if "rmbg2" not in self._cached_sessions:
            if self._cached_sessions or self._restorer_instance is not None:
                logger.info("Liberando sesiones previas de VRAM para cargar RMBG-2.0...")
                self._clear_rembg_sessions()

            from app.core.model_fetcher import ensure_model_file
            model_path = ensure_model_file("rmbg-2.0.onnx")
            import onnxruntime as ort
            eff_providers = self._get_providers(providers)
            sess_options = get_optimized_session_options() or ort.SessionOptions()
            logger.info(f"Cargando sesión RMBG-2.0 con providers: {eff_providers}...")
            self._cached_sessions["rmbg2"] = ort.InferenceSession(
                str(model_path), sess_options, providers=eff_providers
            )
        return self._cached_sessions["rmbg2"]
```

```python
        # En PersistentGPUWorker.remove_background:
        elif model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            logger.info(f"[GPUWorker:{os.getpid()}] Ejecutando {model_name} en GPU con providers: {eff_providers}...")
            if model_name not in self._cached_sessions:
                if self._cached_sessions or self._restorer_instance is not None:
                    self._clear_rembg_sessions()
                self._cached_sessions[model_name] = rembg.new_session(model_name, providers=eff_providers)

            session = self._cached_sessions[model_name]
            out = rembg.remove(input_img, session=session)
            final_alpha = np.array(out.split()[-1], dtype=np.uint8)

        else:
            from app.core.bg_remover import (
                _preprocess_inspyrenet,
                _postprocess_inspyrenet_mask,
                postprocess_inspyrenet,
            )
            logger.info(
                f"[GPUWorker:{os.getpid()}] Ejecutando InSPyReNet (Swin-B 1024x1024)..."
            )
            session = self.get_inspyrenet_session(providers=providers)
            tensor, orig_size = _preprocess_inspyrenet(input_img)
            input_name = session.get_inputs()[0].name
            out = session.run(None, {input_name: tensor})[0]
            raw_mask = _postprocess_inspyrenet_mask(out, orig_size)
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        # Fusión Core-Only con Detector Semántico de Ropa (omite isnet-anime por sesgo de dominio 2D)
        if enable_clothing_protection and model_name not in ("u2net_human_seg", "isnet-anime"):
            try:
                import rembg
                from app.core.bg_remover import fuse_clothing_protection_core_only

                human_sess_key = "u2net_human_seg"
                if human_sess_key not in self._cached_sessions:
                    self._cached_sessions[human_sess_key] = rembg.new_session(
                        human_sess_key, providers=eff_providers
                    )
                human_sess = self._cached_sessions[human_sess_key]
                human_out = rembg.remove(input_img, session=human_sess)
                human_alpha = np.array(human_out.split()[-1], dtype=np.uint8)
                if np.any(human_alpha > 50):
                    final_alpha = fuse_clothing_protection_core_only(
                        main_alpha=final_alpha,
                        clothing_alpha=human_alpha,
                        inner_erosion_px=12,
                    )
            except Exception as e:
                logger.warning(f"Protección de ropa u2net_human_seg omitida en GPU: {e}")
```

```python
    # En PersistentGPUWorker.restore_image:
    def restore_image(
        self,
        image: Image.Image,
        scale_mode: str = "x2",
        use_tiling: bool = True,
        tile_size: int = 512,
        tile_pad: int = 16,
        providers=None,
    ) -> Image.Image:
        """
        Ejecuta restauración Real-ESRGAN en el proceso aislado de GPU.
        Libera previamente las sesiones de segmentación para evitar picos de VRAM (>4GB).
        """
        from app.core.restorer import ImageRestorer

        if self._restorer_instance is None:
            if self._cached_sessions:
                logger.info("Worker: Liberando sesiones de segmentación para cargar Real-ESRGAN...")
                self._clear_rembg_sessions()

            self._restorer_instance = ImageRestorer()
            eff_providers = self._get_providers(providers)
            import onnxruntime as ort
            from app.core.model_fetcher import ensure_model_file
            model_path = ensure_model_file(ImageRestorer.MODEL_FILENAME)
            sess_options = get_optimized_session_options() or ort.SessionOptions()
            logger.info(f"Worker: Inicializando Real-ESRGAN con providers: {eff_providers}...")
            self._restorer_instance._session = ort.InferenceSession(
                str(model_path), sess_options, providers=eff_providers
            )

        return self._restorer_instance.restore_image(
            image,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )
```

---

### 3.3. `app/core/gpu_manager.py`

#### A. Ajustar `_calculate_timeout` (línea ~237-244)
```python
        if command == "remove_background":
            model_name = payload.get("model_name", "")
            if payload.get("enable_clothing_protection", False) and model_name not in ("u2net_human_seg", "isnet-anime"):
                base_timeout += 15.0
            if model_name in ("ensemble", "birefnet-general"):
                base_timeout += 20.0
```

---

### 3.4. `app/core/sam_manager.py`

#### A. Reemplazar `_init_sessions` y añadir `unload_sessions` + `predict_mask_from_yolo_subject`
Sincroniza MobileSAM con `gpu_manager.py` y añade predicción de alta resolución combinando el Polo de Inaccesibilidad (`distanceTransform`) y el Bounding Box de YOLO11-seg:

```python
    def _init_sessions(self, enc_path: Path, dec_path: Path):
        """Inicializa las sesiones de ONNX Runtime respetando el estado de GPUWorkerManager."""
        import onnxruntime as ort
        from app.core.gpu_manager import get_gpu_manager

        gpu_mgr = get_gpu_manager()
        if gpu_mgr.is_gpu_enabled() and not gpu_mgr.has_orphaned_worker():
            providers = get_inference_providers()
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = get_optimized_session_options() or ort.SessionOptions()

        logger.info(f"Cargando MobileSAM Encoder desde {enc_path.name} (providers={providers})...")
        try:
            self.encoder_session = ort.InferenceSession(
                str(enc_path), sess_options, providers=providers
            )
        except Exception as e:
            logger.warning(f"Fallo al cargar Encoder con GPU, usando CPU: {e}")
            self.encoder_session = ort.InferenceSession(
                str(enc_path), sess_options, providers=["CPUExecutionProvider"]
            )

        logger.info(f"Cargando MobileSAM Decoder desde {dec_path.name}...")
        try:
            self.decoder_session = ort.InferenceSession(
                str(dec_path), sess_options, providers=providers
            )
        except Exception as e:
            logger.warning(f"Fallo al cargar Decoder con GPU, usando CPU: {e}")
            self.decoder_session = ort.InferenceSession(
                str(dec_path), sess_options, providers=["CPUExecutionProvider"]
            )

    def unload_sessions(self):
        """Libera las sesiones ONNX de MobileSAM de la VRAM cuando se cambia el modo de dispositivo."""
        import gc
        with self._lock:
            self.encoder_session = None
            self.decoder_session = None
            self.cached_embedding = None
            self.current_image_id = None
            self.orig_size = (0, 0)
            self.scaled_size = (0, 0)
        gc.collect()

    def predict_mask_from_yolo_subject(
        self,
        bbox: Tuple[int, int, int, int],
        coarse_mask: np.ndarray,
        orig_img_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Refina un sujeto detectado por YOLO11-seg usando MobileSAM.
        Combina el punto interior más profundo de la máscara de YOLO (label 1)
        con el Bounding Box del sujeto (labels 2 y 3) para lograr precisión de alta resolución.
        """
        with self._lock:
            if self.cached_embedding is None or self.decoder_session is None:
                return None

            orig_w, orig_h = orig_img_size or self.orig_size
            scale = self.scale_factor or (1024.0 / max(orig_w, orig_h))

            # 1. Calcular el polo de inaccesibilidad (punto más profundo garantizado dentro del sujeto)
            mask_u8 = (coarse_mask > 0).astype(np.uint8)
            if np.any(mask_u8):
                dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                _, _, _, max_loc = cv2.minMaxLoc(dist_map)
                cx, cy = max_loc
            else:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2

            x1, y1, x2, y2 = bbox
            point_coords = np.array(
                [[
                    [float(cx) * scale, float(cy) * scale],
                    [float(x1) * scale, float(y1) * scale],
                    [float(x2) * scale, float(y2) * scale],
                ]],
                dtype=np.float32,
            )
            point_labels = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)
            orig_im_size = np.array([1024.0, 1024.0], dtype=np.float32)

            decoder_inputs = {}
            for inp in self.decoder_session.get_inputs():
                name = inp.name
                if "image_embedding" in name:
                    decoder_inputs[name] = self.cached_embedding
                elif "point_coord" in name:
                    decoder_inputs[name] = point_coords
                elif "point_label" in name:
                    decoder_inputs[name] = point_labels
                elif "mask_input" in name and "has" not in name:
                    decoder_inputs[name] = mask_input
                elif "has_mask_input" in name:
                    decoder_inputs[name] = has_mask_input
                elif "orig_im_size" in name:
                    decoder_inputs[name] = orig_im_size

            try:
                outputs = self.decoder_session.run(None, decoder_inputs)
                raw_mask = outputs[0]
                if raw_mask.ndim == 4:
                    mask_2d = raw_mask[0, 0]
                elif raw_mask.ndim == 3:
                    mask_2d = raw_mask[0]
                else:
                    mask_2d = np.squeeze(raw_mask)

                new_w, new_h = self.scaled_size
                if new_w > 0 and new_h > 0 and mask_2d.shape[0] >= new_h and mask_2d.shape[1] >= new_w:
                    crop_mask = mask_2d[:new_h, :new_w]
                else:
                    crop_mask = mask_2d

                prob_resized = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                prob_norm = 1.0 / (1.0 + np.exp(-np.clip(prob_resized, -15.0, 15.0)))
                return prob_norm
            except Exception as e:
                logger.warning(f"Fallo en predict_mask_from_yolo_subject con BBox, intentando punto simple: {e}")

        bool_mask = self.predict_mask_at_point(cx, cy, orig_img_size=(orig_w, orig_h))
        return bool_mask.astype(np.float32) if bool_mask is not None else None
```

---

### 3.5. `app/ui/main_window.py`

#### A. Liberar sesiones de MobileSAM al cambiar modo GPU/CPU (`_on_device_mode_changed`, línea ~1129)
```python
    def _on_device_mode_changed(self, is_gpu: bool):
        """Aplica el cambio de modo de dispositivo solicitado por el usuario y libera VRAM residual."""
        from app.core.gpu_manager import get_gpu_manager
        from app.core.sam_manager import get_sam_manager

        gpu_mgr = get_gpu_manager()
        success = gpu_mgr.set_gpu_enabled(is_gpu)

        # Liberar sesiones de MobileSAM para que respeten el nuevo proveedor (GPU o CPU)
        get_sam_manager().unload_sessions()

        self.update_device_badge()

        if is_gpu and success:
            from app.config import get_gpu_name
            gpu_name = get_gpu_name() or "DirectML"
            clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
            self._set_status_raw(f"Aceleración por GPU activada ({clean_name}).")
        elif not is_gpu:
            self._set_status_raw("Modo CPU activado (GPU liberada de VRAM).")
        else:
            self._set_status_raw("No se pudo activar la GPU. Se mantiene modo CPU.")
```

#### B. Reemplazar `_on_yolo_finished`, `perform_isolate_subject` y `perform_select_subject`
```python
    def _on_yolo_finished(self, subjects: list):
        self.sidebar.set_detecting_subjects_state(False)
        self.sidebar.set_detected_subjects(subjects)
        if subjects:
            self._set_status_raw(f"YOLO11: {len(subjects)} sujeto(s) detectado(s). Selecciona un chip para aislar.")
            # Precalcular embedding de MobileSAM en segundo plano para aislamiento HD instantáneo
            self._ensure_sam_embedding_async()
        else:
            self._set_status("detect_none_found")

    def _on_yolo_failed(self, err_msg: str):
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status_raw(f"Error YOLO: {err_msg}")

    def _refine_yolo_subject_mask(self, subject) -> Optional[np.ndarray]:
        """
        Convierte la máscara gruesa de YOLO11-seg (160x160) en una máscara alfa de alta definición (uint8 0-255)
        combinando MobileSAM (si está disponible) y Guided Filter perimetral guiado por los píxeles RGB reales.
        """
        if not self.image_state.has_image or subject is None:
            return None

        pil_img = self.image_state.current_image
        orig_w, orig_h = pil_img.size
        coarse_bool = subject.mask
        if coarse_bool.shape != (orig_h, orig_w):
            return None

        base_prob = coarse_bool.astype(np.float32)

        # 1. Si MobileSAM tiene el embedding listo, obtener predicción semántica con BBox + Punto Interior
        from app.core.sam_manager import get_sam_manager
        sam_mgr = get_sam_manager()
        if sam_mgr.has_embedding():
            sam_prob = sam_mgr.predict_mask_from_yolo_subject(
                bbox=subject.bbox,
                coarse_mask=coarse_bool,
                orig_img_size=(orig_w, orig_h),
            )
            if sam_prob is not None:
                sam_bin = sam_prob > 0.5
                inter = np.logical_and(sam_bin, coarse_bool).sum()
                union = np.logical_or(sam_bin, coarse_bool).sum()
                if union > 0 and (inter / union) >= 0.40:
                    base_prob = sam_prob

        # 2. Aplicar Guided Filter de alta resolución en el ROI del Bounding Box para ajustar el borde al píxel
        try:
            import cv2
            from app.core.bg_remover import apply_guided_filter

            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = [int(v) for v in subject.bbox]
            pad = 24
            rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
            rx2, ry2 = min(orig_w, x2 + pad), min(orig_h, y2 + pad)

            if rx2 > rx1 and ry2 > ry1:
                crop_guide = bgr[ry1:ry2, rx1:rx2]
                crop_prob = base_prob[ry1:ry2, rx1:rx2]
                crop_refined = apply_guided_filter(
                    guide_bgr=crop_guide,
                    prob_mask=crop_prob,
                    radius=6,
                    eps=1e-4,
                    band_dilation=12,
                )
                base_prob = base_prob.copy()
                base_prob[ry1:ry2, rx1:rx2] = crop_refined
        except Exception as e:
            logger.warning(f"Guided filter en aislamiento YOLO omitido: {e}")

        return np.clip(np.round(base_prob * 255.0), 0, 255).astype(np.uint8)

    def perform_isolate_subject(self, subject):
        """Aísla el sujeto detectado con refinamiento de bordes HD, dejando transparente el resto."""
        if not self.image_state.has_image or subject is None:
            return

        high_res_mask = self._refine_yolo_subject_mask(subject)
        if high_res_mask is None:
            return

        curr = self.image_state.current_image.convert("RGBA")
        curr_arr = np.array(curr, dtype=np.uint8)

        # Conservar transparencias previas y aplicar gradiente subpíxel en bordes
        curr_arr[..., 3] = np.minimum(curr_arr[..., 3], high_res_mask)
        isolated_img = Image.fromarray(curr_arr, "RGBA")
        self.push_new_image_state(isolated_img, f"Aislar {subject.class_name.capitalize()}")

    def perform_select_subject(self, subject):
        """Carga la máscara refinada del sujeto detectado en la selección de hormigas marchantes."""
        if not self.image_state.has_image or subject is None:
            return

        high_res_mask = self._refine_yolo_subject_mask(subject)
        if high_res_mask is None:
            high_res_mask = (subject.mask.astype(np.uint8) * 255)

        self.canvas._set_selection(high_res_mask)
        self._set_status_raw(f"Seleccionado: {subject.class_name.capitalize()}")
```

---

### 3.6. `app/core/workers.py`

#### A. Cerrar descriptores de imagen en `BatchWorkerThread.run` (líneas ~84-101)
```python
                try:
                    with Image.open(f) as raw_img:
                        img = raw_img.convert("RGBA")

                    res = remover.remove_background(
                        img,
                        model_name=self.model_name,
                        enable_clothing_protection=self.enable_clothing_protection,
                    )

                    out_file = output_path / f"{f.stem}_bgrm.png"
                    res.save(out_file)
                    res.close()
                    img.close()
                    success_count += 1
                except Exception as e:
                    print(f"Error procesando {f.name}: {e}")
```
