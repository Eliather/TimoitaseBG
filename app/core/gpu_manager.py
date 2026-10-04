"""
TimoitaseBG - Administrador del proceso GPU Worker con Watchdog y Fallback a CPU.
Gestiona el ciclo de vida del subproceso persistente, monitoriza los tiempos de respuesta
y si la GPU no responde o se cuelga (TDR/deadlock), la elimina y conmuta de forma
transparente a CPUExecutionProvider.
"""
import atexit
import logging
import multiprocessing
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

import app.config as config
from app.core.gpu_worker import gpu_worker_process_loop

logger = logging.getLogger("TimoitaseBG.GPUManager")


class GPUWorkerManager:
    """
    Cliente y supervisor del proceso persistente GPU Worker.
    """

    def __init__(self):
        self._process: Optional[multiprocessing.Process] = None
        self._request_queue: Optional[multiprocessing.Queue] = None
        self._response_queue: Optional[multiprocessing.Queue] = None
        self._lock = threading.RLock()
        self._shutdown_thread: Optional[threading.Thread] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self._last_status_message: str = ""
        self._has_orphaned_worker: bool = False

        # Asegurar parada limpia al cerrar el programa
        atexit.register(self.shutdown)

    def set_status_callback(self, cb: Optional[Callable[[str], None]]):
        """Registra un callback para emitir notificaciones no invasivas a la interfaz gráfica."""
        self._status_callback = cb

    def get_last_status_message(self) -> str:
        """Retorna el último mensaje de estado emitido (ej. 'GPU no respondió, se procesó con CPU')."""
        return self._last_status_message

    def has_orphaned_worker(self) -> bool:
        """Retorna True si un proceso worker quedó huérfano y la GPU está bloqueada por el resto de la sesión."""
        return self._has_orphaned_worker

    def is_gpu_enabled(self) -> bool:
        """Indica si la inferencia en GPU está autorizada y activa."""
        if self._has_orphaned_worker:
            return False
        return config.ENABLE_GPU_ACCELERATION and (not config.FORCE_CPU_ONLY) and config.is_gpu_available()

    def set_gpu_enabled(self, enabled: bool) -> bool:
        """
        Habilita o deshabilita la inferencia en GPU de forma segura e inmediata.
        Si se desactiva, apaga el proceso worker y libera la VRAM de forma asíncrona en segundo plano
        sin congelar ni demorar la interfaz de usuario (0 ms de bloqueo).
        Si la GPU está bloqueada por cuelgue previo, no permite reactivación y retorna False.
        """
        if enabled and self._has_orphaned_worker:
            logger.warning("GPU bloqueada permanentemente por worker huérfano; no se puede habilitar.")
            return False

        if enabled and not config.is_gpu_available():
            logger.warning("No hay GPU compatible detectada; no se puede habilitar aceleración GPU.")
            return False

        config.set_gpu_acceleration(enabled)

        if not enabled:
            logger.info("Aceleración GPU desactivada por usuario. Liberando worker y VRAM en segundo plano...")
            self.shutdown(blocking=False)
        else:
            logger.info("Aceleración GPU activada por usuario.")

        return self.is_gpu_enabled()

    def _ensure_worker(self):
        """Verifica que el subproceso worker esté vivo; si no, lo inicia limpiamente."""
        if self._has_orphaned_worker:
            raise RuntimeError(
                "No se puede iniciar GPU Worker: el proceso previo quedó huérfano "
                "y la GPU está bloqueada para el resto de la sesión. Use CPU."
            )

        with self._lock:
            # Si hay una liberación en segundo plano en curso, esperar que termine antes de crear otro worker
            if self._shutdown_thread and self._shutdown_thread.is_alive():
                self._shutdown_thread.join(timeout=2.0)
                self._shutdown_thread = None

            if self._process is None or not self._process.is_alive():
                self._cleanup_process()
                preferred_device_id = config.get_best_dml_device_id()
                logger.info(f"Iniciando nuevo proceso persistente GPU Worker (DirectML device_id={preferred_device_id})...")
                self._request_queue = multiprocessing.Queue()
                self._response_queue = multiprocessing.Queue()
                self._process = multiprocessing.Process(
                    target=gpu_worker_process_loop,
                    args=(self._request_queue, self._response_queue, preferred_device_id),
                    daemon=True,
                )
                self._process.start()
                logger.info(f"GPU Worker iniciado exitosamente con PID {self._process.pid} (device_id={preferred_device_id}).")

    def _cleanup_process(self):
        """Limpia los recursos del proceso worker y sus colas."""
        if self._process is not None:
            if self._process.is_alive():
                self.kill_worker()
            else:
                self._process = None

        self._request_queue = None
        self._response_queue = None

    def kill_worker(self) -> bool:
        """
        Termina y mata forzadamente el proceso worker actual.
        Secuencia:
        1. terminate() -> join(timeout=1.5)
        2. kill() -> join(timeout=1.0)
        3. Capa extra en Windows: taskkill /F /T /PID -> join(timeout=1.0)
        4. Si sigue vivo: NO dar por limpio, registrar log CRÍTICO y notificar reinicio.
        Retorna True si el proceso murió, False si quedó huérfano.
        """
        if self._process is None:
            return True

        pid = self._process.pid
        logger.warning(f"Iniciando secuencia de terminación para GPU Worker (PID {pid})...")

        try:
            # 1. Intento ordenado con terminate()
            self._process.terminate()
            self._process.join(timeout=1.5)

            # 2. Si no respondió, forzar con kill()
            if self._process.is_alive():
                logger.warning(f"GPU Worker (PID {pid}) no respondió a terminate(). Aplicando kill()...")
                self._process.kill()
                self._process.join(timeout=1.0)

            # 3. Capa extra en Windows con taskkill /F /T (árbol de procesos completo)
            if self._process.is_alive():
                logger.warning(
                    f"GPU Worker (PID {pid}) continúa activo tras kill(). Ejecutando taskkill /F /T /PID {pid}..."
                )
                try:
                    import subprocess
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3.0,
                    )
                except Exception as tk_err:
                    logger.warning(f"Error ejecutando taskkill: {tk_err}")
                self._process.join(timeout=1.0)

        except Exception as e:
            logger.warning(f"Excepción durante terminación de GPU Worker: {e}")

        # 4. Verificación final de estado del proceso
        if self._process.is_alive():
            logger.critical(
                f"CRÍTICO: El subproceso GPU Worker (PID {pid}) NO pudo ser terminado por el sistema operativo. "
                f"El proceso ha quedado en estado huérfano (posible bloqueo irrecuperable en driver D3D12/GPU). "
                f"NO se marca el proceso como limpio."
            )
            self._has_orphaned_worker = True
            self._request_queue = None
            self._response_queue = None
            self._notify("GPU no respondió y el proceso no pudo cerrarse — se recomienda reiniciar la aplicación")
            return False
        else:
            logger.info(f"GPU Worker (PID {pid}) terminado y confirmado cerrado.")
            self._has_orphaned_worker = False
            self._process = None
            self._request_queue = None
            self._response_queue = None
            return True

    def shutdown(self, blocking: bool = True):
        """
        Detiene ordenadamente el worker al salir de la aplicación o al pasar a CPU.
        Si blocking=False, delega la liberación de VRAM a un hilo secundario
        sin congelar la interfaz de usuario.
        """
        if not blocking:
            with self._lock:
                if self._process and self._process.is_alive():
                    self._shutdown_thread = threading.Thread(
                        target=self._shutdown_worker_sync,
                        daemon=True,
                        name="GPUWorkerShutdownThread",
                    )
                    self._shutdown_thread.start()
            return

        self._shutdown_worker_sync()

    def _shutdown_worker_sync(self):
        """Ejecuta el protocolo de apagado del worker."""
        with self._lock:
            if self._process and self._process.is_alive() and self._request_queue:
                try:
                    self._request_queue.put({"command": "shutdown"})
                    self._process.join(timeout=1.5)
                except Exception:
                    pass
            self._cleanup_process()

    def _notify(self, message: str):
        """Emite una notificación informativa sin interrumpir el flujo."""
        self._last_status_message = message
        logger.info(f"[GPU Status]: {message}")
        if self._status_callback:
            try:
                self._status_callback(message)
            except Exception as e:
                logger.warning(f"Error invocando status_callback: {e}")

    def _calculate_timeout(self, command: str, payload: Dict[str, Any]) -> float:
        """Calcula el timeout máximo permitido para la tarea."""
        if "timeout_override" in payload:
            return float(payload["timeout_override"])

        # Base para inferencia individual
        base_timeout = 40.0

        if command == "remove_background":
            model_name = payload.get("model_name", "")
            if payload.get("enable_clothing_protection", False):
                base_timeout += 15.0
            if model_name in ("ensemble", "birefnet-general"):
                base_timeout += 20.0
        elif command == "restore_image":
            base_timeout = 45.0

        # Imagen de alta resolución
        img = payload.get("image")
        if img is not None and hasattr(img, "size"):
            w, h = img.size
            if max(w, h) > 2000:
                base_timeout += 20.0

        return base_timeout

    def run_inference(
        self,
        command: str,
        payload: Dict[str, Any],
        cpu_fallback_fn: Callable[[], Any],
        timeout_override: Optional[float] = None,
    ) -> Any:
        """
        Ejecuta la inferencia en el proceso persistente de GPU bajo supervisión de watchdog.
        Si la GPU está deshabilitada (ej. FORCE_CPU_ONLY=True), ejecuta directamente en CPU.
        Si la GPU no responde antes del timeout o falla, mata el subproceso, levanta uno limpio
        para la siguiente petición, reintenta automáticamente en CPU y notifica a la barra de estado.
        """
        if self._has_orphaned_worker:
            logger.warning(
                f"GPU Worker bloqueado en estado huérfano. "
                f"Forzando CPUExecutionProvider automáticamente para '{command}' por el resto de la sesión."
            )
            self._notify("CPU (GPU bloqueada — reinicia la app)")
            return cpu_fallback_fn()

        if not self.is_gpu_enabled():
            logger.info(f"GPU inactiva (FORCE_CPU_ONLY={config.FORCE_CPU_ONLY}). Ejecutando {command} en CPU.")
            return cpu_fallback_fn()

        with self._lock:
            self._ensure_worker()
            task_id = uuid.uuid4().hex
            timeout = timeout_override or self._calculate_timeout(command, payload)

            req = {
                "task_id": task_id,
                "command": command,
                "payload": payload,
            }

            try:
                self._request_queue.put(req)
            except Exception as e:
                logger.warning(f"Fallo al enviar petición a la cola GPU: {e}. Cayendo a CPU...")
                self.kill_worker()
                self._notify("GPU no respondió, se procesó con CPU")
                return cpu_fallback_fn()

            # Esperar respuesta con verificación periódica del estado del proceso
            t0 = time.time()
            received_resp = None

            while time.time() - t0 < timeout:
                try:
                    resp = self._response_queue.get(timeout=0.5)
                    if resp.get("task_id") == task_id:
                        received_resp = resp
                        break
                    else:
                        logger.warning(f"Mensaje descartado de tarea anterior: {resp.get('task_id')}")
                except queue.Empty:
                    # Verificar si el proceso worker murió repentinamente
                    if not self._process.is_alive():
                        logger.warning(
                            f"El proceso GPU Worker terminó inesperadamente (exitcode={self._process.exitcode})."
                        )
                        self.kill_worker()
                        self._notify("GPU no respondió, se procesó con CPU")
                        return cpu_fallback_fn()

            # Caso 1: Timeout agotado sin respuesta
            if received_resp is None:
                elapsed = time.time() - t0
                logger.warning(
                    f"Watchdog: GPU Worker no respondió tras {elapsed:.1f}s (timeout={timeout}s). "
                    f"Matando proceso colgado..."
                )
                killed_ok = self.kill_worker()
                if killed_ok:
                    self._notify("GPU no respondió, se procesó con CPU")
                logger.info(f"Reintentando {command} automáticamente con CPUExecutionProvider...")
                return cpu_fallback_fn()

            # Caso 2: Respuesta con error interno de GPU (ej. OOM o fallo de driver)
            if received_resp.get("status") != "ok":
                err_msg = received_resp.get("error", "Error desconocido")
                logger.warning(f"GPU Worker devolvió error: {err_msg}. Conmutando a CPU...")
                self._notify("GPU con error, se procesó con CPU")
                return cpu_fallback_fn()

            # Caso 3: Éxito total
            return received_resp.get("result")


# Instancia singleton del administrador
_gpu_manager_instance: Optional[GPUWorkerManager] = None
_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUWorkerManager:
    """Retorna la instancia global del administrador de GPU Worker."""
    global _gpu_manager_instance
    with _manager_lock:
        if _gpu_manager_instance is None:
            _gpu_manager_instance = GPUWorkerManager()
    return _gpu_manager_instance
