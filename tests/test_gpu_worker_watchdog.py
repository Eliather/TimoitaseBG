import atexit
import unittest
from unittest.mock import patch, MagicMock
from app.core.gpu_manager import GPUWorkerManager

class TestGPUWorkerWatchdog(unittest.TestCase):
    def setUp(self):
        self.manager = GPUWorkerManager(register_atexit=False)

    def tearDown(self):
        try:
            atexit.unregister(self.manager.shutdown)
        except Exception:
            pass
        self.manager._process = None
        self.manager._close_queues()
        from app.config import set_gpu_acceleration
        set_gpu_acceleration(True)

    @patch('app.config.is_gpu_available', return_value=True)
    def test_singleton(self, mock_gpu):
        from app.core.gpu_manager import get_gpu_manager
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        self.assertIs(manager1, manager2)

    @patch('app.core.gpu_manager.multiprocessing.Process')
    def test_ensure_worker(self, mock_process):
        mock_proc = MagicMock()
        mock_proc.pid = 424242
        mock_process.return_value = mock_proc
        self.manager._ensure_worker()
        self.assertIsNotNone(self.manager._process)

    def test_kill_worker(self):
        alive = [True]
        def mock_terminate():
            alive[0] = False
        proc = MagicMock()
        proc.terminate.side_effect = mock_terminate
        proc.is_alive.side_effect = lambda: alive[0]
        self.manager._process = proc

        killed = self.manager.kill_worker()
        self.assertTrue(killed)
        proc.terminate.assert_called_once()
        self.assertIsNone(self.manager._process)

    @patch('platform.system', return_value='Linux')
    @patch('os.killpg', create=True)
    @patch('os.getpgid', return_value=1234, create=True)
    @patch('os.getpgrp', return_value=9999, create=True)
    @patch('os.kill', create=True)
    def test_kill_worker_linux_sigkill(self, mock_os_kill, mock_getpgrp, mock_getpgid, mock_killpg, mock_platform):
        alive = [True]
        def mock_kill_action(*args, **kwargs):
            alive[0] = False
        mock_killpg.side_effect = mock_kill_action
        mock_os_kill.side_effect = mock_kill_action

        proc = MagicMock()
        proc.pid = 1234
        proc.is_alive.side_effect = lambda: alive[0]
        self.manager._process = proc

        killed = self.manager.kill_worker()
        self.assertTrue(killed)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        mock_killpg.assert_called_once()
        self.assertIsNone(self.manager._process)

    def test_shutdown(self):
        self.manager._process = MagicMock()
        self.manager._process.is_alive.return_value = False
        with patch.object(self.manager, '_cleanup_process') as mock_cleanup:
            self.manager.shutdown()
            mock_cleanup.assert_called_once()

    @patch('app.config.is_gpu_available', return_value=True)
    def test_is_gpu_enabled_default(self, mock_gpu):
        mgr = GPUWorkerManager(register_atexit=False)
        self.assertTrue(mgr.is_gpu_enabled())

    @patch('app.config.is_gpu_available', return_value=True)
    def test_set_gpu_enabled(self, mock_gpu):
        mgr = GPUWorkerManager(register_atexit=False)
        res = mgr.set_gpu_enabled(False)
        self.assertFalse(res)
        self.assertFalse(mgr.is_gpu_enabled())
        res_enable = mgr.set_gpu_enabled(True)
        self.assertTrue(res_enable)
        self.assertTrue(mgr.is_gpu_enabled())

    def test_run_inference_cpu_fallback(self):
        fallback = MagicMock(return_value="cpu_result")
        self.manager.set_gpu_enabled(False)
        res = self.manager.run_inference("test", {}, fallback)
        self.assertEqual(res, "cpu_result")
        fallback.assert_called_once()

    def test_get_last_status_message(self):
        self.manager._last_status_message = "Test status"
        self.assertEqual(self.manager.get_last_status_message(), "Test status")

    def test_has_orphaned_worker(self):
        self.assertFalse(self.manager.has_orphaned_worker())

    @patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    def test_worker_providers_cuda_linux(self, mock_ort):
        from app.core.gpu_worker import get_worker_gpu_providers
        providers = get_worker_gpu_providers()
        prov_names = [p[0] if isinstance(p, tuple) else p for p in providers]
        self.assertIn('CUDAExecutionProvider', prov_names)
        self.assertIn('CPUExecutionProvider', prov_names)
        cuda_entry = next(p for p in providers if isinstance(p, tuple) and p[0] == 'CUDAExecutionProvider')
        self.assertEqual(cuda_entry[1].get('arena_extend_strategy'), 'kSameAsRequested')
        self.assertEqual(cuda_entry[1].get('cudnn_conv_algo_search'), 'HEURISTIC')

    @patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='')
    def test_detect_device_cuda(self, mock_gpu_name, mock_ort):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "NVIDIA CUDA (GPU)")

    @patch('onnxruntime.get_available_providers', return_value=['ROCMExecutionProvider', 'CPUExecutionProvider'])
    def test_worker_providers_rocm_amd(self, mock_ort):
        from app.core.gpu_worker import get_worker_gpu_providers
        providers = get_worker_gpu_providers()
        prov_names = [p[0] if isinstance(p, tuple) else p for p in providers]
        self.assertIn('ROCMExecutionProvider', prov_names)
        self.assertIn('CPUExecutionProvider', prov_names)

    @patch('onnxruntime.get_available_providers', return_value=['ROCMExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='')
    def test_detect_device_rocm(self, mock_gpu_name, mock_ort):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "AMD ROCm (GPU)")

    @patch('app.config.get_gpu_name', return_value='AMD Radeon(TM) Graphics')
    def test_is_integrated_gpu_and_tiling(self, mock_gpu_name):
        from app.config import is_integrated_gpu
        self.assertTrue(is_integrated_gpu())

    @patch('onnxruntime.get_available_providers', return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    def test_worker_providers_coreml_macos(self, mock_ort):
        from app.core.gpu_worker import get_worker_gpu_providers
        providers = get_worker_gpu_providers()
        self.assertIn('CoreMLExecutionProvider', providers)
        self.assertIn('CPUExecutionProvider', providers)

    @patch('sys.platform', 'darwin')
    @patch('platform.machine', return_value='arm64')
    @patch('onnxruntime.get_available_providers', return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='Apple M5 Max')
    def test_detect_device_apple_silicon_arm64(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Apple Silicon CoreML (Apple M5 Max - GPU / Neural Engine)")

    @patch('sys.platform', 'darwin')
    @patch('platform.machine', return_value='arm64')
    @patch('onnxruntime.get_available_providers', return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='')
    def test_detect_device_apple_silicon_no_name(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Apple Silicon CoreML (GPU / Neural Engine)")

    @patch('sys.platform', 'darwin')
    @patch('platform.machine', return_value='x86_64')
    @patch('onnxruntime.get_available_providers', return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='AMD Radeon Pro 5500M')
    def test_detect_device_intel_mac_coreml(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "macOS CoreML (GPU: AMD Radeon Pro 5500M)")

    @patch('sys.platform', 'darwin')
    @patch('platform.machine', return_value='x86_64')
    @patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider'])
    def test_detect_device_intel_mac_cpu_fallback(self, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Intel Mac (CPU)")

    @patch('sys.platform', 'darwin')
    @patch('platform.machine', return_value='arm64')
    def test_is_integrated_gpu_apple_silicon_uma(self, mock_machine):
        from app.config import is_integrated_gpu
        self.assertTrue(is_integrated_gpu())

    @patch('sys.platform', 'darwin')
    @patch('subprocess.check_output', return_value=b"Apple M3 Pro\n")
    def test_get_gpu_name_macos_sysctl(self, mock_subprocess):
        import app.config as config
        config._cached_gpu_name = None
        name = config.get_gpu_name()
        self.assertEqual(name, "Apple M3 Pro")
        config._cached_gpu_name = None

    @patch('sys.platform', 'win32')
    @patch('platform.machine', return_value='ARM64')
    @patch('onnxruntime.get_available_providers', return_value=['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'])
    def test_windows_arm64_worker_providers_qnn_priority(self, mock_ort, mock_machine):
        from app.core.gpu_worker import get_worker_gpu_providers
        providers = get_worker_gpu_providers()
        prov_names = [p[0] if isinstance(p, tuple) else p for p in providers]
        self.assertIn('QNNExecutionProvider', prov_names)
        self.assertIn('DmlExecutionProvider', prov_names)
        self.assertIn('CPUExecutionProvider', prov_names)
        # QNNExecutionProvider debe preceder a DmlExecutionProvider
        self.assertLess(prov_names.index('QNNExecutionProvider'), prov_names.index('DmlExecutionProvider'))

    @patch('sys.platform', 'win32')
    @patch('platform.machine', return_value='ARM64')
    @patch('onnxruntime.get_available_providers', return_value=['QNNExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='Snapdragon X Elite - X1E80100')
    def test_windows_arm64_detect_device_qnn(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Qualcomm QNN (NPU: Snapdragon X Elite - X1E80100)")

    @patch('sys.platform', 'win32')
    @patch('platform.machine', return_value='ARM64')
    @patch('onnxruntime.get_available_providers', return_value=['DmlExecutionProvider', 'CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='Qualcomm(R) Adreno(TM) GPU')
    def test_windows_arm64_detect_device_directml_adreno(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "DirectML ARM64 (GPU: Qualcomm(R) Adreno(TM) GPU)")

    @patch('platform.machine', return_value='ARM64')
    def test_windows_arm64_is_integrated_gpu(self, mock_machine):
        from app.config import is_integrated_gpu
        self.assertTrue(is_integrated_gpu())

    @patch('sys.platform', 'linux')
    @patch('platform.machine', return_value='aarch64')
    @patch('subprocess.check_output', side_effect=FileNotFoundError)
    def test_linux_arm64_get_gpu_name_device_tree(self, mock_subprocess, mock_machine):
        import app.config as config
        config._cached_gpu_name = None
        fake_bytes = b"NVIDIA Jetson Orin Nano Developer Kit\x00"
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_bytes', return_value=fake_bytes):
            name = config.get_gpu_name()
            self.assertEqual(name, "NVIDIA Jetson Orin Nano Developer Kit")
        config._cached_gpu_name = None

    @patch('sys.platform', 'linux')
    @patch('platform.machine', return_value='aarch64')
    @patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='Raspberry Pi 5 Model B Rev 1.0')
    def test_linux_arm64_detect_device_cpu_neon_with_soc(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Linux ARM64 NEON (Raspberry Pi 5 Model B Rev 1.0)")

    @patch('sys.platform', 'linux')
    @patch('platform.machine', return_value='aarch64')
    @patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider'])
    @patch('app.config.get_gpu_name', return_value='')
    def test_linux_arm64_detect_device_cpu_neon_clean_fallback(self, mock_gpu_name, mock_ort, mock_machine):
        from app.config import detect_device, set_gpu_acceleration
        set_gpu_acceleration(True)
        device = detect_device()
        self.assertEqual(device, "Linux ARM64 NEON (CPU)")

    @patch('platform.machine', return_value='aarch64')
    def test_linux_arm64_is_integrated_gpu_and_restorer_tiling(self, mock_machine):
        from app.config import is_integrated_gpu
        self.assertTrue(is_integrated_gpu())

    @patch('platform.machine', return_value='aarch64')
    def test_arm_optimized_session_options_threads(self, mock_machine):
        from app.config import get_optimized_session_options
        sess_opts = get_optimized_session_options()
        self.assertIsNotNone(sess_opts)
        self.assertLessEqual(sess_opts.intra_op_num_threads, 4)

    @patch('platform.system', return_value='Linux')
    @patch('os.killpg', create=True)
    @patch('os.getpgid', return_value=5000, create=True)
    @patch('os.getpgrp', return_value=5000, create=True)
    @patch('os.kill', create=True)
    def test_kill_worker_linux_same_pgrp_does_not_call_killpg(
        self, mock_os_kill, mock_getpgrp, mock_getpgid, mock_killpg, mock_platform
    ):
        alive = [True]
        mock_os_kill.side_effect = lambda *a, **kw: alive.__setitem__(0, False)

        proc = MagicMock()
        proc.pid = 5000
        proc.is_alive.side_effect = lambda: alive[0]
        self.manager._process = proc

        killed = self.manager.kill_worker()
        self.assertTrue(killed)
        mock_killpg.assert_not_called()
        mock_os_kill.assert_called_once()

    def test_kill_worker_closes_and_cancels_queues(self):
        proc = MagicMock()
        proc.is_alive.return_value = False
        req_q = MagicMock()
        resp_q = MagicMock()
        self.manager._process = proc
        self.manager._request_queue = req_q
        self.manager._response_queue = resp_q

        self.manager.kill_worker()
        req_q.close.assert_called_once()
        req_q.cancel_join_thread.assert_called_once()
        resp_q.close.assert_called_once()
        resp_q.cancel_join_thread.assert_called_once()
        self.assertIsNone(self.manager._request_queue)
        self.assertIsNone(self.manager._response_queue)

    @patch('app.config.is_gpu_available', return_value=True)
    def test_run_inference_validation_error_does_not_fallback_to_cpu(self, mock_gpu):
        import queue as std_queue
        from app.config import set_gpu_acceleration

        set_gpu_acceleration(True)
        mgr = GPUWorkerManager(register_atexit=False)
        req_q = std_queue.Queue()
        resp_q = std_queue.Queue()
        proc = MagicMock()
        proc.is_alive.return_value = True

        def fake_ensure():
            mgr._process = proc
            mgr._request_queue = req_q
            mgr._response_queue = resp_q

        original_put = req_q.put

        def intercept_put(item):
            original_put(item)
            resp_q.put({
                "task_id": item["task_id"],
                "status": "error",
                "error_type": "validation",
                "error": "MAX_ALLOWED_TILES exceeded",
            })

        req_q.put = intercept_put
        mgr._ensure_worker = fake_ensure
        fallback = MagicMock(return_value="cpu_should_not_run")
        notifications = []
        mgr.set_status_callback(notifications.append)

        with self.assertRaises(ValueError):
            mgr.run_inference("restore_image", {}, fallback, timeout_override=2.0)

        fallback.assert_not_called()
        self.assertEqual(notifications, [])

    @patch('app.config.is_gpu_available', return_value=True)
    def test_run_inference_releases_lock_during_wait_and_supports_cancel(self, mock_gpu):
        import queue as std_queue
        import threading
        from app.config import set_gpu_acceleration

        set_gpu_acceleration(True)
        mgr = GPUWorkerManager(register_atexit=False)
        req_q = std_queue.Queue()
        resp_q = std_queue.Queue()
        proc = MagicMock()
        proc.is_alive.return_value = True
        proc.terminate.side_effect = lambda: setattr(proc.is_alive, "return_value", False)

        def fake_ensure():
            mgr._process = proc
            mgr._request_queue = req_q
            mgr._response_queue = resp_q

        mgr._ensure_worker = fake_ensure
        cancel_flag = {"cancelled": False}
        lock_acquired_from_other_thread = {"ok": False}

        def ui_thread_check():
            import time
            time.sleep(0.15)
            acquired = mgr._lock.acquire(timeout=0.5)
            lock_acquired_from_other_thread["ok"] = acquired
            if acquired:
                mgr._lock.release()
            cancel_flag["cancelled"] = True

        t = threading.Thread(target=ui_thread_check)
        t.start()

        with self.assertRaises(InterruptedError):
            mgr.run_inference(
                "remove_background",
                {},
                lambda: "cpu",
                timeout_override=5.0,
                cancel_check=lambda: cancel_flag["cancelled"],
            )

        t.join(timeout=2.0)
        self.assertTrue(lock_acquired_from_other_thread["ok"])

    def test_gpu_worker_restore_reraises_value_error_and_resets_on_gpu_error(self):
        from PIL import Image
        from app.core.gpu_worker import PersistentGPUWorker

        worker = PersistentGPUWorker(preferred_device_id=0)
        fake_restorer = MagicMock()
        fake_restorer._restore_image_impl.side_effect = ValueError("MAX_ALLOWED_TILES")
        worker._restorer_instance = fake_restorer

        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        with self.assertRaises(ValueError):
            worker.restore_image(img)

        # Ahora simular fallo GPU runtime: debe resetear _restorer_instance a None y usar CPU
        fake_restorer._restore_image_impl.side_effect = RuntimeError("DML device removed")
        worker._restorer_instance = fake_restorer
        with patch("app.core.restorer.ImageRestorer") as mock_restorer_cls:
            mock_cpu_inst = MagicMock()
            mock_cpu_inst._restore_image_impl.return_value = img
            mock_restorer_cls.return_value = mock_cpu_inst
            out = worker.restore_image(img)
            self.assertIs(out, img)
            self.assertIsNone(worker._restorer_instance)
