import unittest
from unittest.mock import patch, MagicMock
from app.core.gpu_manager import GPUWorkerManager

class TestGPUWorkerWatchdog(unittest.TestCase):
    def setUp(self):
        self.manager = GPUWorkerManager()

    @patch('app.config.is_gpu_available', return_value=True)
    def test_singleton(self, mock_gpu):
        from app.core.gpu_manager import get_gpu_manager
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        self.assertIs(manager1, manager2)

    @patch('app.core.gpu_manager.multiprocessing.Process')
    def test_ensure_worker(self, mock_process):
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
    @patch('os.kill', create=True)
    def test_kill_worker_linux_sigkill(self, mock_os_kill, mock_platform):
        alive = [True]
        def mock_kill_action(*args, **kwargs):
            alive[0] = False
        mock_os_kill.side_effect = mock_kill_action

        proc = MagicMock()
        proc.pid = 1234
        proc.is_alive.side_effect = lambda: alive[0]
        self.manager._process = proc

        killed = self.manager.kill_worker()
        self.assertTrue(killed)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertIsNone(self.manager._process)

    def test_shutdown(self):
        self.manager._process = MagicMock()
        self.manager._process.is_alive.return_value = False
        with patch.object(self.manager, '_cleanup_process') as mock_cleanup:
            self.manager.shutdown()
            mock_cleanup.assert_called_once()

    @patch('app.config.is_gpu_available', return_value=True)
    def test_is_gpu_enabled_default(self, mock_gpu):
        mgr = GPUWorkerManager()
        self.assertTrue(mgr.is_gpu_enabled())

    @patch('app.config.is_gpu_available', return_value=True)
    def test_set_gpu_enabled(self, mock_gpu):
        mgr = GPUWorkerManager()
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
        self.assertIn('CUDAExecutionProvider', providers)
        self.assertIn('CPUExecutionProvider', providers)

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
