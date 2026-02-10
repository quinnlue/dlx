import unittest

import numpy as np

from dlx.nn.tensor import Tensor
from dlx.utils.backend import Device, cpu, cuda, _has_cupy, xp


class TestTensorDeviceResolution(unittest.TestCase):
    """Tensor should resolve its device from explicit arg, data, or the default."""

    def test_default_device_is_assigned(self):
        t = Tensor(xp.array([1.0, 2.0]))
        self.assertIsInstance(t.device, Device)

    def test_explicit_device_object(self):
        d = Device("cpu")
        t = Tensor(xp.array([1.0]), device=d)
        self.assertEqual(t.device, d)

    def test_explicit_device_string(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        self.assertEqual(t.device, "cpu")

    def test_device_inherited_from_tensor_data(self):
        a = Tensor(xp.array([1.0]), device="cpu")
        b = Tensor(a)  # should inherit device from a
        self.assertEqual(b.device, a.device)


class TestTensorIsCuda(unittest.TestCase):
    def test_cpu_tensor_is_not_cuda(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        self.assertFalse(t.is_cuda)

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_tensor_is_cuda(self):
        t = Tensor(xp.array([1.0]), device="cuda")
        self.assertTrue(t.is_cuda)


class TestTensorTo(unittest.TestCase):
    def test_to_same_device_returns_same_object(self):
        t = Tensor(xp.array([1.0, 2.0, 3.0]), device="cpu")
        t2 = t.to("cpu")
        self.assertIs(t, t2)

    def test_to_returns_correct_data(self):
        data = xp.array([1.0, 2.0, 3.0])
        t = Tensor(data, device="cpu")
        t2 = t.to("cpu")
        self.assertTrue(xp.allclose(t.data, t2.data))

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cpu_to_cuda(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]), device="cpu")
        t_cuda = t.to("cuda")
        self.assertTrue(t_cuda.is_cuda)
        self.assertTrue(np.allclose(t.data, t_cuda.data.get()))

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_to_cpu(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]), device="cuda")
        t_cpu = t.to("cpu")
        self.assertFalse(t_cpu.is_cuda)
        self.assertTrue(np.allclose(t.data.get(), t_cpu.data))

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_to_preserves_grad(self):
        t = Tensor(np.array([1.0, 2.0]), device="cpu", requires_grad=True)
        t.grad = Tensor(np.array([0.5, 0.5]), device="cpu", requires_grad=False)
        t_cuda = t.to("cuda")
        self.assertTrue(t_cuda.is_cuda)
        self.assertIsNotNone(t_cuda.grad)
        self.assertTrue(t_cuda.grad.is_cuda)


class TestTensorCpuCudaShortcuts(unittest.TestCase):
    def test_cpu_shortcut(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        t2 = t.cpu()
        self.assertEqual(t2.device, "cpu")

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_shortcut(self):
        t = Tensor(np.array([1.0]), device="cpu")
        t2 = t.cuda()
        self.assertTrue(t2.is_cuda)

    @unittest.skipIf(_has_cupy, "cupy IS available – skip cuda-unavailable test")
    def test_cuda_shortcut_without_cupy_raises(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        with self.assertRaises(RuntimeError):
            t.cuda()


class TestCheckDevice(unittest.TestCase):
    def test_same_device_no_error(self):
        a = Tensor(xp.array([1.0]), device="cpu")
        b = Tensor(xp.array([2.0]), device="cpu")
        # Should not raise
        a._check_device(b)

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_different_device_raises(self):
        a = Tensor(np.array([1.0]), device="cpu")
        b = Tensor(np.array([2.0]), device="cuda")
        with self.assertRaises(RuntimeError):
            a._check_device(b)

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_add_different_devices_raises(self):
        a = Tensor(np.array([1.0]), device="cpu")
        b = Tensor(np.array([2.0]), device="cuda")
        with self.assertRaises(RuntimeError):
            _ = a + b


class TestTensorDeviceInfo(unittest.TestCase):
    def test_repr_contains_device(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        self.assertIn("device", repr(t))

    def test_str_contains_device(self):
        t = Tensor(xp.array([1.0]), device="cpu")
        self.assertIn("device", str(t))


if __name__ == "__main__":
    unittest.main()
