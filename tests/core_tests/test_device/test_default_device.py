import unittest

from dlx.nn.tensor import Tensor
from dlx.utils.backend import (
    Device,
    cpu,
    _has_cupy,
    get_default_device,
    set_default_device,
    xp,
)


class TestGetDefaultDevice(unittest.TestCase):
    def test_returns_device_instance(self):
        d = get_default_device()
        self.assertIsInstance(d, Device)

    def test_default_device_is_cpu_when_no_cuda(self):
        if not _has_cupy:
            self.assertEqual(get_default_device(), "cpu")


class TestSetDefaultDevice(unittest.TestCase):
    def setUp(self):
        self._original = get_default_device()

    def tearDown(self):
        set_default_device(self._original)

    def test_set_with_string(self):
        set_default_device("cpu")
        self.assertEqual(get_default_device(), "cpu")

    def test_set_with_device_object(self):
        d = Device("cpu")
        set_default_device(d)
        self.assertEqual(get_default_device(), d)

    def test_tensor_uses_updated_default(self):
        set_default_device("cpu")
        t = Tensor(xp.array([1.0]))
        self.assertEqual(t.device, "cpu")

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_set_to_cuda(self):
        set_default_device("cuda")
        self.assertEqual(get_default_device(), "cuda")

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_tensor_on_cuda_after_set(self):
        set_default_device("cuda")
        t = Tensor([1.0, 2.0])
        self.assertTrue(t.is_cuda)

    def test_set_invalid_device_raises(self):
        with self.assertRaises(ValueError):
            set_default_device("tpu")


if __name__ == "__main__":
    unittest.main()
