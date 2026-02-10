import unittest

from dlx.utils.backend import Device, cpu, cuda, _has_cupy


class TestDeviceInit(unittest.TestCase):
    def test_cpu_device_creation(self):
        d = Device("cpu")
        self.assertEqual(d.type, "cpu")

    def test_invalid_device_raises_value_error(self):
        with self.assertRaises(ValueError):
            Device("tpu")

    def test_invalid_device_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            Device("gpu")

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_device_creation(self):
        d = Device("cuda")
        self.assertEqual(d.type, "cuda")

    @unittest.skipIf(_has_cupy, "cupy IS available – skip cuda-unavailable test")
    def test_cuda_without_cupy_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            Device("cuda")


class TestDeviceXp(unittest.TestCase):
    def test_cpu_xp_is_numpy(self):
        import numpy as np
        d = Device("cpu")
        self.assertIs(d.xp, np)

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_xp_is_cupy(self):
        import cupy
        d = Device("cuda")
        self.assertIs(d.xp, cupy)


class TestDeviceEquality(unittest.TestCase):
    def test_equal_devices(self):
        a = Device("cpu")
        b = Device("cpu")
        self.assertEqual(a, b)

    def test_device_equals_string(self):
        d = Device("cpu")
        self.assertEqual(d, "cpu")
        self.assertNotEqual(d, "cuda")

    def test_device_not_equal_to_unrelated_type(self):
        d = Device("cpu")
        self.assertEqual(d.__eq__(42), NotImplemented)

    def test_hash_same_for_equal_devices(self):
        a = Device("cpu")
        b = Device("cpu")
        self.assertEqual(hash(a), hash(b))

    def test_devices_usable_as_dict_keys(self):
        a = Device("cpu")
        b = Device("cpu")
        mapping = {a: "first"}
        self.assertEqual(mapping[b], "first")


class TestDeviceRepr(unittest.TestCase):
    def test_cpu_repr(self):
        d = Device("cpu")
        self.assertEqual(repr(d), "device('cpu')")

    @unittest.skipUnless(_has_cupy, "cupy not available")
    def test_cuda_repr(self):
        d = Device("cuda")
        self.assertEqual(repr(d), "device('cuda')")


class TestSingletons(unittest.TestCase):
    def test_cpu_singleton_exists(self):
        self.assertIsNotNone(cpu)
        self.assertEqual(cpu.type, "cpu")

    def test_cuda_singleton_matches_availability(self):
        if _has_cupy:
            self.assertIsNotNone(cuda)
            self.assertEqual(cuda.type, "cuda")
        else:
            self.assertIsNone(cuda)


if __name__ == "__main__":
    unittest.main()
