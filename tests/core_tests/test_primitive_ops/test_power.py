import os
import sys
import unittest

import torch

from dlx.nn.tensor import Tensor
from dlx.utils.backend import xp


class TestPower(unittest.TestCase):
    def test_power_tensor_scalar(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        b_my = a_my.pow(2)
        b_pt = a_pt.pow(2)

        c_my = b_my.mean()
        c_pt = b_pt.mean()

        c_my.backward()
        c_pt.backward()

        self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))


    def test_power_tensor_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        b_my = Tensor(xp.array([4.0, 5.0, 6.0]), requires_grad=True)


        with self.assertRaises(NotImplementedError):
            a_my.pow(b_my)

if __name__ == "__main__":
    unittest.main()
        