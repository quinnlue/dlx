import unittest

import torch

from dlx.utils.backend import xp
from dlx.nn.tensor import Tensor


class TestExponentials(unittest.TestCase):
    def test_log_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        b_my = a_my.log()
        b_pt = a_pt.log()

        c_my = b_my.mean()
        c_pt = b_pt.mean()

        c_my.backward()
        c_pt.backward()

        self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))

    def test_exp_tensor(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        b_my = a_my.exp()
        b_pt = a_pt.exp()

        c_my = b_my.mean()
        c_pt = b_pt.mean()

        c_my.backward()
        c_pt.backward()

        self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))

    def test_exp_then_log(self):
        a_my = Tensor(xp.array([1.0, 2.0, 3.0]), requires_grad=True)
        a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        b_my = a_my.exp().log()
        b_pt = a_pt.exp().log()
        
        c_my = b_my.mean()
        c_pt = b_pt.mean()

        c_my.backward()
        c_pt.backward()

        self.assertTrue(xp.allclose(c_my.data, c_pt.detach().numpy()))
        self.assertTrue(xp.allclose(a_my.grad.data, a_pt.grad.detach().numpy()))


if __name__ == "__main__":
    unittest.main()