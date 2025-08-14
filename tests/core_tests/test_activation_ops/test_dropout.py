import unittest

import torch
import torch.nn.functional as F

from dlx.utils.backend import xp
from dlx.nn.tensor import Tensor
from dlx.nn.module import Module
from dlx.utils.backend import set_seed


class TestDropout(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def assert_close(self, a, b, atol=1e-3):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_dropout_tensor(self):
        shapes = [
            (1,),
            (2,3),
            (3,4,5),
            (4,5,6,7),
        ]

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            p = 0.1
            mask = (xp.random.rand(*a_my.data.shape) < (1 - p)).astype(xp.float32)
            b_my = a_my._dropout(p=p, train=True, mask=mask)
            b_pt = a_pt * torch.tensor(mask, dtype=a_pt.dtype)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            print("Max forward diff:", xp.max(xp.abs(b_my.data - b_pt.detach().numpy())))
            print("Max backward diff:", xp.max(xp.abs(a_my.grad.data - a_pt.grad.detach().numpy())))

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy(), atol=1e-2)

    def test_dropout_module(self):
        shapes = [
            (1,),
            (2,3),
            (3,4,5),
            (4,5,6,7),
        ]

        module = Module()

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            p = 0.2
            mask = (xp.random.rand(*a_my.data.shape) < (1 - p)).astype(xp.float32)
            b_my = module.dropout(a_my, p=p, mask=mask)
            b_pt = a_pt * torch.tensor(mask, dtype=a_pt.dtype)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            print("Max forward diff:", xp.max(xp.abs(b_my.data - b_pt.detach().numpy())))
            print("Max backward diff:", xp.max(xp.abs(a_my.grad.data - a_pt.grad.detach().numpy())))

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy(), atol=1e-2)

    def test_dropout_eval_mode(self):
        shapes = [
            (1,),
            (2,3),
            (3,4,5),
        ]

        module = Module()

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            # Test dropout in eval mode (should not apply dropout)
            module.is_training = False
            b_my = module.dropout(a_my, p=0.5)
            b_pt = F.dropout(a_pt, p=0.5, training=False)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            print("Max forward diff (eval):", xp.max(xp.abs(b_my.data - b_pt.detach().numpy())))
            print("Max backward diff (eval):", xp.max(xp.abs(a_my.grad.data - a_pt.grad.detach().numpy())))

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy(), atol=1e-2)

            # Reset to training mode
            module.is_training = True


if __name__ == "__main__":
    unittest.main()
