import unittest

import torch

from dlx.nn.tensor import Tensor
from dlx.utils.backend import xp


class TestGELU(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-3):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_gelu_tensor(self):
        shapes = [
            (1,),
            (2,3),
            (3,4,5),
            (4,5,6,7),
        ]

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            b_my = a_my._gelu()
            b_pt = torch.nn.functional.gelu(a_pt)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            print("Max forward diff:", xp.max(xp.abs(b_my.data - b_pt.detach().numpy())))
            print("Max backward diff:", xp.max(xp.abs(a_my.grad.data - a_pt.grad.detach().numpy())))

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy(), atol=1e-2)


if __name__ == "__main__":
    unittest.main()