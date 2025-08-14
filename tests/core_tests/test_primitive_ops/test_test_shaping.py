import unittest

import torch
from dlx.nn.tensor import Tensor
from dlx.utils.backend import xp



class TestShaping(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_transpose(self):
        shapes = [
            (2, 3),
            (3, 2, 4),
            (4, 5, 6, 7),
        ]

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            # create permutation: just reverse axes for fun
            perm = tuple(reversed(range(len(shape))))

            b_my = a_my.transpose(perm)
            b_pt = a_pt.permute(*perm)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            self.assert_close(c_my.data, c_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())

    def test_reshape(self):
        shapes = [
            (2, 3),
            (3, 4, 2),
            (4, 5, 6),
        ]

        for shape in shapes:
            a_my = Tensor(xp.random.randn(*shape).astype(xp.float32), requires_grad=True)
            a_pt = torch.tensor(a_my.data, requires_grad=True)

            # flatten and reshape back in a different way
            new_shape = (xp.prod(shape) // 2, 2) if xp.prod(shape) % 2 == 0 else (xp.prod(shape),)

            b_my = a_my.reshape(new_shape)
            b_pt = a_pt.reshape(new_shape)

            c_my = b_my.mean()
            c_pt = b_pt.mean()

            c_my.backward()
            c_pt.backward()

            self.assert_close(b_my.data, b_pt.detach().numpy())
            self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())



if __name__ == "__main__":
    unittest.main()