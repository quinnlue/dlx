import unittest


import torch
from dlx.nn.tensor import Tensor
from dlx.nn.losses import MeanSquaredError
from dlx.utils.backend import xp    


class TestMSELoss(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-5):
        self.assertTrue(xp.allclose(a, b, atol=atol))

    def test_mse_loss_2d(self):
        a_my = Tensor(xp.array([[1., 2., 3.], [4., 5., 6.]]), requires_grad=True)
        b_my = Tensor(xp.array([[7., 8., 9.], [10., 11., 12.]]), requires_grad=False)

        a_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        b_pt = torch.tensor([[7., 8., 9.], [10., 11., 12.]], requires_grad=False)


        loss_my = MeanSquaredError(a_my, b_my)
        loss_pt = torch.nn.MSELoss()(a_pt, b_pt)

        loss_my.backward()
        loss_pt.backward()

        self.assert_close(loss_my.data, loss_pt.detach().numpy())
        self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())

    def test_mse_loss_1d(self):
        a_my = Tensor(xp.array([1., 2., 3.]), requires_grad=True)
        b_my = Tensor(xp.array([4., 5., 6.]), requires_grad=False)

        a_pt = torch.tensor([1., 2., 3.], requires_grad=True)
        b_pt = torch.tensor([4., 5., 6.], requires_grad=False)

        loss_my = MeanSquaredError(a_my, b_my)
        loss_pt = torch.nn.MSELoss()(a_pt, b_pt)

        loss_my.backward()
        loss_pt.backward()

        self.assert_close(loss_my.data, loss_pt.detach().numpy())
        self.assert_close(a_my.grad.data, a_pt.grad.detach().numpy())



if __name__ == "__main__":
    unittest.main()