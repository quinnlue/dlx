from ..utils.backend import Device, get_default_device
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=True, requires_mask=False, name=None, eps=1e-5, dtype=None, device=None):
        # Device management
        if device is None:
            if isinstance(data, Tensor):
                device = data.device
            else:
                device = get_default_device()
        elif isinstance(device, str):
            device = Device(device)
        self.device = device

        xp = self.device.xp

        # Data management
        if dtype is None:
            dtype = xp.float32

        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, xp.ndarray) and data.dtype == dtype:
            self.data = data
        else:
            if hasattr(data, 'get'):
                data = data.get()
            self.data = xp.asarray(data, dtype=dtype)

        if dtype == xp.float16:
            self.eps = 1e-5
        else:
            self.eps = 1e-8


        self.requires_grad = requires_grad
        self.parents = ()
        self.grad_fn = None
        self.grad = None
        self.requires_mask = requires_mask
        self.mask = None
        self.name = name


    # Utils ------------------------------------------------------------------------------------ #
    @property
    def is_cuda(self):
        return self.device.type == "cuda"
        
    @property
    def n_params(self):
        if self.data.ndim == 1:
            return self.data.shape[0]
        elif self.data.ndim == 2:
            return self.data.shape[0] * self.data.shape[1]
        else:
            raise ValueError(f"Tensor has {self.data.ndim} dimensions, expected 1 or 2")

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def __len__(self):
        return len(self.data)

    def astype(self, dtype):
        # Edits type in place
        self.data = self.data.astype(dtype)
        return self

    def to(self, device):
        """Move tensor to a different device, returning a new Tensor."""
        if isinstance(device, str):
            device = Device(device)
        if device == self.device:
            return self

        if self.device.type == "cuda":
            raw = self.data.get()
        else:
            raw = self.data

        new_data = device.xp.asarray(raw)
        out = Tensor(new_data, requires_grad=self.requires_grad, device=device)
        if self.grad is not None:
            out.grad = self.grad.to(device)
        return out

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def _check_device(self, other):
        if self.device != other.device:
            raise RuntimeError(
                f"Tensors on different devices: {self.device} vs {other.device}"
            )

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    # Shaping ------------------------------------------------------------------------------------ #
    @staticmethod
    def _reduce_broadcast(grad_arr, target_shape):
        """Sum-reduce grad_arr along axes that were broadcast to match target_shape."""
        if grad_arr.shape == target_shape:
            return grad_arr
        ndim_diff = grad_arr.ndim - len(target_shape)
        target_ext = (1,) * ndim_diff + target_shape
        axes = tuple(
            i for i, (g, t) in enumerate(zip(grad_arr.shape, target_ext)) if t == 1
        )
        if axes:
            grad_arr = grad_arr.sum(axis=axes, keepdims=True)
        return grad_arr.reshape(target_shape)

    @staticmethod
    def _expand_reduction_grad(grad_data, input_shape, axis):
        rank = len(input_shape)
        if axis is None:
            norm_axes = tuple(range(rank))
        else:
            norm_axes = axis if isinstance(axis, tuple) else (axis,)
            norm_axes = tuple(a if a >= 0 else rank + a for a in norm_axes)
        grad_shape = list(input_shape)
        for ax in norm_axes:
            grad_shape[ax] = 1
        return grad_data.reshape(grad_shape), norm_axes


    def transpose(self, axes):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.transpose(self.data, axes), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            inv_axes = tuple(xp.argsort(xp.array(axes)).tolist())
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.transpose(inv_axes),)
            out.grad_fn = grad_fn
        return out

    def reshape(self, shape):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.reshape(self.data, shape), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            in_shape = self.data.shape
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad.reshape(in_shape),)
            out.grad_fn = grad_fn
        return out


    # Reductions ------------------------------------------------------------------------------------ #

    def mean(self, axis=None, keepdims=False):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_expanded, norm_axes = Tensor._expand_reduction_grad(grad.data, self.data.shape, axis)
                denom = 1
                for ax in norm_axes:
                    denom *= int(self.data.shape[ax])
                grad_self = xp.ones_like(self.data) * (grad_expanded / denom)
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def _extremum(self, axis, keepdims, xp_fn):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp_fn(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_expanded, _ = Tensor._expand_reduction_grad(grad.data, self.data.shape, axis)
                extreme_vals = xp_fn(self.data, axis=axis, keepdims=True)
                mask = (self.data == extreme_vals)
                grad_self = grad_expanded * mask
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def max(self, axis=None, keepdims=False):
        return self._extremum(axis, keepdims, self.device.xp.max)

    def min(self, axis=None, keepdims=False):
        return self._extremum(axis, keepdims, self.device.xp.min)

    def sum(self, axis=None, keepdims=False):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_expanded, _ = Tensor._expand_reduction_grad(grad.data, self.data.shape, axis)
                grad_self = xp.ones_like(self.data) * grad_expanded
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def pow(self, other):
        return self.__pow__(other)

    # Unary ops ------------------------------------------------------------------------------------ #

    def __neg__(self):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.negative(self.data), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (-grad,)
            out.grad_fn = grad_fn
        return out

    # Binary ops ------------------------------------------------------------------------------------ #

    def __add__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=device)
        self._check_device(other)

        out = Tensor(xp.add(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     device=device)

        if out.requires_grad:
            out.parents = (self, other)

            def grad_fn(grad):
                grad.requires_grad = False
                grad_self  = Tensor._reduce_broadcast(grad.data, self.data.shape)
                grad_other = Tensor._reduce_broadcast(grad.data, other.data.shape)
                return (Tensor(grad_self,  requires_grad=False, device=device),
                        Tensor(grad_other, requires_grad=False, device=device))

            out.grad_fn = grad_fn
        return out

    def __radd__(self, other):
        return self + other



    def __matmul__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=device)
        self._check_device(other)

        out = Tensor(xp.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                def safe_transpose(x):
                    if x.ndim < 2:
                        return x
                    return xp.swapaxes(x, -1, -2)

                grad.requires_grad = False
                other_T = safe_transpose(other.data)
                self_T  = safe_transpose(self.data)

                grad_self = xp.matmul(grad.data, other_T)
                if self.data.ndim > 2 and other.data.ndim == 2:
                    grad_other = xp.matmul(self_T, grad.data)
                    sum_axes = tuple(range(grad_other.ndim - 2))
                    if sum_axes:
                        grad_other = xp.sum(grad_other, axis=sum_axes)
                else:
                    grad_other = xp.matmul(self_T, grad.data)
                return (Tensor(grad_self, requires_grad=False, device=device), Tensor(grad_other, requires_grad=False, device=device))
            out.grad_fn = grad_fn
        return out

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return other.__matmul__(self)

    def __sub__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=device)
        self._check_device(other)

        out = Tensor(xp.subtract(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self, other)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self  = Tensor._reduce_broadcast(grad.data, self.data.shape)
                grad_other = Tensor._reduce_broadcast(-grad.data, other.data.shape)
                return (Tensor(grad_self, requires_grad=False, device=device), Tensor(grad_other, requires_grad=False, device=device))
            out.grad_fn = grad_fn
        return out



    def __pow__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, (int, float)):
            raise NotImplementedError("Power must be a scalar")

        out = Tensor(xp.power(self.data, other), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * other * xp.power(self.data, other - 1)
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def __rsub__(self, other):
        return -self + other

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape}, dtype={self.data.dtype}, device={self.device})"

    def __str__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, device={self.device})"

    def __mul__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=device)
        self._check_device(other)

        out = Tensor(xp.multiply(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self, other)

            def grad_fn(grad):
                grad.requires_grad = False
                g_self = grad.data * other.data
                g_self = Tensor._reduce_broadcast(g_self, self.data.shape)

                g_other = grad.data * self.data
                g_other = Tensor._reduce_broadcast(g_other, other.data.shape)

                return (Tensor(g_self, requires_grad=False, device=device),
                        Tensor(g_other, requires_grad=False, device=device))
            out.grad_fn = grad_fn
        return out

    def __rmul__(self, other):
        return self * other



    def __div__(self, other):
        xp = self.device.xp
        device = self.device
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=device)

        if not isinstance(self, Tensor):
            self = Tensor(self, requires_grad=False, device=device)
        self._check_device(other)

        out = Tensor(xp.divide(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self, other)

            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data / other.data
                grad_self = Tensor._reduce_broadcast(grad_self, self.data.shape)

                grad_other = -grad.data * self.data / (other.data ** 2)
                grad_other = Tensor._reduce_broadcast(grad_other, other.data.shape)

                return (Tensor(grad_self, requires_grad=False, device=device),
                        Tensor(grad_other, requires_grad=False, device=device))

            out.grad_fn = grad_fn
        return out

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, device=self.device)
        return other.__truediv__(self)


    # Not implemented ------------------------------------------------------------------------------------ #
    def __idiv__(self, other):
        raise NotImplementedError("idiv not implemented")

    def __itruediv__(self, other):
        raise NotImplementedError("itruediv not implemented")

    def __iadd__(self, other):
        raise NotImplementedError("iadd not implemented")

    def __imul__(self, other):
        raise NotImplementedError("imul not implemented")

    def __isub__(self, other):
        raise NotImplementedError("isub not implemented")

    # Elementwise math ------------------------------------------------------------------------------------ #

    def exp(self):
        xp = self.device.xp
        device = self.device
        out = Tensor(xp.exp(self.data), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad * out,)
            out.grad_fn = grad_fn
        return out

    def log(self):
        xp = self.device.xp
        device = self.device
        safe_data = xp.maximum(self.data, self.eps)
        out = Tensor(xp.log(safe_data), requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                safe_self = xp.maximum(self.data, self.eps)
                return (grad / Tensor(safe_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    # Activation functions ------------------------------------------------------------------------------------ #

    def _sigmoid(self):
        xp = self.device.xp
        device = self.device
        sigmoid_data = 1.0/(1.0+xp.exp(-self.data))
        out = Tensor(sigmoid_data, requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * sigmoid_data * (1 - sigmoid_data)
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def _softmax(self, axis):
        xp = self.device.xp
        device = self.device
        max_values = self.data.max(axis=axis, keepdims=True)
        shifted = self.data - max_values
        exp_shifted = xp.exp(shifted)
        sum_exp = exp_shifted.sum(axis=axis, keepdims=True)
        softmax_data = exp_shifted / (sum_exp + self.eps)

        out = Tensor(softmax_data, requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                dot = (grad.data * softmax_data).sum(axis=axis, keepdims=True)
                grad_input = softmax_data * (grad.data - dot)
                return (Tensor(grad_input, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def _dropout(self, p: float=0.1, train: bool=True, mask=None):
        xp = self.device.xp
        device = self.device
        if not train:
            return self
        if mask is None:
            mask = xp.random.rand(*self.data.shape) < (1 - p)
        out = Tensor(self.data * mask, requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                return (grad * mask,)
            out.grad_fn = grad_fn
        return out

    def _relu(self):
        xp = self.device.xp
        device = self.device
        relu_data = xp.maximum(0, self.data)
        out = Tensor(relu_data, requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                grad_self = grad.data * (self.data > 0)
                return (Tensor(grad_self, requires_grad=False, device=device),)
            out.grad_fn = grad_fn
        return out

    def _gelu(self):
        xp = self.device.xp
        device = self.device
        gelu_data = 0.5 * self.data * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (self.data + 0.044715 * xp.power(self.data, 3))))
        out = Tensor(gelu_data, requires_grad=self.requires_grad, device=device)
        if out.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                u = xp.sqrt(2 / xp.pi) * (self.data + 0.044715 * xp.power(self.data, 3))
                tanh_u = xp.tanh(u)
                left = 0.5 * (1 + tanh_u)
                right = 0.5 * self.data * (1 - tanh_u ** 2) * xp.sqrt(2 / xp.pi) * (1 + 3 * 0.044715 * xp.power(self.data, 2))
                grad_self = grad.data * (left + right)
                return (Tensor(grad_self, requires_grad=False, device=device),)

            out.grad_fn = grad_fn
        return out

    # Indexing ------------------------------------------------------------------------------------ #

    def __getitem__(self, key):
        xp = self.device.xp
        device = self.device
        out_data = self.data[key]
        out = Tensor(out_data, requires_grad=self.requires_grad, device=device)

        if self.requires_grad:
            out.parents = (self,)
            def grad_fn(grad):
                grad.requires_grad = False
                full = xp.zeros_like(self.data)
                try:
                    xp.add.at(full, key, grad.data)
                except Exception:
                    full[key] = grad.data
                return (Tensor(full, requires_grad=False, device=device),)
            out.grad_fn = grad_fn

        return out

    # Autograd ------------------------------------------------------------------------------------ #

    def backward(self, grad=None):
        xp = self.device.xp
        device = self.device
        if grad is None:
            grad = Tensor(xp.ones_like(self.data), requires_grad=False, device=device)
        self.grad = grad

        topo = []
        visited = set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            for p in getattr(v, "parents", ()):
                if p is not v:
                    build(p)
            topo.append(v)

        build(self)

        for v in reversed(topo):
            if v.grad is None:
                continue
            if v.grad_fn is None or not getattr(v, "parents", None):
                continue

            parent_grads = v.grad_fn(v.grad)
            for p, g in zip(v.parents, parent_grads):
                if g is None or not getattr(p, "requires_grad", False):
                    continue
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.data += g.data

    def zero_grad(self):
        self.grad = None
        self.grad_fn = None
        self.parents = ()
