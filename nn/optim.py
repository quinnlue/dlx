from .tensor import Tensor
from ..utils.backend import Device, get_default_device
import numpy as np
import os
from ..utils.lr_scheduler import LRScheduler

class Optimizer:
    def __init__(
        self, 
        params, lr: LRScheduler | float = 1e-3, 
        clip_norm=1.0, 
        precision: tuple[np.dtype, np.dtype] | np.dtype | None = None,
        device: "str | Device | None" = None,
    ):
        # ── Resolve device ──
        if device is None:
            # Infer from the first parameter tensor, fall back to default
            first = next(iter(params.values()), None)
            if first is not None and hasattr(first, "device"):
                device = first.device
            else:
                device = get_default_device()
        elif isinstance(device, str):
            device = Device(device)
        self.device = device
        xp = self.device.xp

        # Set precision -------------------------------------------------------------
        if precision is not None:
            if isinstance(precision, tuple):
                if len(precision) != 2:
                    raise ValueError("precision must contain a tuple of two elements")
                self.model_dtype, self.master_dtype = precision
                self.mixed_precision = True
            elif isinstance(precision, np.dtype):
                self.model_dtype = self.master_dtype = precision
                self.mixed_precision = False
            else:
                raise ValueError("precision must be a dtype or tuple of dtypes")
        else:
            self.model_dtype = self.master_dtype = xp.float32
            self.mixed_precision = False

        if self.model_dtype == xp.float16:
            self.model_eps = 1e-5
        elif self.model_dtype == xp.float32 or self.model_dtype == xp.float64:
            self.model_eps = 1e-8
        else:
            raise ValueError("model_dtype must be float16, float32, or float64")
        
        if self.master_dtype == xp.float16:
            self.master_eps = 1e-5
        elif self.master_dtype == xp.float32 or self.master_dtype == xp.float64:
            self.master_eps = 1e-8
        else:
            raise ValueError("master_dtype must be float16, float32, or float64")
    
        # Set params -------------------------------------------------------------
        self._raw_params = params
        self.params = {}
        for name, param in self._raw_params.items():
            self.params[name] = {
                "param": param.astype(self.model_dtype),
            }

            # Set master param if mixed precision
            if self.mixed_precision:
                self.params[name]["param_master"] = Tensor(param.data.astype(self.master_dtype), requires_grad=False, device=self.device)
                self.params[name]["param_master"].requires_grad = False

        # Set lr scheduler -------------------------------------------------------------
        self.lr_scheduler = lr if isinstance(lr, LRScheduler) else None
        self.lr = lr
        self.t = 0

        self.clip_norm = clip_norm
        self.is_cuda = self.device.type == "cuda"


    def _clip_norm(self, grad: Tensor, total_norm: float):
        if self.clip_norm is None:
            raise ValueError("clip_norm is not set")
        
        if total_norm > self.clip_norm:
            grad.data *= (self.clip_norm / total_norm)
        return grad

    def get_lr(self, step: int):
        if self.lr_scheduler is not None:
            return self.lr_scheduler(step)
        else:
            return self.lr

    def set_lr_scheduler(self, lr_scheduler: LRScheduler):
        self.lr_scheduler = lr_scheduler

    def reduce_like(self, grad: Tensor, target_shape: tuple) -> Tensor:
        reduced = Tensor._reduce_broadcast(grad.data, target_shape)
        return Tensor(reduced, requires_grad=False, device=self.device)

    def zero_grad(self):
        for param in self.params.values():
            if param['param'].grad is not None:
                param['param'].grad = None

    def _save_params(self, path):
        # ONLY CALL WITHIN THE SPECIFIC OPTIMIZER (AdamW, Standard, etc.)
        xp = self.device.xp
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/model", exist_ok=True)

        for name, param in self.params.items():
            if self.mixed_precision:
                xp.save(f"{path}/model/{name}.npy", param['param_master'].data)
            else:
                xp.save(f"{path}/model/{name}.npy", param['param'].data)

        xp.save(f"{path}/optim/t.npy", self.t)

    def _load_params(self, path):
        # ONLY CALL WITHIN THE SPECIFIC OPTIMIZER (AdamW, Standard, etc.)
        xp = self.device.xp
        for name, param in self.params.items():
            if self.mixed_precision:
                param['param_master'].data = xp.load(f"{path}/model/{name}.npy")
                param['param'].data = param['param_master'].data.astype(self.model_dtype)
            else:
                param['param'].data = xp.load(f"{path}/model/{name}.npy")


    def _get_total_norm(self):
        xp = self.device.xp
        total_norm = 0.0
        for param in self.params.values():
            if param['param'].grad is None:
                continue
            total_norm += xp.linalg.norm(param['param'].grad.data) ** 2
        return xp.sqrt(total_norm)

class AdamW(Optimizer):
    def __init__(
        self,
        params, 
        lr: LRScheduler | float = 1e-3, 
        clip_norm=1.0, 
        weight_decay=0.01, 
        beta_1=0.9, 
        beta_2=0.95, 
        precision: tuple[np.dtype, np.dtype] | np.dtype | None = None,
        device: "str | Device | None" = None,
    ):
        super().__init__(params, lr=lr, clip_norm=clip_norm, precision=precision, device=device)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clip_norm = clip_norm

        xp = self.device.xp
        for name, param in self.params.items():
            # Create momentum and variance tensors with the correct dtype and shape
            param_shape = param['param'].data.shape
            self.params[name].update({
                'm_t': xp.zeros(param_shape, dtype=self.master_dtype),
                'v_t': xp.zeros(param_shape, dtype=self.master_dtype),
            })

    def save_state(self, path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/optim", exist_ok=True)
        os.makedirs(f"{path}/optim/m_t", exist_ok=True)
        os.makedirs(f"{path}/optim/v_t", exist_ok=True)

        self._save_params(path)
        for name, param in self.params.items():
            np.save(f"{path}/optim/m_t/{name}.npy", param['m_t'])
            np.save(f"{path}/optim/v_t/{name}.npy", param['v_t'])

    def load_state(self, path):
        xp = self.device.xp
        self._load_params(path)
        for name, param in self.params.items():
            param['m_t'] = xp.array(np.load(f"{path}/optim/m_t/{name}.npy"))
            param['v_t'] = xp.array(np.load(f"{path}/optim/v_t/{name}.npy"))

        self.t = int(np.load(f"{path}/optim/t.npy"))

    def step(self):
        xp = self.device.xp
        if self.clip_norm is not None:
            total_norm = self._get_total_norm()
            clip_coef = 1.0
            if total_norm > self.clip_norm:
                clip_coef = float(self.clip_norm / (total_norm + 1e-8))
    
        for name, param in self.params.items():
            if param['param'].grad is None:
                continue

            grad = param['param'].grad

            if self.clip_norm is not None:
                grad = self._clip_norm(grad, total_norm)
                param['param'].grad = grad  # Ensure clipped grad is used for update and logging


            if self.mixed_precision:
                master_param_tensor = param['param_master']
                dtype = self.master_dtype
            else:
                master_param_tensor = param['param']
                dtype = self.model_dtype

            m_t = param['m_t']
            v_t = param['v_t']

            if grad.shape != master_param_tensor.data.shape:
                grad = self.reduce_like(grad, master_param_tensor.data.shape)
            if m_t.shape != master_param_tensor.data.shape:
                m_t = self.reduce_like(m_t, master_param_tensor.data.shape)
            if v_t.shape != master_param_tensor.data.shape:
                v_t = self.reduce_like(v_t, master_param_tensor.data.shape)

            grad_data = grad.data.astype(dtype)

            # ---------- decoupled weight-decay (AdamW style) ----------
            if self.weight_decay != 0.0:
                wd_lr = self.get_lr(self.t + 1) * self.weight_decay
                master_param_tensor.data = master_param_tensor.data - wd_lr * master_param_tensor.data
            # ----------------------------------------------------------

            m_t = m_t * self.beta_1 + (1 - self.beta_1) * grad_data
            v_t = v_t * self.beta_2 + (1 - self.beta_2) * (grad_data ** 2)

            m_hat = m_t / xp.maximum(1 - self.beta_1 ** (self.t + 1), self.master_eps)
            v_hat = v_t / xp.maximum(1 - self.beta_2 ** (self.t + 1), self.master_eps)

            # gradient update (Adam)
            adam_update = (
                self.get_lr(self.t + 1)
                * m_hat
                / (xp.sqrt(v_hat) + self.master_eps).clip(min=self.master_eps)
            )

            # Apply global clipping on the update magnitude to ensure smaller clip_norm yields smaller step
            if self.clip_norm is not None:
                # clip_coef computed once globally above; default to 1.0 if not set (e.g., when clip_norm is None)
                try:
                    scale = clip_coef
                except NameError:
                    scale = 1.0
                adam_update = adam_update * scale

            master_param_tensor.data = master_param_tensor.data - adam_update

            param['m_t'] = m_t
            param['v_t'] = v_t

            # sync master param to model param
            if self.mixed_precision:
                param['param'].data = master_param_tensor.data.astype(self.model_dtype)

        self.t += 1

class SGD(Optimizer):
    def __init__(self, params, lr: LRScheduler | float = 1e-3, clip_norm=1.0):
        super().__init__(params, lr=lr, clip_norm=clip_norm)
        self.clip_norm = clip_norm

    def step(self):
        # Increase timestep
        self.t += 1
        lr_t = self.get_lr(self.t)
        if self.clip_norm is not None:
            total_norm = self._get_total_norm()

        for param in self.params.values():
            if param['param'].grad is None:
                continue

            grad = param['param'].grad
            grad = self._clip_norm(grad, total_norm)
            param['param'].grad = grad

            grad = self.reduce_like(grad, param['param'].data.shape)

            param['param'].data = param['param'].data - lr_t * grad.data

