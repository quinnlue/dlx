from ..utils.backend import xp
from ..nn.module import Module
from ..nn.tensor import Tensor

class Transformer(Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, module_dict=None):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio



        if self.d_head * n_heads != d_model:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        
        self.qkv = self.linear(d_model, d_model * 3, module_dict=module_dict, layer_type="linear", name="qkv")
        self.o = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="o")

        self.proj_up = self.linear(d_model, d_model * mlp_ratio, module_dict=module_dict, layer_type="linear", name="proj_up")
        self.proj_down = self.linear(d_model * mlp_ratio, d_model, module_dict=module_dict, layer_type="linear", name="proj_down")

        self.ln1 = self.layer_norm(shape=d_model, axis=-1, module_dict=module_dict, name="ln1")
        self.ln2 = self.layer_norm(shape=d_model, axis=-1, module_dict=module_dict, name="ln2")

    
    def attend(self, x: Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q = qkv[:, :, :self.d_model]
        k = qkv[:, :, self.d_model:self.d_model * 2]
        v = qkv[:, :, self.d_model * 2:]

        q  = q.reshape((B, T, self.n_heads, self.d_head))
        k  = k.reshape((B, T, self.n_heads, self.d_head))
        v  = v.reshape((B, T, self.n_heads, self.d_head))

        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        kt = k.transpose((0, 1, 3, 2))

        atten_scores = q @ kt * (1 / (self.d_head ** 0.5))
        casual_mask = xp.triu(xp.ones((T, T)) * -xp.inf, k=1).astype(atten_scores.dtype)
        atten_scores = atten_scores + casual_mask

        atten_probs = self.softmax(atten_scores, axis=3)

        output = atten_probs @ v

        output = output.transpose((0, 2, 1, 3))

        output = output.reshape((B, T, -1))

        output = self.o(output)

        return output
    
    def forward(self, x: Tensor):
        B, T, _ = x.shape

        residual = x
        x = self.ln1(x)

        atten_out = self.attend(x)

        x = atten_out + residual

        # MLP
        residual = x
        x = self.ln2(x)

        x_mlp = self.proj_up(x)

        x_mlp = self.gelu(x_mlp)

        x_mlp = self.proj_down(x_mlp)

        x = x_mlp + residual

        return x
    
    def __call__(self, x: Tensor):
        return self.forward(x)




