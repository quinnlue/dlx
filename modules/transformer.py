from ..utils.backend import xp
from ..nn.module import Module
from ..nn.tensor import Tensor

class Transformer(Module):
    def __init__(
        self, 
        d_model, 
        n_heads, 
        mlp_ratio=4, 
        lora=False,
        lora_r=16,
        lora_alpha=16,
        module_dict=None
    ):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if self.d_head * n_heads != d_model:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        
        self.qkv = self.linear(d_model, d_model * 3, module_dict=module_dict, layer_type="linear", name="qkv")
        self.o = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="o")

        if self.lora:
            self.q_lora_A = self.linear(d_model, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="q_lora_A")
            self.k_lora_A = self.linear(d_model, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="k_lora_A")
            self.v_lora_A = self.linear(d_model, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="v_lora_A")
            self.q_lora_B = self.linear(lora_r, d_model, module_dict=module_dict, layer_type="linear", bias=False, name="q_lora_B")
            self.k_lora_B = self.linear(lora_r, d_model, module_dict=module_dict, layer_type="linear", bias=False, name="k_lora_B")
            self.v_lora_B = self.linear(lora_r, d_model, module_dict=module_dict, layer_type="linear", bias=False, name="v_lora_B")
            self.o_lora_A = self.linear(d_model, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="o_lora_A")
            self.o_lora_B = self.linear(lora_r, d_model, module_dict=module_dict, layer_type="linear", bias=False, name="o_lora_B")

            self.proj_up_lora_A = self.linear(d_model, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="proj_up_lora_A")
            self.proj_up_lora_B = self.linear(lora_r, d_model * mlp_ratio, module_dict=module_dict, layer_type="linear", bias=False, name="proj_up_lora_B")
            self.proj_down_lora_A = self.linear(d_model * mlp_ratio, lora_r, module_dict=module_dict, layer_type="linear", bias=False, name="proj_down_lora_A")
            self.proj_down_lora_B = self.linear(lora_r, d_model, module_dict=module_dict, layer_type="linear", bias=False, name="proj_down_lora_B")
            self.scaling = self.lora_alpha / self.lora_r

        self.proj_up = self.linear(d_model, d_model * mlp_ratio, module_dict=module_dict, layer_type="linear", name="proj_up")
        self.proj_down = self.linear(d_model * mlp_ratio, d_model, module_dict=module_dict, layer_type="linear", name="proj_down")

        self.ln1 = self.layer_norm(shape=d_model, axis=-1, module_dict=module_dict, name="ln1")
        self.ln2 = self.layer_norm(shape=d_model, axis=-1, module_dict=module_dict, name="ln2")


    def cached_attend(self, x: Tensor, k_cache: Tensor, v_cache: Tensor):
        # kv_cache is of shape (batch_size, max_seq_len, d_model)
        # x is of shape (batch_size, seq_len, d_model)
        B, T, _ = x.shape
        if B != 1:
            raise ValueError(f"Inference mode only supports batch size 1")

        x = x[:, :-1, :]
        qkv = self.qkv(x)
        q = qkv[:, :, :self.d_model]
        k = qkv[:, :, self.d_model:self.d_model * 2]
        v = qkv[:, :, self.d_model * 2:]

        if self.lora:
            q_lora_delta = self.scaling * (x @ self.q_lora_A.weight@ self.q_lora_B.weight)
            k_lora_delta = self.scaling * (x @ self.k_lora_A.weight@ self.k_lora_B.weight)
            v_lora_delta = self.scaling * (x @ self.v_lora_A.weight@ self.v_lora_B.weight)
            q = q + q_lora_delta
            k = k + k_lora_delta
            v = v + v_lora_delta


        k_cache[:, T-1, :] = k
        v_cache[:, T-1, :] = v

        q = q.reshape((B, 1, self.n_heads, self.d_head))
        k_cached = k_cache[:, :T, :].reshape((B, T, self.n_heads, self.d_head))
        v_cached = v_cache[:, :T, :].reshape((B, T, self.n_heads, self.d_head))

        q = q.transpose((0, 2, 1, 3))
        k_cached = k_cached.transpose((0, 2, 1, 3))
        v_cached = v_cached.transpose((0, 2, 1, 3))
        kt_cached = k_cached.transpose((0, 1, 3, 2))

        atten_scores = q @ kt_cached * (1 / (self.d_head ** 0.5))

        atten_probs = self.softmax(atten_scores, axis=3) # (B, n_heads, 1, T)

        output = atten_probs @ v_cached

        output = output.transpose((0, 2, 1, 3))
        output = output.reshape((B, 1, -1))
        if self.lora:
            output = output + self.scaling * (output @ self.o_lora_A.weight@ self.o_lora_B.weight)
        output = self.o(output)
        return output



    
    def attend(self, x: Tensor):
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q = qkv[:, :, :self.d_model]
        k = qkv[:, :, self.d_model:self.d_model * 2]
        v = qkv[:, :, self.d_model * 2:]

        if self.lora:
            q_lora_delta = self.scaling * (x @ self.q_lora_A.weight@ self.q_lora_B.weight)
            k_lora_delta = self.scaling * (x @ self.k_lora_A.weight@ self.k_lora_B.weight)
            v_lora_delta = self.scaling * (x @ self.v_lora_A.weight@ self.v_lora_B.weight)
            q = q + q_lora_delta
            k = k + k_lora_delta
            v = v + v_lora_delta

        q  = q.reshape((B, T, self.n_heads, self.d_head))
        k  = k.reshape((B, T, self.n_heads, self.d_head))
        v  = v.reshape((B, T, self.n_heads, self.d_head))



        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        kt = k.transpose((0, 1, 3, 2))

        atten_scores = q @ kt * (1 / (self.d_head ** 0.5))
        causal_mask = xp.triu(xp.full((T, T), fill_value=-xp.inf, dtype=atten_scores.dtype), k=1)

        atten_scores = atten_scores + causal_mask

        atten_probs = self.softmax(atten_scores, axis=3)

        output = atten_probs @ v

        output = output.transpose((0, 2, 1, 3))

        output = output.reshape((B, T, -1))

        if self.lora:
            output = output + self.scaling * (output @ self.o_lora_A.weight@ self.o_lora_B.weight)

        output = self.o(output)

        return output


    def mlp(self, x: Tensor):
        if self.lora:
            proj_up_lora_delta = self.scaling * (x @ self.proj_up_lora_A.weight @ self.proj_up_lora_B.weight)
            x = self.proj_up(x) + proj_up_lora_delta
            
            x = self.gelu(x)
            
            proj_down_lora_delta = self.scaling * (x @ self.proj_down_lora_A.weight @ self.proj_down_lora_B.weight)
            x = self.proj_down(x) + proj_down_lora_delta
        else:
            x = self.proj_up(x)
            x = self.gelu(x)
            x = self.proj_down(x)
        return x

    def forward(
        self, x: Tensor, 
        k_cache: Tensor | None = None, 
        v_cache: Tensor | None = None
    ):
        if k_cache is not None and v_cache is not None:
            x = x + self.cached_attend(self.ln1(x), k_cache, v_cache)
        else:
            x = x + self.attend(self.ln1(x))

        x = self.ln2(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        return x
    
    # def forward(self, x: Tensor):
    #     residual = x
    #     x = self.ln1(x)

    #     atten_out = self.attend(x)

    #     x = atten_out + residual

    #     # MLP
    #     residual = x
    #     x = self.ln2(x)

    #     x_mlp = self.proj_up(x)

    #     x_mlp = self.gelu(x_mlp)

    #     x_mlp = self.proj_down(x_mlp)

    #     x = x_mlp + residual

    #     return x
    
    def __call__(self, x: Tensor, k_cache: Tensor | None = None, v_cache: Tensor | None = None):
        return self.forward(x, k_cache, v_cache)




