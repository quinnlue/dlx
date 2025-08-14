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
        # (self, in_features, out_features, module_dict, layer_dict, bias=True)
        self.q = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="q")
        self.k = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="k")
        self.v = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="v")
        self.o = self.linear(d_model, d_model, module_dict=module_dict, layer_type="linear", name="o")

        self.proj_up = self.linear(d_model, d_model * mlp_ratio, module_dict=module_dict, layer_type="linear", name="proj_up")
        self.proj_down = self.linear(d_model * mlp_ratio, d_model, module_dict=module_dict, layer_type="linear", name="proj_down")

        self.ln1 = self.layer_norm(axis=-1, module_dict=module_dict, name="ln1")
        self.ln2 = self.layer_norm(axis=-1, module_dict=module_dict, name="ln2")

    
    def attend(self, x: Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q  = q.reshape((B, T, self.n_heads, self.d_head))
        k  = k.reshape((B, T, self.n_heads, self.d_head))
        v  = v.reshape((B, T, self.n_heads, self.d_head))

        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        kt = k.transpose((0, 1, 3, 2))

        atten_scores = q @ kt * (1 / (self.d_head ** 0.5))
        # TODO: remove hard coded float32
        casual_mask = xp.triu(xp.ones((T, T)) * -xp.inf, k=1).astype(xp.float32)
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

class Embedding(Module):
    def __init__(self, vocab_size, d_model, max_seq_len, module_dict=None, layer_dict=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = Tensor(xp.random.randn(vocab_size, d_model), requires_grad=True)
        self.pe = Tensor(xp.random.randn(max_seq_len, d_model), requires_grad=True)

        self.register_parameter(param=self.W, module_dict=module_dict, layer_type="embedding", layer_dict=layer_dict, name="embed")
        self.register_parameter(param=self.pe, module_dict=module_dict, layer_type="embedding", layer_dict=layer_dict, name="pe")
    
    def get_sentence_embedding(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = idx.astype(xp.int32)
        B, T = idx.shape
        embed_vectors = self.W[idx]
        pe_slice = self.pe[:T][None, :, :]
        output = embed_vectors + pe_slice

        return output



