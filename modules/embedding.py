from dlx.utils.backend import xp
from dlx.nn.tensor import Tensor
import dlx.nn as nn

class Embedding(nn.Module):
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

        if idx.dtype not in [xp.int32, xp.int64, xp.uint32, xp.uint64]:
            raise ValueError(f"Index must be an integer type, got {idx.dtype}")

        B, T = idx.shape
        embed_vectors = self.W[idx]
        pe_slice = self.pe[:T][None, :, :]
        output = embed_vectors + pe_slice

        return output
