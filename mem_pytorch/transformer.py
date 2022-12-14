from functools import partial
import pdb
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from mem_pytorch.triton.layer_norm import layernorm
from mem_pytorch.triton.softmax import softmax
from mem_pytorch.triton.cross_entropy import cross_entropy_fn
from mem_pytorch.triton.bmm import fused_relu_squared
from mem_pytorch.triton.dropout import dropout_fn
from mem_pytorch.triton.flash_attention import triton_flash_attention

from mem_pytorch.triton.utils import exists, default

from mem_pytorch.flash import plain_cosine_sim_attention, flash_cosine_sim_attention

# classes


def l2norm(t):
    return F.normalize(t, dim = -1)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, use_triton = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.use_triton = use_triton

    def forward(self, x, **kwargs):
        use_triton = kwargs.get('use_triton', self.use_triton)
        normed = layernorm(x, self.norm.weight, use_triton = use_triton)
        return self.fn(normed, **kwargs) + x

# helpers classes

class CosineAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8,
        l2norm_groups = 1,
        use_cuda_kernel = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = scale
        self.heads = heads

        self.l2norm_groups = l2norm_groups
        self.attn_fn = plain_cosine_sim_attention if not use_cuda_kernel else partial(flash_cosine_sim_attention, **kwargs)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, scale, l2norm_groups = self.heads, self.scale, self.l2norm_groups

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self.attn_fn(q, k, v, causal = True, scale = scale, groups = l2norm_groups)

        o = rearrange(o, 'b h n d -> b n (h d)')
        return self.to_out(o)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads
        d_head = self.dim_head
        BATCH = x.shape[0]
        N_CTX = x.shape[1]
        H = h
        D_HEAD = d_head
        # dtype = x.dtype
        in_dtype = torch.float16
        out_dtype = torch.float32

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))


        # BATCH = x.shape[0]
        # H is self.heads
        # SEQ_LEN is x.shape[1]
        # DIM_HEAD is x.shape[2]

        # reshape q, k, v to (BATCH, H, N_CTX, D_HEAD)
        query = q.reshape(x.shape[0], h, x.shape[1], d_head)
        k = k.reshape(x.shape[0], h, x.shape[1], d_head)
        v = v.reshape(x.shape[0], h, x.shape[1], d_head)

        # # cast to float16
        query = query.to(in_dtype)
        k = k.to(in_dtype)
        v = v.to(in_dtype)

        # # einsum transform q, k, v to (BATCH, H, N_CTX, N_CTX)

        out = triton_flash_attention(query, k, v, self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')
  
        # # out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # # cast to float32
        out = out.to(out_dtype)
        # # pdb.set_trace()

        out = self.to_out(out)

        return out

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        inner_dim = dim * mult
        self.dropout = dropout
        self.proj_in_weight = nn.Parameter(torch.randn(dim, inner_dim))
        # self.proj_in_weight = nn.Parameter(torch.randn(dim, dim))
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(self, x, use_triton = None):
        use_triton = default(use_triton, self.use_triton)

        x = fused_relu_squared(x, self.proj_in_weight, use_triton = False) # disabled until I can fix it
        x = dropout_fn(x, self.dropout, use_triton = use_triton)

        x = self.proj_out(x)
        return x

# main class

class TritonTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        ff_mult = 4,
        attn_dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        wrapper = partial(PreNormResidual, dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                wrapper(CosineAttention(dim, heads = heads, dim_head = dim_head, causal = causal, use_cuda_kernel= True)),
                wrapper(FeedForward(dim, dropout = ff_dropout, mult = ff_mult, use_triton = use_triton))
            ]))

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        # mask

        self.use_triton = use_triton
        self.causal = causal
        mask = torch.ones(max_seq_len, max_seq_len, dtype = torch.bool).triu(1) if causal else None
        self.register_buffer('mask', mask, persistent = False)

    def forward(
        self,
        x,
        mask = None,
        *,
        labels = None,
        use_triton = None
    ):
        use_triton = default(use_triton, self.use_triton)
        n, device = x.shape[1], x.device

        # embed token and add positional embedding

        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        # generate mask, depending on whether autoregressive or not

        assert not (self.causal and exists(mask)), 'mask is not needed during autoregressive mode'

        if self.causal and not use_triton:
            mask = self.mask[:n, :n]
            mask = rearrange(mask, 'i j -> () i j')
        elif not self.causal and exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = ~mask

        # go through layers

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x, use_triton = use_triton)

        x = layernorm(x, self.norm.weight, use_triton = False, stable = True) # turn this back on
        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        loss = cross_entropy_fn(logits, labels, ignore_index = 0, use_triton = False)

        return loss
