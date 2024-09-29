import torch
import numpy as np
from torch.nn import Softmax


def custom_sdp_attention(q, k, v):
    '''
    simple implementation of scaled dot product attention using matrices q, k, and v
    attention(q, k, v) = softmax((q*k)/d)*v

    input: vectors q: n x dk, k: n x dk, v: n x dv

    output: attention value a: nxdv
    '''
    d = q.shape[-1]
    s = Softmax(dim=1)
    print(s(torch.matmul(q, k.T)/np.sqrt(d)))
    self_attention = torch.matmul(s(torch.matmul(q, k.T)/np.sqrt(d)), v)
    return self_attention


n = 4
dk = 8
dv = 10
q = torch.rand(n, dk)
k = torch.rand(n, dk)
v = torch.rand(n, dv)
attention_custom = custom_sdp_attention(q, k, v)
attention_pytorch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print(torch.allclose(attention_custom, attention_pytorch))
