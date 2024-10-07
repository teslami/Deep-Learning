import torch
import numpy as np
from torch.nn import Softmax


def multihead_attention(num_head, q, k, v):
    '''
    performing multihead attention
    num_head: number of heads
    embeddings: tensor of size bxnxd
        b: batch size
        n: number of tokens in each bactch
        d: embedding size
        q, k, v: query, key, value
    '''
    d = q.shape[1]
    head_dim = int(d/num_head)
    output = torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=torch.int64)
    q = torch.split(q, head_dim, dim=1)
    k = torch.split(k, head_dim, dim=1)
    v = torch.split(v, head_dim, dim=1)

    for i in range(num_head):
        q_head_i = q[i]
        k_head_i = k[i]
        v_head_i = v[i]
        attetion_head = custom_sdp_attention(q_head_i, k_head_i, v_head_i)
        output[:, i*head_dim:(i+1)*head_dim, :] = attetion_head
    return output


def custom_sdp_attention(q, k, v):
    '''
    simple implementation of scaled dot product attention using matrices q, k, and v
    attention(q, k, v) = softmax((q*k)/d)*v

    input: vectors q: n x dk, k: n x dk, v: n x dv

    output: attention value a: nxdv
    '''
    d = q.shape[-1]
    s = Softmax(dim=2)
    self_attention = torch.matmul(s(torch.matmul(q, torch.transpose(k, 1, 2))/np.sqrt(d)), v)
    return self_attention


batch_size = 2
n = 18  # number of sequences
dk = 8
dv = 10
q = torch.rand(batch_size, n, dk)
k = torch.rand(batch_size, n, dk)
v = torch.rand(batch_size, n, dv)
# attention_custom = custom_sdp_attention(q, k, v)
# attention_pytorch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
# print(torch.allclose(attention_custom, attention_pytorch))
multihead_attention(3, q, k, v)
