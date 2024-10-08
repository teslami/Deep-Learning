import torch
import attention_function
from torch.nn import Linear, LayerNorm
from positional_encoding import positional_encoding


class multi_head_attention_block(torch.nn.Module):
    def __init__(self, num_att_block, token_embedding, dim_hidden_linear, feedforward_dim):
        '''
        token_embedding: token embeddings of size bxnxd
            b: batch size
            n: token size
            d: embedding dim
        '''
        super(multi_head_attention_block, self).__init__()
        self.token_embedding = token_embedding
        self.num_att_block = num_att_block
        self.embedding_dim = token_embedding.shape[2]
        self.num_tokens = token_embedding.shape[1]
        self.linear_q = Linear(self.embedding_dim, dim_hidden_linear)
        self.linear_k = Linear(self.embedding_dim, dim_hidden_linear)
        self.linear_v = Linear(self.embedding_dim, dim_hidden_linear)
        self.layer_norm = LayerNorm(normalized_shape=self.embedding_dim)
        self.feedforward_linear1 = Linear(self.embedding_dim, feedforward_dim)
        self.feedforward_linear2 = Linear(feedforward_dim, self.embedding_dim)

    def forward(self):
        '''
        performs multihead attention on embedding of size b, n, m
            b: batch size
            n: number of tokens
            m: embedding length

        computing positional encoding, adding it to toekn embeddings,
        applying multihead attention

        returns new computed embedding
        '''
        embedding = self.token_embedding
        pos_encoding = positional_encoding(self.num_tokens, self.embedding_dim)
        pos_encoding = pos_encoding.expand(embedding.shape[0], -1, -1)  # duplicating this for all token embeddings
        embedding = embedding+pos_encoding  # size: bxnxd
        embedding = self.layer_norm(embedding.float())
        #  generating vectors q, k and v using three different linear transformations
        q = self.linear_q(embedding)
        k = self.linear_k(embedding)
        v = self.linear_v(embedding)
        output_multihead_attention = attention_function.multihead_attention(
            self.num_att_block,
            q, k, v)
        return output_multihead_attention


transformer_encode = multi_head_attention_block(
    num_att_block=3, token_embedding=torch.rand(2, 30, 256),
    dim_hidden_linear=512, feedforward_dim=1024)
output_embeddings = transformer_encode()