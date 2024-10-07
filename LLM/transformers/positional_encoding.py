import numpy as np
import torch


def positional_encoding(n_tokens, d):
    '''
    computing positional encoding for position pos of n_tokens number 
    of tokens, each token has embedding of dimention d
    PE(pos, 2i) = sin(pos/1000^2i/dmodel)
    PE(pos, 2i+1) = cos(pos/1000^2i/dmodel)
    '''
    all_PE = []
    for pos in range(n_tokens):
        PE = []
        for i in range(d):
            if i%2 == 0:
                PE.append(np.sin(pos/(1000)**((2*i)/d)))
            else:
                PE.append(np.cos(pos/(1000)**((2*i)/d)))
        all_PE.append(PE)
    all_PE = torch.tensor(np.array(all_PE))
    return all_PE

