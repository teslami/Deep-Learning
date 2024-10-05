import numpy as np


def positional_encoding(pos, d):
    '''
    computing positional encoding for position pos
    of embedding dimention d
    PE(pos, 2i) = sin(pos/1000^2i/dmodel)
    PE(pos, 2i+1) = cos(pos/1000^2i/dmodel)
    '''
    PE = []
    for i in range(d):
        if i%2==0:
            #print(np.sin(pos/(1000)**((2*i)/d)))
            PE.append(np.sin(pos/(1000)**((2*i)/d)))
        else:
            PE.append(np.cos(pos/(1000)**((2*i)/d)))
    return PE


