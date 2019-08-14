import torch

def expandEmbeddingByN(emb:torch.nn.Embedding, expand_size=4, word_padding_idx=1 ,last=False):
    if expand_size <1:
        emb.padding_idx=word_padding_idx
        return emb

    original_size=emb.weight.size()
    new_emb=torch.nn.Embedding(original_size[0]+expand_size, original_size[1], padding_idx=word_padding_idx)
    original_data=emb.weight.data
    expand_data=torch.nn.Embedding(expand_size, original_size[1]).weight.data

    if last:
        data=torch.cat( [original_data, expand_data] )
    else:
        data=torch.cat( [expand_data, original_data] )
    
    new_emb.weight.data.copy_(data)

    return new_emb
