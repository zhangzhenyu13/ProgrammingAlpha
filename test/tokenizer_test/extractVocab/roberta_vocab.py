from fairseq.data import encoders
import os
import programmingalpha 
from programmingalpha.tokenizers.tokenizer import OnmtDictionary
import torch

def extractVocab(model_path = programmingalpha.RoBertaBase):
    ckpt = torch.load(os.path.join(model_path, "model.pt"), map_location='cpu')
    args = ckpt["args"]
    for file, arg in {
        'code': 'bpe_codes',
        'bpecodes': 'bpe_codes',
        'sentencepiece.bpe.model': 'sentencepiece_vocab',
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            # kwargs[arg] = path
            setattr(args, arg, path)

    setattr(args, "bpe", "gpt2")
    bpe=encoders.build_bpe(args)
    
    with open(os.path.join(model_path, "dict.txt"), "r") as f:
        dictionary=OnmtDictionary.load(f)

    vocab=[]
    for word in dictionary.symbols:
        try:
            if word in dictionary.symbols[:dictionary.nspecial]:
                vocab.append(word)
                print("sp tokens:", word)
                continue
            print(word, "--->", bpe.decode(word),"--->", dictionary.index(word) )
            vocab.append(word )
        except:
            print("decoding error, append to vocab directly")
            vocab.append(word)
    print(len(vocab), vocab[:5], vocab[-5:])

    with open(os.path.join(model_path, "vocab.txt",), "w", encoding="utf-8") as f:
        f.writelines(map(lambda w: w+"\n", vocab) )
        print("**"*20)
        print("wirte vocab.txt")


if __name__ == '__main__':
    extractVocab()