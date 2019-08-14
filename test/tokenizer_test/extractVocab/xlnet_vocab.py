from pytorch_transformers import XLNetTokenizer
import os
import programmingalpha
from programmingalpha.tokenizers.tokenizer import Seq2SeqAdapterTokenizer

def extractVocab(model_path = programmingalpha.XLNetBaseCased):
    tokenizer=XLNetTokenizer.from_pretrained(model_path )
    tokenizer=Seq2SeqAdapterTokenizer(tokenizer)
    vocab_file = os.path.join(model_path,"vocab.txt")

    tokens = []
    id = 0
    while id < 32000:
        try:
            word = tokenizer.decode([id])
        except:
            print(id, "exceeded!")
            break

        print(word, id)

        tokens.append(str(id))

        id += 1


    with open(vocab_file, "w", encoding="utf-8") as f:
        tokens = map(lambda w: w + "\n", tokens)
        f.writelines(tokens)
        print("**"*20)
        print("write vocab.txt")

if __name__ == '__main__':
    extractVocab()