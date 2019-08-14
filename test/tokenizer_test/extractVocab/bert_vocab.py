from pytorch_transformers import BertTokenizer
import os
import programmingalpha
from programmingalpha.tokenizers.tokenizer import Seq2SeqAdapterTokenizer

def extractVocab(model_path = programmingalpha.BertLargeUnCased):
    tokenizer=BertTokenizer.from_pretrained(model_path )
    tokenizer=Seq2SeqAdapterTokenizer(tokenizer)
    with open(os.path.join(model_path,"vocab.txt"), "r", encoding="utf-8") as f:
        vocab=map(lambda w: w.strip(), f.readlines() )

    vocab_file = os.path.join(programmingalpha.BertRoot,"vocab.txt")

    tokens = []
    id = 0

    for word in vocab:
        print(word, id, tokenizer.decode([id]) )   
        tokens.append(str(id))
        id += 1

    with open(vocab_file, "w", encoding="utf-8") as f:
        tokens = map(lambda w: w + "\n", tokens)
        f.writelines(tokens)
        print("write vocab.txt")

if __name__ == '__main__':
    extractVocab()
