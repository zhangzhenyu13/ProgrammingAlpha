from pytorch_transformers import GPT2Tokenizer
import os
import programmingalpha
from programmingalpha.tokenizers.tokenizer import Seq2SeqAdapterTokenizer

def extractVocab(model_path = programmingalpha.GPT2Base):
    tokenizer=GPT2Tokenizer.from_pretrained(model_path )
    tokenizer= Seq2SeqAdapterTokenizer(tokenizer)

    vocab_file = os.path.join(model_path,"vocab.txt")
    with open(os.path.join(model_path,"vocab.json"), "r", encoding="utf-8") as f:
        import json
        vocab=json.load(f)


    tokens = []
    id = 0
    for word in vocab.keys():
        print(word, id, tokenizer.decode([id]))
        tokens.append(str(id))
        id += 1


    with open(vocab_file, "w", encoding="utf-8") as f:
        tokens = map(lambda w: w + "\n", tokens)
        f.writelines(tokens)
        print("write vocab.txt")

if __name__ == '__main__':
    extractVocab()