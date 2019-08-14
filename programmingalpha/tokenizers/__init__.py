from .tokenizer import CoreNLPTokenizer,SpacyTokenizer, Seq2SeqAdapterTokenizer, RoBertaTokenizer
import os
import programmingalpha

def ngrams(words, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        if uncased:
            words =tuple(map(lambda s:s.lower(),words))

        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

def get_tokenizer(model_path=None, name="bert"):
    tokenizer=None

    if name=="bert":
        from pytorch_transformers import BertTokenizer
        tokenizer=BertTokenizer.from_pretrained(model_path)
        tokenizer=Seq2SeqAdapterTokenizer(tokenizer)
    if name=="gpt2":
        from pytorch_transformers import GPT2Tokenizer
        tokenizer=GPT2Tokenizer.from_pretrained(model_path)
        tokenizer=Seq2SeqAdapterTokenizer(tokenizer)
    if name=="xlnet":
        from pytorch_transformers import XLNetTokenizer
        tokenizer= XLNetTokenizer.from_pretrained(model_path)
        tokenizer=Seq2SeqAdapterTokenizer(tokenizer)
    if name=="roberta":
        tokenizer=RoBertaTokenizer(model_path)

    if tokenizer is  None:
        raise RuntimeError("tokenizer:{} is not supported!".format(name))
    
    return tokenizer


def tokenizeFolder(tokenizer, folder:str,
                    folder_tok:str):
    os.makedirs(folder_tok, exist_ok=True)
    for file in os.listdir(folder):
        raw_txt_file = folder + file  # os.path.join(folder, file)
        tok_txt_file = folder_tok + file  # os.path.join(folder_tok, file)

        with open(raw_txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            add_sp = "target" in raw_txt_file
            lines_tok = map(lambda line: tokenizer.tokenizeLine(line, add_sp=add_sp) + "\n", lines)
            with open(tok_txt_file, "w", encoding="utf-8") as f:
                f.writelines(lines_tok)
