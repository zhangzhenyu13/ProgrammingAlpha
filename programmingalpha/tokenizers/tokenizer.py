from programmingalpha.Utility import getLogger
logger = getLogger(__name__)


#general tokenizer interface
class Tokenizer(object):
    def tokenizeLine(self):
        raise NotImplementedError
    def encode(self,text, **kwargs):
        raise NotImplementedError
    def decode(self,ids, **kwargs):
        raise NotImplementedError


#adpater for tokenizers of huggingface
from pytorch_transformers import PreTrainedTokenizer
class Seq2SeqAdapterTokenizer(Tokenizer):
    
    def __init__(self, tokenizer:PreTrainedTokenizer):
        self.tokenizer=tokenizer
        self.bos="<s>"
        self.eos="<\s>"
        self.unk="<unk>"
        self.pad="<pad>"
        self.__NUM="[MATH]"
        self.__CODE="[CODE]"
        spt={"_math_token":self.__NUM, "_code_token":self.__CODE}
        self.tokenizer.add_special_tokens(spt)

    def encode(self, text):
        ids=self.tokenizer.encode(text)
        ids=" ".join(map(lambda id: str(id),ids) )
        return ids
    
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):

        return  self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
    
    def tokenizeLine(self, text, add_sp=False, **kwargs):
        tokenized= self.encode(text)

        if add_sp:
            tokenized=" ".join([self.bos, tokenized, self.eos])

        return tokenized

    def  tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
#adapter for pytorch/fairseq tokenizer method
import argparse
from fairseq.data import encoders
from fairseq.tokenizer import tokenize_line
import os
import numpy as np
import nltk
from fairseq.data import Dictionary

class OnmtDictionary(Dictionary):
    def __init__(
            self,
            pad='<pad>',
            eos='</s>',
            unk='<unk>',
            bos='<s>',
            extra_special_symbols=None,
    ):
        super(OnmtDictionary, self).__init__(pad=pad, eos=eos, unk=unk, bos=bos, extra_special_symbols=extra_special_symbols)
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        #self.unk_index = self.add_symbol(unk)
        #self.pad_index = self.add_symbol(pad)
        #self.bos_index = self.add_symbol(bos)
        #self.eos_index = self.add_symbol(eos)
        self.math_word="[NUM]"
        self.code_word="[CODE]"
        self.math_index=self.add_symbol(self.math_word)
        self.code_index=self.add_symbol(self.code_word)

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

     

class RoBertaTokenizer(Tokenizer):
    def add_args(self,parser:argparse.ArgumentParser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bpe', type=str, default="gpt2",
                            help='num encoder layers')

    def __init__(self, model_path):
        parser=argparse.ArgumentParser()
        self.add_args(parser)
        args = parser.parse_args()
        print(args)

        #build bpe
        self.bpe = encoders.build_bpe(args)
        #some parameters
        self.bos="<s>"
        self.eos="<\s>"
        self.unk="<unk>"
        self.pad="<pad>"
        #load dictionary of bpe tokens
        with open(os.path.join(model_path, "dict.txt"), "r", encoding="utf-8") as f:
            self.dictionary = OnmtDictionary.load(f)


    def encode(self, sentence: str, add_sp=True, **kwargs):
        sentence=" ".join(nltk.word_tokenize(sentence) )

        bpe_sentence = self.bpe.encode(sentence)
        bpe_sentence=bpe_sentence.split()
        
        return bpe_sentence

    def decode(self, token_ids, **kwargs):
        tokens = np.array( token_ids, dtype=int)
        assert len(tokens.size())==1

        if tokens[0] == self.dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def tokenizeLine(self, sentence:str, add_sp=False) -> str:
        

        bpe_sentence = self.encode(sentence, add_sp=add_sp)

        if add_sp:
            bpe_sentence=" ".join([self.bos,bpe_sentence, self.eos])
        else:
            bpe_sentence=" ".join(bpe_sentence)
            
        return bpe_sentence

    
