import json
import pexpect
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#general tokenizer interface
class Tokenizer(object):
    def tokenizeLine(self):
        raise NotImplementedError
    def encode(self,text, **kwargs):
        raise NotImplementedError
    def decode(self,ids, **kwargs):
        raise NotImplementedError

class CoreNLPTokenizer(object):

    def __init__(self):
        """
        Args:
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        #for file in `find /home/zhangzy/stanford-corenlp-full-2018-10-05/ -name "*.jar"`; do export
        #CLASSPATH="$CLASSPATH:`realpath $file`"; done
        import os
        path="/home/LAB/zhangzy/stanford-corenlp-full-2018-10-05/"
        files=os.listdir(path)
        jars=[]
        for f in files:
            if f[-4:]==".jar":
                jars.append(os.path.join(path,f))

        self.classpath = ":".join(jars)
        self.mem = "4g"
        self._launch()
        logger.info("init core_nlp tokenizer finished!")

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']

        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            return [token]

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{\r\n  "sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        tokens = tuple([self._convert(t["word"]) for s in output['sentences'] for t in s['tokens']])
        #tokens = tuple([t["word"] for s in output['sentences'] for t in s['tokens']])

        return tokens


import spacy

class SpacyTokenizer(object):

    def __init__(self):
        """
        Args:
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = 'en'
        nlp_kwargs = {'parser': False}
        nlp_kwargs['tagger'] = False
        nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)
        logger.info("init spacy tokenizer finished!")

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = tuple(map(lambda t:t.text,self.nlp.tokenizer(clean_text)))

        return tokens


class SimpleTokenizer(object):
    def tokenize(self,txt):
        return txt.split()


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

        return bpe_sentence

    

if __name__ == '__main__':
    tokenizer1=CoreNLPTokenizer()
    tokenizer2=SpacyTokenizer()
    
    s="I am a very powerful! (greatest) man"
    print(tokenizer1.tokenize(s))
    print(tokenizer2.tokenize(s))
    from programmingalpha.tokenizers import ngrams
    for n in range(1,5):
        print(ngrams(tokenizer2.tokenize(s),n=n))
