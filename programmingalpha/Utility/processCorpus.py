from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
import json
from programmingalpha import AlphaConfig, AlphaPathLookUp
from programmingalpha.tokenizers import get_tokenizer
import programmingalpha
import os
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

class CorpusProcessor(object):
    def __init__(self, config_file):
        
        self.config = AlphaConfig.loadConfig(
            os.path.join(AlphaPathLookUp.ConfigPath, config_file)
        )
        
        self.tokenizer = get_tokenizer(name=self.config.tokenizer, model_path=self.config.model_path)
        
        self.textExtractor=InformationAbstrator(maxClip=100, tokenizer=None)
        self.textExtractor.initParagraphFilter(self.textExtractor.lexrankSummary)
        
    def _getPreprocess(self,txt, maxLen):
        textExtractor = self.textExtractor

        textExtractor.maxClip = maxLen
        txt_processed = textExtractor.clipText(txt)

        return " ".join(txt_processed)


class E2EProcessor(CorpusProcessor):

    def __init__(self, config_file):
        CorpusProcessor.__init__(self, config_file)


        config_answer_alpha = self.config.answer_alpha

        self.question_len = config_answer_alpha["question_len"]
        self.context_len = config_answer_alpha["context_len"]


    def process(self, question:json, relatives:list):

        textExtractor=self.textExtractor
        title=" ".join( textExtractor.tokenizer.tokenize(question["Title"]) )
        body=question["Body"]

        question,_=self._getPreprocess(body, self.question_len)

        question=title+" "+question

        context=[]
        for post in relatives:

            if "answers" not in post:
                continue

            answers=post["answers"]
            if len(answers) ==0:
                continue

            ans_txt=answers[0]["Body"]
            context.append(ans_txt)


        context=" ".join(context)

        context,_=self._getPreprocess(context, self.context_len)


        record={"question":question,"context":context}

        return record



