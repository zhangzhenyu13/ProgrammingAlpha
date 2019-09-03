from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
import json
from programmingalpha import AlphaConfig, AlphaPathLookUp
from programmingalpha.tokenizers import get_tokenizer
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


    def _getBestAnswers(self,post):
        if "answers" not in post:
            return []

        answers=post["answers"]
        if len(answers) ==0:
            return [] 

        if "AcceptedAnswerId" in post:
            ans= answers[-1]
            del answers[-1]
            answers.insert(0, ans)

        return answers
