from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
import json
from programmingalpha import loadConfig
from programmingalpha.tokenizers import BertTokenizer
import programmingalpha
import os

never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[NUM]", "[CODE]")


class CorpusProcessor(object):
    def __init__(self, config_file):

        self.textExtractor=InformationAbstrator(maxClip=100, tokenizer=BertTokenizer(
            programmingalpha.ModelPath + "answerNets/vocab.txt", never_split=never_split)
        )
        self.config = loadConfig(
            os.path.join(programmingalpha.ConfigPath, config_file)
        )


    def _getPreprocess(self,txt, maxLen, cal_lose=False):
        textExtractor = self.textExtractor

        textExtractor.maxClip = maxLen
        if cal_lose:
            original = " ".join(textExtractor.processor.getPlainTxt(txt))
            before_len = len(original.split())
            if before_len < 5:
                # logger.info("bad zero:{}\n=>{}".format(txt,original))
                return "", 0
        txt_processed = textExtractor.clipText(txt)

        if cal_lose:
            after_len = len(" ".join(txt_processed).split())
            lose_rate = after_len / before_len
            return " ".join(txt_processed), lose_rate
        else:
            return " ".join(txt_processed), None


class E2EProcessor(CorpusProcessor):

    def __init__(self, config_file):
        CorpusProcessor.__init__(self, config_file)

        self.question_len=150
        self.context_len=1350

        config_answer_alpha = self.config.answer_alpha

        self.question_len = config_answer_alpha["question_len"]
        self.answer_len = config_answer_alpha["answer_len"]


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


class PairProcessor(CorpusProcessor):
    def __init__(self, config_file):
        CorpusProcessor.__init__(self,config_file)
        self.question_len=300
        self.post_len=1200
        self.answers=3
        #print(config_file, self.config, self.config)
        know_alpha_config = self.config.know_alpha

        self.question_len = know_alpha_config["question_len"]
        self.answers = know_alpha_config["answers"]
        self.post_len = know_alpha_config["post_len"]



    def process(self, question, posts):
        docs_list=[]
        textExtractor=self.textExtractor

        id=0
        for post in posts:

            title = " ".join(textExtractor.tokenizer.tokenize(post["question"]["Title"]))

            body,_ =self._getPreprocess(post["question"]["Body"],self.question_len)

            answers=list(
                map(lambda ans: ans["Body"], post["answers"])
            )[:self.answers]

            answers_txt,_ = self._getPreprocess(" ".join(answers), self.post_len)

            docs_list.append({
                "Id":post["Id"],
                "text": " ".join([title, body, answers_txt])
            })
            id+=1

        question_title = " ".join(textExtractor.tokenizer.tokenize(question["Title"]))
        question_body,_ = self._getPreprocess(question["Body"], self.question_len)
        record={"posts": docs_list, "question": " ".join([question_title, question_body])}

        return record

