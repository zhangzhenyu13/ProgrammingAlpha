from programmingalpha.Utility.processCorpus import CorpusProcessor

class E2EProcessor(CorpusProcessor):

    def __init__(self, config_file):
        CorpusProcessor.__init__(self, config_file)

        self.question_len = self.config.question_len
        self.context_len = self.config.context_len



    def processEnc(self, question, relevant_posts):

        textExtractor=self.textExtractor
        title=" ".join( textExtractor.tokenizer.tokenize(question["Title"]) )
        body=question["Body"]

        question = self._getPreprocess(body, self.question_len)

        question=title+" "+question

        context=[]
        for post in relevant_posts:

            answers=self._getBestAnswers(post)
            if not answers:
                continue
            ans_txt=answers[0]["Body"]
            context.append(ans_txt)


        context=" ".join(context)

        context = self._getPreprocess(context, self.context_len)


        question_words=self.tokenizer.tokenizeLine(question).split()
        context_words=self.tokenizer.tokenizeLine(context).split()

        
        question_words=question_words[:self.question_len]
        context_words=context_words[:self.context_len]

        q_c_text=" ".join(question_words+context_words)


        return q_c_text


    def processDec(self, ids):
        if type(ids)==str:
            ids=ids.split()
            ids=map(lambda id: int(id), ids)
            ids=list(ids)
            
        text=self.tokenizer.decode(ids)
        return text