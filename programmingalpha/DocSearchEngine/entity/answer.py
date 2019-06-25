# -*- UTF-8 -*-
import json

from programmingalpha.DocSearchEngine.util.preprocessor import PreprocessPostContent


class Answer:

    def __init__(self, body, created_date, score=0, comment_count=0):
        self.body = body
        self.score = score
        self.comment_count = comment_count
        self.created_date = created_date
        self.parsed_body = ''

    def to_dict(self):
        dic = {'body': self.body, 'score': self.score, 'comment_count': self.comment_count,
               'created_date': self.created_date}
        return dic

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        self.parsed_body = " ".join(body_para_list)
        return self.parsed_body


if __name__ == '__main__':
    ans = Answer("body1", "2018-08-09", 55, 21)
    s = ans.toJSON()
    print(s)
