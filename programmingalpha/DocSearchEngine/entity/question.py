# -*- UTF-8 -*-
import json

from programmingalpha.DocSearchEngine.util.preprocessor import PreprocessPostContent


class Question:

    def __init__(self, es_id, title, body, comment_count, score, tag_list, created_date):
        self.es_id = es_id  # 作为 post 的 id, For AnsAlpha
        self.title = title
        self.body = body
        self.comment_count = comment_count
        self.score = score
        self.tag_list = tag_list
        self.created_date = created_date
        self.parsed_body = ''

    def to_dict(self):
        dic = {'es_id':self.es_id, 'title': self.title, 'body': self.body, 'comment_count': self.comment_count,
               'score': self.score, 'tag_list': self.tag_list, 'created_date': self.created_date}
        return dic

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        self.parsed_body = " ".join(body_para_list)
        return self.parsed_body


