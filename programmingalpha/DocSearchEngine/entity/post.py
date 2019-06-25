# -*- UTF-8 -*-
import json

from programmingalpha.DocSearchEngine.util.preprocessor import PreprocessPostContent


# Json 序列化(For SearchHandler)
class PostJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Post):
            ans_ls = []
            for ans_obj in obj.answer_obj_list:
                ans_ls.append(ans_obj.to_dict())
            dic = {'question_obj': obj.question_obj.to_dict(), 'answer_obj_list': ans_ls}
            return dic
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


# Json 序列化(For AnsAlphaHandler)
class PostJSONEncoder2(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Post):
            all_text = obj.question_obj.title + ' ' + obj.question_obj.parsed_body
            for i, ans_obj in enumerate(obj.answer_obj_list):
                if i == 3:
                    break
                all_text = all_text + ' ' + ans_obj.parsed_body

            dic = {'id': obj.question_obj.es_id, 'text': all_text}
            return dic
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class Post:

    def __init__(self, question_obj, answer_obj_list=None):
        self.question_obj = question_obj
        self.answer_obj_list = answer_obj_list
        self.title_relevance = 0
        self.tag_relevance = 0
        self.question_tfidf = 0
        self.answer_tfidf = 0
        self.score = 0
        self.code_relevance = 0
        self.all_score = 0

    def concat_answer_body(self):
        answer_body_list = []
        for answer_obj in self.answer_obj_list:
            answer_body_list.append(answer_obj.parse_body())

        return " ".join(answer_body_list)

    def get_question_body_code(self):
        code_snippet_list = PreprocessPostContent().get_single_code(self.question_obj.body)
        single_code_list = []
        for code_snippet in code_snippet_list:
            code_list = code_snippet.split()
            if len(code_list) == 1:
                single_code_list.extend(code_list)

        return single_code_list

    def get_answer_body_code(self):
        answer_body_list = []
        for answer_obj in self.answer_obj_list:
            answer_body_list.append(answer_obj.body)

        code_snippet_list = PreprocessPostContent().get_single_code(" ".join(answer_body_list))
        single_code_list = []
        for code_snippet in code_snippet_list:
            code_list = code_snippet.split()
            if len(code_list) == 1:
                single_code_list.extend(code_list)

        return single_code_list

    def set_title_relevance(self, title_relevance):
        self.title_relevance = title_relevance

    def set_tag_relevance(self, tag_relevance):
        self.tag_relevance = tag_relevance

    def set_question_body_tfidf(self, tfidf):
        self.question_tfidf = tfidf

    def set_answer_body_tfidf(self, tfidf):
        self.answer_tfidf = tfidf

    def set_score(self, score):
        self.score = score

    def set_code_relevance(self, code_relevance):
        self.code_relevance = code_relevance

    def cal_all_score(self):
        ls = [self.title_relevance, self.tag_relevance, self.question_tfidf, self.answer_tfidf, self.score,
              self.code_relevance, self.all_score]
        self.all_score = sum(ls)

    def __gt__(self, other):
        return self.all_score > other.all_score
