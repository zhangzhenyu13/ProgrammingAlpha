# -*- UTF-8 -*-

from programmingalpha.DocSearchEngine2.entity.answer import Answer
from programmingalpha.DocSearchEngine2.entity.post import Post
from programmingalpha.DocSearchEngine2.entity.question import Question
from programmingalpha.DocSearchEngine2.retriever import search_es
from programmingalpha.DocSearchEngine2.util import tokenizer
from programmingalpha.DocSearchEngine2.util.preprocessor import PreprocessPostContent
from multiprocessing import Pool
import numpy as np


class Query(object):
    def __init__(self, title, body='', tag_list=None, num_works=2):
        self.title = title
        self.body = body
        self.tag_list = tag_list
        self.created_date = ""
        self.searched_post_list = []
        # 多线程(在flask wsgi server上会出现问题)
        self.num_works = num_works
        # self.processes = Pool(num_works)

    def get_results(self):
        return self.searched_post_list

    def get_origin_results(self):
        origin_results = [posts.origin_source for posts in self.searched_post_list]
        return origin_results

    def search(self, url, size):
        search_result_list = search_es.search(self.title, url, size)
        # Post list
        post_obj_list = []
        for result in search_result_list:
            result_id = result['_id']
            result_source = result['_source']

            # Question
            question = Question(result_id, result_source['Title'], result_source['Body'],
                                result_source['CommentCount'], result_source['Score'],
                                result_source['Tags'], result_source['CreationDate'])
            # Answer list
            # body, created_date, score=0, comment_count=0
            answers = result_source['answers']
            answer_list = []
            for answer in answers:
                # body, created_date, score=0, comment_count=0)
                answer = Answer(answer['Body'], answer['CreationDate'], answer['Score'], answer['CommentCount'])
                answer_list.append(answer)

            # Add Post to post list
            post_obj_list.append(Post(question, answer_list, result_source))

        self.searched_post_list = post_obj_list
        # 使用ES返回的 _score 值
        for i in range(len(search_result_list)):
            es_tfidf = search_result_list[i]['_score']
            self.searched_post_list[i].set_question_body_tfidf(es_tfidf)

    def parse_body(self):
        processor = PreprocessPostContent()
        body_para_list = processor.getProcessedParagraphs(self.body)
        body = " ".join(body_para_list)
        return body

    def __calculate_a_title_relevance(self, question_obj):
        query_title_word_list = tokenizer.tokenize(self.title)
        if len(query_title_word_list) == 0:
            return 0

        question_title_word_list = tokenizer.tokenize(question_obj.title)
        # lower
        question_title_word_list = [w.lower() for w in question_title_word_list]
        query_title_word_list = [w.lower() for w in query_title_word_list]

        overlap = [value for value in query_title_word_list if value in question_title_word_list]
        ret = len(overlap) / len(query_title_word_list)

        return ret

    def calculate_title_relevance(self):
        with Pool(self.num_works) as processes:
            processes.map_async(self.__calculate_a_title_relevance, self.searched_post_list)

        # for post in self.searched_post_list:
        # post.set_title_relevance(self.__calculate_a_title_relevance(post.question_obj))

    def __calculate_a_tag_relevance(self, question_obj):
        if len(self.tag_list) == 0:
            return 0

        overlap = [value for value in self.tag_list if value in question_obj.tag_list]
        ret = len(overlap) / len(self.tag_list)

        return ret

    def calculate_tag_relevance(self):
        with Pool(self.num_works) as processes:
            processes.map_async(self.__calculate_a_tag_relevance, self.searched_post_list)
        # for post in self.searched_post_list:
        #     post.set_tag_relevance(self.__calculate_a_tag_relevance(post.question_obj))

    def __calculate_a_score(self, post_obj, alpha=0.8):
        # 因为调用了ES TFIDF 值，所以要调用question和answer的 parse_body()
        post_obj.question_obj.parse_body()  # question的 parse_body()
        comment_count = post_obj.question_obj.comment_count
        vote_score = post_obj.question_obj.score
        for answer_obj in post_obj.answer_obj_list:
            answer_obj.parse_body()  # answer的 parse_body()
            comment_count += answer_obj.comment_count
            vote_score += answer_obj.score

        score = (1 - alpha) * comment_count + alpha * vote_score
        return score

    def calculate_score(self, alpha=0.8):
        with Pool(self.num_works) as processes:
            processes.map_async(self.__calculate_a_score, self.searched_post_list)
        # for post in self.searched_post_list:
        #     post.set_score(self.__calculate_a_score(post, alpha))

    def __get_body_code(self):
        code_snippet_list = PreprocessPostContent().get_single_code(self.body)
        single_code_list = []
        for code_snippet in code_snippet_list:
            code_list = code_snippet.split()
            if len(code_list) == 1:
                single_code_list.extend(code_list)

        return single_code_list

    def __calculate_a_code_relevance(self, post_obj):
        query_code_list = self.__get_body_code()
        if len(query_code_list) == 0:
            return 0

        post_code_list = []
        post_code_list.extend(post_obj.get_question_body_code())
        post_code_list.extend(post_obj.get_answer_body_code())

        overlap = [code for code in query_code_list if code in post_code_list]
        code_relevance = len(overlap) / len(query_code_list)

        return code_relevance

    def calculate_code_relevance(self):
        for post in self.searched_post_list:
            code_relevance = self.__calculate_a_code_relevance(post)
            post.set_code_relevance(code_relevance)

    # Standard normalization
    def __normalize(self, score_list):
        return np.divide(np.subtract(score_list, np.average(score_list)), (np.std(score_list) + 0.0001))

    def normalized_post_score(self):
        title_relevance_list = []
        tag_relevance_list = []
        question_tfidf_list = []
        answer_tfidf_list = []
        score_list = []
        code_relevance_list = []

        for searched_post in self.searched_post_list:
            title_relevance_list.append(searched_post.title_relevance)
            tag_relevance_list.append(searched_post.tag_relevance)
            question_tfidf_list.append(searched_post.question_tfidf)
            answer_tfidf_list.append(searched_post.answer_tfidf)
            score_list.append(searched_post.score)
            code_relevance_list.append(searched_post.code_relevance)

        title_relevance_list = self.__normalize(title_relevance_list)
        tag_relevance_list = self.__normalize(tag_relevance_list)
        question_tfidf_list = self.__normalize(question_tfidf_list)
        answer_tfidf_list = self.__normalize(answer_tfidf_list)
        score_list = self.__normalize(score_list)
        code_relevance_list = self.__normalize(code_relevance_list)

        for i in range(len(self.searched_post_list)):
            self.searched_post_list[i].title_relevance = title_relevance_list[i]
            self.searched_post_list[i].tag_relevance = tag_relevance_list[i]
            self.searched_post_list[i].question_tfidf = question_tfidf_list[i]
            self.searched_post_list[i].answer_tfidf = answer_tfidf_list[i]
            self.searched_post_list[i].score = score_list[i]
            self.searched_post_list[i].code_relevance = code_relevance_list[i]

    def calculate_posts_all_score(self):
        for post in self.searched_post_list:
            post.cal_all_score()

    def arrange(self):
        self.calculate_title_relevance()
        self.calculate_code_relevance()
        self.calculate_tag_relevance()
        self.calculate_score()
        self.normalized_post_score()
        self.calculate_posts_all_score()
        self.searched_post_list = sorted(self.searched_post_list, reverse=True)


if __name__ == '__main__':
    tag_list1 = ['<Java>', '<java>', '<println>']
    tag_list2 = ['<c++>', '<java>', '<python>', 'pycharm']
    tag_list3 = ['<c++>', '<JAVA>', '<python>', 'pycharm']

    query = Query("How to use println in java", "Please show me how to use <code>println()<code> in java", tag_list1,)
    query.search(url="http://10.1.1.9:9266", size=2)
    query.arrange()
    for pos in query.searched_post_list:
        print(pos.question_obj.title)

    query.search(url="http://10.1.1.9:9266", size=2)
    query.arrange()

    results = query.get_origin_results()
    for result in results:
        print(result)