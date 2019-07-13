# -*- UTF-8 -*-
from typing import Optional, Awaitable

from tornado.web import RequestHandler
from tornado.web import Application
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
import regex as re
import json

from programmingalpha.DocSearchEngine.entity.post import PostJSONEncoder, PostJSONEncoder2
from programmingalpha.DocSearchEngine.entity.query import Query


# 独立的检索服务
class SearchHandler(RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def prepare(self):
        pass

    def get(self):
        title = self.get_query_argument('title', default='')
        body = self.get_query_argument('body', default='')
        created_date = self.get_query_argument('created_date', default='')
        tags = self.get_query_argument('tags', default='')
        tag_list = re.findall('(\<[^ \>]+\>)', tags)
        size = self.get_query_argument('size', default=10)

        if title == '' and body == '':
            self.write("Please input <b>title</b> and <b>body</b> parameters!")
            return

        query = Query(title=title, body=body, tag_list=tag_list, created_date=created_date)
        query.search(size=size)
        query.arrange()
        post_results = query.get_results()
        results = json.dumps(post_results, cls=PostJSONEncoder)
        self.set_header('Content-type', 'application/json')
        self.write(results)

    def post(self):
        self.get()

    def on_finish(self):
        pass


# 为AnsAlpha封装的服务
# 返回的数据是：{"posts":[{"Id":1,"text":"hi, world!"},{"Id":2,"text":"How is it?"}],"question":"Some greetings?"}
class KnowAlphaHandler(RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def prepare(self):
        pass

    def get(self):
        title = self.get_query_argument('title', default='')
        size = self.get_query_argument('size', default=100)

        if title == '':
            self.write("Please input <b>title</b> parameters!")
            return

        query = Query(title=title)
        query.search(size=size)
        query.arrange()
        post_results = query.get_results()
        
        post_json_list = []
        for obj in post_results:
            all_text = obj.question_obj.title + ' ' + obj.question_obj.parsed_body
            for i, ans_obj in enumerate(obj.answer_obj_list):
                if i == 3:
                    break
                all_text = all_text + ' ' + ans_obj.parsed_body

            dic = {'id': obj.question_obj.es_id, 'text': all_text}
            post_json_list.append(dic)
        
        results = {}
        results['posts'] = post_json_list
        results['question'] = title
        self.set_header('Content-type', 'application/json')
        self.write(results)

    def post(self):
        self.get()

    def on_finish(self):
        pass
