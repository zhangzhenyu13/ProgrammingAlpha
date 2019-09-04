# -*- UTF-8 -*-
import json
import os
from typing import Optional, Awaitable

from tornado.web import RequestHandler

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.Utility import getLogger

logger = getLogger(__name__)

# 独立的检索服务
class SearchHandler(RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def initialize(self, config_args):
        self.args = config_args
        logger.info("{} server initing".format(self.args.ServiceName))

    def prepare(self):
        pass

    def get(self):
        query_argument = json.loads(self.request.body)
        title = ''
        body = ''
        tag_list = ''
        size = self.args.doc_size
        if 'Title' in query_argument:
            title = query_argument.get('Title')
        if 'Body' in query_argument:
            body = query_argument.get('Body')
        if 'Tags' in query_argument:
            tag_list = query_argument.get('Tags')
        if 'size' in query_argument:
            size = query_argument.get('size')

        if title == '' and body == '':
            logger.error("Parameters Error: You must <b>Title</b> and <b>Body</b> parameters!")
            return

        query = Query(title=title, body=body, tag_list=tag_list, num_works=self.args.num_works)
        query.search(url=self.args.es_url, size=size)
        query.arrange()

        results = query.get_origin_results()
        results = json.dumps(results)
        self.set_header('Content-type', 'application/json')
        self.write(results)


    def post(self):
        self.get()

    def on_finish(self):
        pass