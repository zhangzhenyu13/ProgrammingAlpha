# coding=utf-8
import json

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy


class DocSearcherHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self, config_file)

    def processCore(self, data):

        title = data["Title"]
        body = data["Body"]
        tag_list = data["Tags"]
        size = data["size"]
        query = Query(title=title, body=body, tag_list=tag_list)

        query.search(size=size)
        query.arrange()

        results = query.get_origin_results()
        return results
