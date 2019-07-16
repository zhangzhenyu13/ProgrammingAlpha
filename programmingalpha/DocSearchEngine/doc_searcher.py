# coding=utf-8
import json

from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
from programmingalpha.DocSearchEngine.entity.post import PostJSONEncoder2
from programmingalpha.DocSearchEngine.entity.query import Query


class DocSearcherHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self, config_file)

    def processCore(self, data):

        title = data["title"]
        size = data["size"]

        query = Query(title=title)
        query.search(size=size)
        query.arrange()
        post_results = query.get_results()

        post_json_list = json.dumps(post_results, cls=PostJSONEncoder2)
        results = {}
        results['posts'] = post_json_list
        results['question'] = title

        return results
