from programmingalpha.alphaservices.HTTPServers.flask_http4doc_searcher import AlphaHTTPProxy4DocSearcher
import json

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

class DocSearcherHTTPProxy(AlphaHTTPProxy4DocSearcher):
    def __init__(self, config_file):
        super().__init__(config_file)
        logger.info("{} server initing".format(self.args.ServiceName))

    def processCore(self, data):
        print("*************************************begin search1")
        title = data["Title"]
        body = data["Body"]
        tag_list = data["Tags"]
        size = data["size"]
        query = Query(title=title, body=body, tag_list=tag_list, num_works=self.args.num_works)
        print("*************************************begin search2")
        query.search(size=size)
        query.arrange()
        results = query.get_origin_results()

        print("-------", len(results), type(results))
        return results
