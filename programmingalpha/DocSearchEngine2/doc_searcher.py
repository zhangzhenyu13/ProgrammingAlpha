from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
import json

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

class DocSearcherHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self, config_file)
        logger.info("{} server initing".format(self.args.ServiceName))

    def processCore(self, data):
        print("*************************************begin search1")

        title = data["Title"]
        body = data["Body"]
        tag_list = data["Tags"]
        size = data["size"]
        query = Query(title=title, body=body, tag_list=tag_list)
        print("*************************************begin search2")
        query.search(size=size)
        query.arrange()

        results = query.get_origin_results()
        return results
