from programmingalpha.alphaservices.HTTPServers.tornado_http import AlphaHTTPProxy
import json

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

class DocSearcherHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        super().__init__(config_file)
        logger.info("{} server initing".format(self.args.ServiceName))

    def processCore(self, data):
        title = ''
        body = ''
        tag_list = ''

        size = self.args.doc_size
        if 'Title' in data:
            title = data.get('Title')
        if 'Body' in data:
            body = data.get('Body')
        if 'Tags' in data:
            tag_list = data.get('Tags')
        if 'size' in data:
            size = data.get('size')

        if title == '' and body == '':
            logger.error("Parameters Error: You must <b>Title</b> and <b>Body</b> parameters!")
            return

        query = Query(title=title, body=body, tag_list=tag_list, num_works=self.args.num_works)
        query.search(url=self.args.es_url, size=size)
        query.arrange()
        results = query.get_origin_results()

        return results
