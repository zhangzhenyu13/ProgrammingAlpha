from programmingalpha.alphaservices.KafkaMPI.kafka_node import AlpahaKafkaNode

from programmingalpha.DocSearchEngine2.entity.query import Query
from programmingalpha.Utility import getLogger
logger=getLogger(__name__)

class DocSearcherKafkaNode(AlpahaKafkaNode):
    def __init__(self, config_file):
        super().__init__(config_file)
        logger.info("doc searcher node initing")

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

        
        results={
            "question":data,
            "posts":[],
            "doc_status":"Failed!"
        }

        if title == '' and body == '':
            logger.error("Parameters Error: You must <b>Title</b> and <b>Body</b> parameters!")
            return results

        try:
            query = Query(title=title, body=body, tag_list=tag_list, num_works=self.args.num_works)
            query.search(url=self.args.es_url, size=size)
            query.arrange()
            results["posts"] = query.get_origin_results()
            results["doc_status"]="Success!"
        except Exception as e:
            logger.error(e.args)

        return results
 