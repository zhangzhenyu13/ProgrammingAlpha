import requests
import json
from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
from  programmingalpha.Utility import getLogger

logger = getLogger(__name__)


class RequesterPortal(object):
    def __init__(self,host, port):
        self.host=host
        self.port=port

    def _getPostUrl(self):
        post_url = "http://{}:{}/methodCore".format(self.host, self.port)
        return  post_url

    def request(self, post_data):
        post_url= self._getPostUrl()
        logger.info("requesting: {}".format(post_url))

        postData = json.dumps(post_data)
        response = requests.post(url=post_url, json= postData)

        results = json.loads(response.text)

        return results


class RequesterServices(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)

        config=self.args

        self.doc_searcher_portal=RequesterPortal(host=config.doc_searcher_service["host"], port=config.doc_searcher_service["port"])
        self.know_alpha_portal=RequesterPortal(host=config.know_alpha_service["host"], port=config.know_alpha_service["port"])
        self.answer_alpha_portal=RequesterPortal(host=config.answer_alpha_service["host"], port=config.answer_alpha_service["port"])
        #self.tokenizer_portal=RequesterPortal(host=config.tokenizer_service["host"], port=config.tokenizer_service["port"])


        logger.info("main portal: requester services loaded")

    def requestDocService(self, question):
        return self.doc_searcher_portal.request(question)

    def requestKnowService(self, docs_list):
        return self.know_alpha_portal.request(docs_list)

    def requestAnswerService(self, qc_data):
        return self.answer_alpha_portal.request(qc_data)

    def requestTokenizerService(self, t_data):
        return self.tokenizer_portal.request(t_data)

    def processCore(self, question):
        #data struct
        assert "Title" in question
        #question["Title"]=question["title"]

        if "Body" not in question:
            question["Body"]=""
            logger.info("no body is available")


        if "Tags" not in question:
            question["Tags"]=[]
            logger.info("no tags is available")

        if "size" not in question:
            question["size"]=100
            logger.info("no size infor is available")

        #query doc searcher
        posts_list=self.requestDocService(question)
        #logger.info("retrieved {} posts, with keywords as-->{}".format(len(posts_list),posts_list[0].keys()) )


        #query doc ranker
        rank_query={"question":question, "posts":posts_list}
        ranks_data=self.requestKnowService(rank_query)

        logger.info("found {} useful posts".format(len(ranks_data)))

        useful_posts=[]
        for rank in ranks_data:
            post=posts_list[rank["Id"]]
            useful_posts.append(post)

        
        #query answer alpha

        answer_query_data={"question":question,"posts":useful_posts}

        answer=self.requestAnswerService(answer_query_data)

        logger.info("finished generating answer")

        return {"generated-answers":answer, "useful-reading-posts":useful_posts}
