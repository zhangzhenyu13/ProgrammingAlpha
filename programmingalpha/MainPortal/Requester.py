import requests
import json
from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
import logging
from programmingalpha.Utility.processCorpus import E2EProcessor, PairProcessor
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


        self.e2e_processor=E2EProcessor(config.global_config)

        self.pair_processor=PairProcessor(config.global_config)

        logger.info("main portal: requester services loaded")

    def requestDocService(self, question):
        return self.doc_searcher_portal.request(question)

    def requestKnowService(self, docs_list):
        return self.know_alpha_portal.request(docs_list)

    def requestAnswerService(self, qc_data):
        return self.answer_alpha_portal.request(qc_data)


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


        #query doc searcher
        docs_list=self.requestDocService(question)
        print("docs_list-->",docs_list[0].keys(),docs_list)

        #re-strcut question
        #question["Title"]=question["title"]
        #question["Body"]=question["body"]
        #question["Tags"]=question["tag_list"]

        id=0
        for doc in docs_list:
            doc["Id"]=id
            id+=1

        #query doc ranker
        rank_query=self.pair_processor.process(question, docs_list)

        ranks_data=self.requestKnowService(rank_query)

        docs={doc["Id"]:doc for doc in docs_list}

        useful_posts=[]
        for rank in ranks_data:
            post=docs[rank["Id"]]
            useful_posts.append(post)

        #query answer alpha
        res=self.e2e_processor.process(question, useful_posts)
        question=res["question"]
        context=res["context"]

        qc_data=[
            {
                "id":0,
                "src": " ".join( [question, "[SEP]", context] )
            }
            ]

        answer=self.requestAnswerService(qc_data)

        return answer, useful_posts
