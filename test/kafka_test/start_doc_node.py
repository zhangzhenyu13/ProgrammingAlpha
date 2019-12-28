#from programmingalpha.DocSearchEngine2.service.http_server import search_server
from programmingalpha.DocSearchEngine2.doc_searcher_on_kafka import DocSearcherKafkaNode
if __name__ == '__main__':
    config_file = "docSearcherService.json"
    server = DocSearcherKafkaNode(config_file)
    #print("init server")
    server.start()
    #search_server(config_file)