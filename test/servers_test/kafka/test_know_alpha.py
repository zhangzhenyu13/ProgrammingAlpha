from programmingalpha.retrievers.retriever_on_kafka import KnowAlphaKafkaNode

config_file="kafka/knowAlphaService.json"
server=KnowAlphaKafkaNode(config_file)

server.start()