from programmingalpha.answer_generations.AnswerNetServices import AnswerAlphaKafkaNode

config_file="kafka/answerAlphaService.json"
server=AnswerAlphaKafkaNode(config_file)

server.start()

