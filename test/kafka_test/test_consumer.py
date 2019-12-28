from kafka import KafkaConsumer
from kafka.errors import KafkaError

consumer=KafkaConsumer('test', bootstrap_servers=['10.1.1.1:9092'],auto_offset_reset='earliest')
for msg in consumer:
    print("{}:{}:{}: key={}, value={}".format(
        msg.topic, msg.partition, msg.offset, msg.key, msg.value
    ))
