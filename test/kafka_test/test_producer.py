from kafka import KafkaProducer
from kafka.errors import KafkaError
import json

servers=['10.1.1.1:9092']
def on_send_success(record_matadata):
    print(record_matadata.topic)
    print(record_matadata.partition)
    print(record_matadata.offset)

def on_send_error(excp):
    print("error:", excp.args)
    excp.with_traceback()
    
#producer 1
producer=KafkaProducer(bootstrap_servers=servers)
future=producer.send('test', b'hello, kafka!')

try:
    record_matadata=future.get(timeout=10)
    print(record_matadata.topic)
    print(record_matadata.partition)
    print(record_matadata.offset)
except KafkaError as e:
    print(e.args)
    print(e)


producer.send('test', key=b'foo', value=b'bar').add_callback(on_send_success).add_errback(on_send_error)
#producer.close()

producer=KafkaProducer(bootstrap_servers=servers,value_serializer= lambda m: json.dumps(m).encode('ascii') )

producer.send('test', {'hi':'to you'}).add_callback(on_send_success).add_errback(on_send_error)

producer.flush()
