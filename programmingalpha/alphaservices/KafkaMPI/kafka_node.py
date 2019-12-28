# -*- UTF-8 -*-
import os
import kafka
from kafka.errors import KafkaError
import time

from programmingalpha import AlphaConfig, AlphaPathLookUp

#from multiprocessing import Process, Event
import json
from  programmingalpha.Utility import getLogger

logger = getLogger(__name__)


class AlpahaKafkaNode(object):
    def __init__(self,config_file):
        super().__init__()
        self.args = AlphaConfig.loadConfig( os.path.join( AlphaPathLookUp.ConfigPath, config_file ) )
        #self.is_ready = Event()
        self.producer=kafka.KafkaProducer(bootstrap_servers=self.args.producer["servers"],value_serializer= lambda m: json.dumps(m).encode('ascii') )
        self.consumer=kafka.KafkaConsumer(self.args.consumer["topic"], group_id=self.args.consumer["group_id"], 
            bootstrap_servers=self.args.consumer["servers"], value_deserializer= lambda m: json.loads(m.decode("ascii")))

    def processCore(self, data):
        raise NotImplementedError
    
    def push_msg(self, msg):
        future=self.producer.send(self.args.producer['topic'], msg)
        try:
            record_matadata=future.get(timeout=10)
            logger.info("topic: {}, partition: {}, offset: {}".format(
                record_matadata.topic, record_matadata.partition, record_matadata.offset
            ))
        except KafkaError as e:
            logger.info("Kafka error: {}".format(e.args))

    def start(self):
        logger.info("\n*************{} node is running*************\n".format(self.args.node))
        for msg in self.consumer:
            try:
                data=self.processCore(msg)
                self.push_msg(data)
            except Exception as e:
                logger.info("kafka node running error: {}".format(e.args))
    