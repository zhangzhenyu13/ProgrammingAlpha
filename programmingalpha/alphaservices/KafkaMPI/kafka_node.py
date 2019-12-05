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
        self.producer=kafka.KafkaProducer(**self.args.producer)
        self.consumer=kafka.KafkaConsumer(**self.args.consumer)
        self.topic=self.args.topic

    def push_msg(self, data):
        msg=json.dumps(data)
        self.producer.send(topic=self.topic,value=msg)

    def pull_msg(self, time_out=5):
        msg=self.consumer.poll()
        data=json.loads(msg)
        return data

    def processCore(self, data):
        raise NotImplementedError

    def start(self):
        logger.info("\n*************{} node is running*************\n".format(self.args.node))
        time_out=5
        
        while True:
            try:
                data=self.pull_msg(time_out=time_out)
                data=self.processCore(data)
                self.push_msg(data)
                time_out=5
                time.sleep(time_out)
            except KafkaError as e:
                e.with_traceback()
                time_out*=2