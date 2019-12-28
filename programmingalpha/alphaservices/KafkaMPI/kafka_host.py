# -*- UTF-8 -*-
import os
import tornado
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from typing import Optional, Awaitable

from programmingalpha import AlphaConfig, AlphaPathLookUp
from programmingalpha.alphaservices.KafkaMPI.kafka_node import AlpahaKafkaNode
#from multiprocessing import Process, Event
import json
from  programmingalpha.Utility import getLogger
import threading
import random
import queue
import asyncio

logger = getLogger(__name__)


class AlphaKafkaHost(AlpahaKafkaNode):
    def __init__(self,config_file):
        super().__init__(config_file)
        self.processed={} # Id-value dict data
        self.Id_queue_free=queue.Queue()
        self.Id_size=1000
        [self.Id_queue_free.put(i) for i in range(1000)]
        self.Id_set_used=set()

        def func_poll():
            for result in self.consumer:
                Id, data= result["Id"], result["data"]
                self.processed[Id].set_result(data)
                self.releaseId(Id)

        self.poll_thread=threading.Thread(target=func_poll )
        self.poll_thread.setDaemon(True)
        self.poll_thread.start()

    def processCore(self, data):
        raise NotImplementedError
    
    def useId(self):
        if self.Id_queue_free.empty():
            [self.Id_queue_free.put(i) for i in range(self.Id_size, self.Id_size*2)]
            self.Id_size*=2

        Id=self.Id_queue_free.get()
        self.Id_set_used.add(Id)
        return Id

    def releaseId(self, Id):
        self.Id_queue_free.put(Id)
        self.Id_set_used.remove(Id)
        
    async def getResult(self, Id):
        fu=asyncio.Future()
        #fu.set_result
        self.processed[Id]=fu
        res=await fu
        del self.processed[Id]
        return res


    def create_tornado_app(self):
        producer, topic= self.producer, self.topic
        useId=self.useId
        getResult=self.getResult
        
        class ALphaHandler(RequestHandler):
                
            @tornado.web.asynchronous
            @tornado.gen.coroutine
            def post(self):
                
                query_argument = json.loads(self.request.body)
                Id=useId()
                value={"Id":Id, "data":query_argument}
                #push to kafka
                producer.send(topic=topic, value=value)
                
                
                #result= asyncio.wait_for(getResult(Id), 100)
                try:
                    getResult(Id).send(None)
                except StopIteration as e:
                    result=e.value

                self.set_header('Content-type', 'application/json')
                
                self.write(result)


            def get(self):
                self.post()


        app = Application([
            (r"/methodCore", ALphaHandler)
        ])
        return app

    def start(self):
        app=self.create_tornado_app()

        http_server = HTTPServer(app)
        http_server.listen(port=self.args.port, address=self.args.listen_ip)
        logger.info("\n*************{} service is running({}:{})*************\n".format(self.args.ServiceName, self.args.listen_ip, self.args.port))
        IOLoop.current().start()


