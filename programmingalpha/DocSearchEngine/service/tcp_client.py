# coding=utf-8

from tornado import ioloop, gen
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient

from programmingalpha.DocSearchEngine.service import config

tcp_client = TCPClient()

@gen.coroutine
def client():
    while True:
        try:
            stream = yield tcp_client.connect('10.1.1.32', config.options['tcp_server_port'])

            while True:
                msg = "Hello from client".encode()
                yield stream.write(msg)

        except StreamClosedError:
            print("Client Closed")
            break

if __name__ == '__main__':
    loop = IOLoop.current()
    loop.run_sync(client)
