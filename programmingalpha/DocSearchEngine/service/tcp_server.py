# coding=utf-8
from tornado import ioloop, gen
from tornado.ioloop import IOLoop
from tornado.tcpserver import TCPServer
from tornado.iostream import StreamClosedError

from programmingalpha.DocSearchEngine.service import config


class TestTCPServer(TCPServer):
    @gen.coroutine
    async def handle_stream(self, stream, address):
        while True:
            try:
                msg = yield stream.read_bytes(20)
                print(msg, 'from', address)
                yield stream.write("Received your msg, from server".encode())
                if msg == 'exit':
                    stream.close()
            except StreamClosedError:
                print("-----Server Closed")
                break


if __name__ == '__main__':
    server = TestTCPServer()
    server.listen(config.options['tcp_server_port'])
    server.start()
    print("TCP Server starting...")
    ioloop.IOLoop.current().start()
