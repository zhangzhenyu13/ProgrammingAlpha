# -*- UTF-8 -*-
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application

#from programmingalpha.DocSearchEngine.service import config
from programmingalpha.DocSearchEngine.service.http_handler import SearchHandler, KnowAlphaHandler

def search_server(config):
    app = Application([
        (r"/search", SearchHandler)
    ])
    http_server = HTTPServer(app)
    http_server.listen(config.http_port)
    print('Development server is running at http://10.1.1.32:%s/search' % config.http_port)
    print('Notice: You can use parameters: title, body, created_date, tags, size')
    IOLoop.current().start()


def know_alpha_server(config):
    app = Application([
        (r"/know_alpha", KnowAlphaHandler)
    ])
    http_server = HTTPServer(app)
    http_server.listen(config.http_port)
    print('Development server is running at http://10.1.1.32:%s/know_alpha' % config.http_port)
    print('Notice: You can use parameters: title and size')
    IOLoop.current().start()


if __name__ == '__main__':
    import programmingalpha
    import os

    config = programmingalpha.loadConfig(os.path.join(programmingalpha.ConfigPath, "docSearcherService.json"))

    know_alpha_server(config)
