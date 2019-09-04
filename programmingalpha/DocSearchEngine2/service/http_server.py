# -*- UTF-8 -*-
import os

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application

from programmingalpha.DocSearchEngine2.service.http_handler import SearchHandler
from programmingalpha import AlphaConfig, AlphaPathLookUp
from  programmingalpha.Utility import getLogger

logger = getLogger(__name__)

def search_server(config_file):
    config_args = AlphaConfig.loadConfig(os.path.join(AlphaPathLookUp.ConfigPath, config_file))
    app = Application([
        (r"/methodCore", SearchHandler, dict(config_args=config_args))
    ])
    http_server = HTTPServer(app)
    http_server.listen(config_args.port)
    logger.info("\n*************{} service is running in port {}*************\n".format(config_args.ServiceName, config_args.port))
    IOLoop.current().start()


if __name__ == '__main__':
    pass
