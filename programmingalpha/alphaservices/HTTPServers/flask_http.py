from multiprocessing import Process, Event
import programmingalpha
import os
import json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AlphaHTTPProxy(Process):
    def __init__(self,config_file):
        super().__init__()
        self.args = programmingalpha.loadConfig( os.path.join( programmingalpha.ConfigPath, config_file ) )
        self.is_ready = Event()


    def processCore(self, data):
        raise NotImplementedError

    def create_flask_app(self):
        try:
            from flask import Flask, request
            from flask_compress import Compress
            from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
        except ImportError:
            raise ImportError('Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.')

        app = Flask(__name__)

        @app.route('/methodCore', methods=['POST', 'GET'])
        @as_json
        def encode_query():
            data = request.form if request.form else request.json
            if type(data)==str:
                data=json.loads(data)
            print("query data--->", data)

            try:
                logger.info('new request from %s' % request.remote_addr)
                return self.processCore(data)
            except Exception as e:
                logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def run(self):
        app = self.create_flask_app()
        self.is_ready.set()
        app.run(port=self.args.port, threaded=True, host=self.args.listen_ip)
