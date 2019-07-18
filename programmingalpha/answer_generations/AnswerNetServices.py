#!/usr/bin/env python
import configargparse

from flask import Flask, jsonify, request
from onmt.translate import TranslationServer, ServerModelError
import os

from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
import programmingalpha

STATUS_OK = "ok"
STATUS_ERROR = "error"



class AnswerAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.translation_server = TranslationServer()
        self.translation_server.start( os.path.join(programmingalpha.ConfigPath, args.model_config) )

    def processCore(self, data):
        inputs = data
        out = {}
        try:
            translation, scores, n_best, times = self.translation_server.run(inputs)
            assert len(translation) == len(inputs)
            assert len(scores) == len(inputs)

            out = [[{"src": inputs[i]['src'], "tgt": translation[i],
                     "n_best": n_best,
                     "pred_score": scores[i]}
                    for i in range(len(translation))]]
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)
