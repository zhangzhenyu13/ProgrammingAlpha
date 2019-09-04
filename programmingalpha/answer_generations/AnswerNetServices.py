from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy

from flask import Flask, jsonify, request
from programmingalpha.answer_generations .translation_server import TranslationServer, ServerModelError
import os

from programmingalpha import AlphaPathLookUp

from.answer_alpha_input import E2EProcessor

STATUS_OK = "ok"
STATUS_ERROR = "error"



class AnswerAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.translation_server = TranslationServer()
        self.translation_server.start( os.path.join(AlphaPathLookUp.ConfigPath, args.model_config) )
        self.e2e_processor=E2EProcessor(config_file)


    def processCore(self, data):
        q_c_text = self.e2e_processor.processEnc(data["question"], data["posts"])
        query_data = [{"id":0,"src":q_c_text}]
        out = {}
        try:
            translation, scores, n_best, times = self.translation_server.run(query_data)
            assert len(translation) == len(query_data)
            assert len(scores) == len(query_data)
            
            out = {"src": q_c_text, "tgt": translation[0], "tgt_txt": self.e2e_processor.processDec(translation[0]),
                     "n_best": n_best,
                     "pred_score": scores[0]}
            
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)
