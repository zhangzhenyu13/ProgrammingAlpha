import programmingalpha
from programmingalpha.tokenizers import get_tokenizer
from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy
from onmt.translate.translation_server import ServerModelError
from flask import jsonify
from programmingalpha.Utility import getLogger

logger=getLogger(__name__)
STATUS_ERROR = "error"

class TokenizerHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.tokenizer=get_tokenizer(model_path=programmingalpha.BertBaseUnCased,name=args.tokenizer)


    def processCore(self, data):
        out = {}
        logger.info("input --> {}".format(data))
        try:
            out=None
            if data["flow"]=="enc":
                text=data["text"]
                out=self.tokenizer.encode(text)
                #after encode, out is a str
                #out=" ".join(map(lambda id: str(id), out))
            elif data["flow"]=="dec":
                ids=data["ids"]
                if type(ids)==str:
                    ids=list(map(lambda id : int(id), ids.split()))
                out=self.tokenizer.decode(ids)
            else:
                raise ValueError("{} is not a executable flow".format(data["flow"]))
                
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR
        
        logger.info("return --> {}".format(out))
        return jsonify(out)
