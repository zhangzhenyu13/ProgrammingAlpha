import programmingalpha
from programmingalpha.tokenizers import get_tokenizer
from programmingalpha.alphaservices.HTTPServers.flask_http import AlphaHTTPProxy

class AnswerAlphaHTTPProxy(AlphaHTTPProxy):
    def __init__(self, config_file):
        AlphaHTTPProxy.__init__(self,config_file)
        args=self.args
        self.tokenizer=get_tokenizer(args.tokenizer)


    def processCore(self, data):
        inputs = data
        out = {}
        try:
            out=None
            if data["flow"]=="enc":
                text=data["text"]
                out=self.tokenizer.encode(text)
                out=" ".join(map(lambda id: str(id), out))
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

        return jsonify(out)
